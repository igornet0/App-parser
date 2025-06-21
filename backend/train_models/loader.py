import numpy as np
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.amp import autocast
from typing import List, Dict, Any, Union, Generator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tqdm import tqdm
from accelerate import Accelerator

from .parsing_schem import parsing_json_schema
from core import data_helper
from backend.MMM import (Agent,
                         AgentManager)

from backend.Dataset import LoaderTimeLine
from .transform_data import TimeSeriesTransform

import logging

logger = logging.getLogger("train_models.loader")

class Loader:

    def __init__(self, agent_type: str = "", model_type: str = "MMM"):
        self.agent_type = agent_type
        self.model_type = model_type
        self._multi_agent = False
        accelerator = Accelerator()
        self.device = torch.device(accelerator.device)

    def load_model(self, count_agents: int = 1) -> AgentManager:
        logger.info(f"Loading Agent: {self.agent_type}|{self.model_type} with count: {count_agents}")
        config_model = data_helper.get_model_config(self.agent_type, self.model_type)

        RP_I = config_model.get("RANDOM_INDICATETS", False)

        try:
            agent_manager = AgentManager(config=config_model,
                                         count_agents=count_agents,
                                         schema_RP=self._load_schema(self.agent_type, self.model_type),
                                         RP_I=RP_I)
        except Exception as e:
            logger.error(f"Error loading agent: {self.agent_type} - {str(e)}")
            return None

        return agent_manager

    @staticmethod
    def _load_schema(agent_type, model_type) -> Union[Dict[str, Generator[None, None, Any]], None]:
        
        config_model = data_helper.get_model_config(agent_type, model_type)

        schema = config_model.get("schema")
        if not schema:
            return None
        
        logger.info(f"Loading schema for: {agent_type}")
        if not schema:
            raise ValueError(f"Schema not found for agent type: {agent_type}")

        return parsing_json_schema(schema)

    def load_agent_data(self, loaders: List[LoaderTimeLine], agent, batch_size, mixed) -> TimeSeriesTransform:        
        return TimeSeriesTransform(loaders=loaders, 
                                   agent=agent, 
                                   batch_size=batch_size,
                                   mixed=mixed)

    def _train_single_agent(self, agent: Agent, loaders: List[LoaderTimeLine], epochs, batch_size, 
                        base_lr, weight_decay, patience, mixed, mixed_precision):
        
        test_tensor = torch.randn(2, 2).to(self.device)
        try:
            print("MPS test:", test_tensor @ test_tensor.T)
        except Exception as e:
            print(f"MPS error: {e}")
        
        is_cuda = self.device.type == 'cuda'
        is_mps = self.device.type == 'mps'

        # Инициализация модели
        model = agent.model.to(self.device)

        if torch.__version__ >= "2.0" and is_cuda:
            model = torch.compile(model)

        # Автоматически отключаем mixed_precision для CPU
        effective_mp = mixed_precision and is_cuda  # MPS имеет ограниченную поддержку
        if is_mps and mixed_precision:
            print("⚠️ Mixed precision на MPS может работать некорректно. Рекомендуется использовать fp32")
        
        # Загрузка данных
        loader = self.load_agent_data(loaders, agent, batch_size, mixed)


        optimizer, scheduler, scaler = agent.init_model_to_train(base_lr, weight_decay, 
                                  is_cuda, effective_mp, patience)
        
        best_loss = float('inf')
        history_loss = []
        history_state = []
        version = 0
        # Цикл эпох
        for epoch in range(epochs):
            epoch_loss = 0.0
            agent.model.train()
            start_time = time.time()

            pbar = tqdm(enumerate(loader), 
                        total=len(loader), 
                        desc=f"Epoch {epoch+1}/{epochs} | Agent {agent.id}| ",
                        bar_format="{l_bar}|{bar:20}|{r_bar}", 
                        leave=False)

            # Итерация по батчам с прогресс-баром
            for batch_idx, batch in pbar:
                
                optimizer.zero_grad()
                args = [arg.to(self.device) for arg in batch if arg is not None]
                x, y, *_ = args

                with autocast(device_type=self.device.type, enabled=effective_mp and (is_cuda or is_mps)):
                # with torch.no_grad():
                    
                    outputs = agent.trade([x, *_])
                
                    # print(outputs.shape, y.shape)
                    loss = agent.loss_function(outputs, y)
                    # print(f"Output min: {outputs.min().item()}, max: {outputs.max().item()}") 
                
                assert torch.isnan(loss).sum() == 0, "Найдены NaN в loss"
                # print(outputs.shape, y.shape)
                # print(outputs)
                # print(f"Loss: {loss.item():.4f} | Batch size: {x.size(0)}")

                # Обновление градиентов
                if effective_mp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)  # Важно для клиппинга при использовании scaler
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # print(f"Loss: {loss.item():.4f} | Batch size: {x.size(0)} | Output shape: {outputs.shape}")
                    loss.backward()  # Обновление градиентов
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Клиппинг
                    optimizer.step()
                
                current_loss = loss.item()
                epoch_loss += current_loss
                pbar.set_postfix({
                        'loss': f"{current_loss:.4f}",  # Текущий loss батча
                        'avg_loss': f"{epoch_loss/(batch_idx+1):.4f}",  # Средний loss
                        'lr': f"{optimizer.param_groups[0]['lr']:.2e}"  # Learning rate
                    })

            # Статистика эпохи
            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)  
            history_loss.append(avg_loss)
            lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - start_time
            version += 1
            agent.set_version(version)
            
            # Форматированный вывод
            status = "🟢 Improved!" if avg_loss < best_loss else "🟡 No improvement"
            print(
                f"\nAgent {agent.id} | Epoch {epoch+1:02d}/{epochs} "
                f"| Loss: {avg_loss:.4f} ({best_loss:.4f}) Hisstory AVG loss ({sum(history_loss)/len(history_loss):.4f}) "
                f"| LR: {lr:.2e} | Time: {epoch_time:.1f}s\n"
                f"{status}{' | ⏹ Early Stopping' if patience <= history_state.count(False) else ''}"
            )
            
            # Ранняя остановка
            if avg_loss < best_loss:
                best_loss = avg_loss
                history_state.append(True)
                if len(history_state) % 10 == 0:

                    agent.save_model(epoch=epoch, 
                                     optimizer=optimizer, 
                                     scheduler=scheduler, 
                                     best_loss=best_loss)
            else:
                history_state.append(False)
                if sum(history_state[-patience:]) == 0:
                    print(f"🛑 Early stopping triggered for Agent {agent.id}")
                    break

        # Загрузка лучших весов
        # model.load_state_dict(torch.load(agent.weights_path))
        print(f"\n⭐ Agent {agent.id} Best Loss: {best_loss:.4f}\n")
        
        agent.save_json(
            epoch=epoch, 
            history_loss=history_loss, 
            best_loss=best_loss, 
            base_lr=base_lr, 
            batch_size=batch_size, 
            weight_decay=weight_decay)

        return history_loss

    def train_model(self, loaders: List[LoaderTimeLine], agent_manager: AgentManager, 
                    mixed: bool = True):
        
        logger.info("🚀 Starting ensemble training")
        config_train = data_helper.get_model_config(self.agent_type, self.model_type)
        
        # Конфигурация обучения
        epochs = config_train["epochs"]
        batch_size = config_train["batch_size"]
        num_workers = config_train.get("num_workers", 4)
        base_lr = config_train["lr"]
        weight_decay = config_train["weight_decay"]
        patience = config_train["patience"]
        mixed_precision = config_train.get("mixed_precision", False)
        
        # Обучение с прогресс-баром
        for agent in (pbar := tqdm(agent_manager.agent, desc="Agents")):
            pbar.set_description(f"🏋️ Training Agent {agent.id}")
            self._train_single_agent(agent, loaders, epochs, batch_size, 
                                base_lr, weight_decay, patience, mixed, 
                                mixed_precision)

        logger.info("✅ All agents trained successfully")
        
    def train_multi_agent(self, loaders, bath_size=10):
        logger.info("Training multi-agent model")
        for agent in self.agent:
            self.train_model(loaders, bath_size)