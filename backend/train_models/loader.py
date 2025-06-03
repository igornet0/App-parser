import numpy as np
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import List, Dict, Any, Union, Generator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tqdm import tqdm
from accelerate import Accelerator

from .parsing_schem import parsing_json_schema
from core import data_manager
from backend.MMM import (DataGenerator,
                         Agent,
                         AgentPReadTime,
                         AgentManager)

from backend.Dataset import Dataset, DatasetTimeseries, LoaderTimeLine
from .transform_data import TimeSeriesTransform

import logging

logger = logging.getLogger("train_models.loader")

class Loader:

    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self._multi_agent = False
        accelerator = Accelerator()
        self.device = torch.device(accelerator.device)
        self.scaler = GradScaler()

    def load_model(self, count_agents: int = 1) -> AgentManager:
        logger.info(f"Loading Agent: {self.agent_type}")
        config_model = data_manager.get_model_config(self.agent_type)

        RM_I = config_model.get("RANDOM_INDICATETS", False)

        try:
            agent_manager = AgentManager(agent_type=self.agent_type,
                                         config=config_model,
                                         count_agents=count_agents,
                                         schema_RP=self._load_schema(self.agent_type),
                                         RM_I=RM_I)
        except Exception as e:
            logger.error(f"Error loading agent: {self.agent_type} - {str(e)}")
            return None

        return agent_manager
    
    @staticmethod
    def vectorized_quantile_loss(predictions, targets):
        """
        Комбинированная функция потерь для финансовых прогнозов
        Args:
            predictions: Tensor [batch_size, pred_len, 2]
                - predictions[..., 0]: price predictions
                - predictions[..., 1]: volatility predictions
            targets: Tensor [batch_size, pred_len]
        
        Returns:
            loss: Combined loss value
        """
        print("vectorized_quantile_loss")
        print(f"predictions.shape: {predictions.shape}\ntargets.shape: {targets.shape}")
        price_pred = predictions[..., 0]
        direction_pred = predictions[..., 2]  # Добавить выход для направления
        
        # MSE для цены
        price_loss = F.mse_loss(price_pred, targets)
        
        # Binary cross-entropy для направления
        true_direction = (targets[:,1:] > targets[:,:-1]).float()
        direction_loss = F.binary_cross_entropy_with_logits(
            direction_pred[:,:-1], true_direction)
        
        return 0.7*price_loss + 0.3*direction_loss

    @staticmethod
    def _load_schema(agent_type) -> Union[Dict[str, Generator[None, None, Any]], None]:
        
        config_model = data_manager.get_model_config(agent_type)
        schema = config_model.get("schema")
        if not schema:
            return None
        
        logger.info(f"Loading schema for: {agent_type}")
        if not schema:
            raise ValueError(f"Schema not found for agent type: {agent_type}")

        return parsing_json_schema(schema)

    def prepare_datasets(dataframes: dict[str, DatasetTimeseries], window_size=30, test_size=0.2, target_col='close'):
        """
        Подготавливает данные из нескольких датасетов для обучения нейронной сети.

        Параметры:
        - dataframes: список датафреймов с колонками [datetime, open, max, min, close, value].
        - window_size: размер временного окна (количество шагов в последовательности).
        - test_size: доля данных для тестового набора (по времени).
        - target_col: целевая колонка для предсказания (по умолчанию 'close').

        Возвращает:
        - X_train, X_test, y_train, y_test: массивы numpy для обучения и тестирования.
        """
        all_X_train = []
        all_X_test = []
        all_y_train = []
        all_y_test = []

        dataframes = {data_manager.coin_list_one_hot}

        for coin, dt in dataframes.items():
            dt.sort("datetime")

        # Определяем все возможные coin_id и обучаем OneHotEncoder
        num_coins = len(dataframes)
        encoder = OneHotEncoder(sparse=False)
        encoder.fit([[i] for i in range(num_coins)])
        
        for coin_id, df in enumerate(dataframes):
            # Сортировка по времени
            df = df.sort_values('datetime')
            
            # Заполнение пропусков
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Кодируем coin_id в one-hot
            coin_id_arr = np.array([[coin_id]] * len(df))
            coin_ids_encoded = encoder.transform(coin_id_arr)
            
            # Нормализация числовых признаков
            numeric_cols = ['open', 'max', 'min', 'close', 'value']
            scaler = MinMaxScaler()
            scaled_numeric = scaler.fit_transform(df[numeric_cols])
            
            # Объединяем нормализованные данные с one-hot encoded coin_id
            processed_data = np.hstack([scaled_numeric, coin_ids_encoded])
            
            # Создаем последовательности (окна)
            X = []
            y = []
            for i in range(len(processed_data) - window_size):
                X.append(processed_data[i:i + window_size])
                target_idx = numeric_cols.index(target_col)
                y.append(scaled_numeric[i + window_size, target_idx])  # Используем масштаб целевой переменной
                
            X = np.array(X)
            y = np.array(y)
            
            # Разделение на train/test (последние `test_size` данных)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            all_X_train.append(X_train)
            all_X_test.append(X_test)
            all_y_train.append(y_train)
            all_y_test.append(y_test)
        
        # Объединяем данные всех монет
        X_train = np.concatenate(all_X_train, axis=0)
        X_test = np.concatenate(all_X_test, axis=0)
        y_train = np.concatenate(all_y_train, axis=0)
        y_test = np.concatenate(all_y_test, axis=0)
        
        return X_train, X_test, y_train, y_test
    
    def _train_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0.0
        
        for x, y, time_x in train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            time_x = time_x.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=self.amp):
                preds = self.model(x, time_x)
                loss = self.vectorized_quantile_loss(preds, y)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * x.size(0)
        
        return total_loss / len(train_loader.dataset)
    
    def train(self):
        train_gen, val_gen = self.data_manager.get_generators()
        
        train_loader = self._get_dataloader(train_gen)
        val_loader = self._get_dataloader(val_gen)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'],
            fused=True  # Использование fused implementation
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config['lr'],
            total_steps=self.config['epochs'] * len(train_loader),
            pct_start=0.3
        )

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config['epochs']):
            train_loss = self._train_epoch(train_loader, optimizer)
            val_loss = self._evaluate(val_loader)
            
            # Обновление планировщика
            scheduler.step()
            
            # Логирование и early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print("Early stopping triggered")
                    break

            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    @staticmethod
    def train_agent(loader, agent):
        logger.info("🚀 Starting ensemble training")

    @torch.no_grad()
    def _evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        
        for x, y, time_x in dataloader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            time_x = time_x.to(self.device, non_blocking=True)
            
            with autocast(enabled=self.amp):
                preds = self.model(x, time_x)
                loss = self.vectorized_quantile_loss(preds, y)
            
            total_loss += loss.item() * x.size(0)
        
        return total_loss / len(dataloader.dataset)

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
        
        # Оптимизатор и планировщик
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr * agent.lr_factor,
            weight_decay=weight_decay,
            fused=is_cuda
        )
        
        # Загрузка данных
        loader = self.load_agent_data(loaders, agent, batch_size, mixed)

        # Автоматически отключаем mixed_precision для CPU
        effective_mp = mixed_precision and is_cuda  # MPS имеет ограниченную поддержку
        if is_mps and mixed_precision:
            print("⚠️ Mixed precision на MPS может работать некорректно. Рекомендуется использовать fp32")
        
        # Инициализация GradScaler только при необходимости
        scaler = GradScaler(enabled=effective_mp)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            "min",
            factor=0.5,
            patience=patience
        )
        
        best_loss = float('inf')
        history_loss = []
        history_state = []
        
        # Цикл эпох
        for epoch in range(epochs):
            epoch_loss = 0.0
            model.train()
            start_time = time.time()

            pbar = tqdm(enumerate(loader), 
                        total=len(loader), 
                        desc=f"Epoch {epoch+1}/{epochs} | Agent {agent.id}| ",
                        bar_format="{l_bar}|{bar:20}|{r_bar}", 
                        leave=False)
            
            # Итерация по батчам с прогресс-баром
            for batch_idx, batch in pbar:
                x, y, time_x = batch

                # Перенос данных на устройство
                x = x.to(self.device)
                y = y.to(self.device)
                time_x = time_x.to(self.device)
                
                optimizer.zero_grad()
                with autocast(device_type=self.device.type, enabled=effective_mp and (is_cuda or is_mps)):
                    assert torch.isnan(x).sum() == 0, "Найдены NaN во входных данных X"
                    assert torch.isinf(time_x).sum() == 0, "Найдены inf в Time"
                    assert torch.isnan(time_x).sum() == 0, "Найдены NaN в Time"
                    
                    outputs = agent.trade([x, time_x])

                    loss = agent.loss_function(outputs, y)
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
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Клиппинг
                    optimizer.step()
                
                current_loss = loss.item()
                scheduler.step(current_loss)
                epoch_loss += current_loss
                pbar.set_postfix({
                        'loss': f"{current_loss:.4f}",  # Текущий loss батча
                        'avg_loss': f"{epoch_loss/(batch_idx+1):.4f}",  # Средний loss
                        'lr': f"{optimizer.param_groups[0]['lr']:.2e}"  # Learning rate
                    })

            # Статистика эпохи
            avg_loss = epoch_loss / len(loader)
            history_loss.append(avg_loss)
            lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - start_time
            
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
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    filename = data_manager["models pth"] / f"{agent.name}_{agent.id}_{timestamp}_{id(agent)}.pth"

                    agent.save_model(epoch=epoch, 
                                     optimizer=optimizer, 
                                     scheduler=scheduler, 
                                     best_loss=best_loss, 
                                     filename=filename)
            else:
                history_state.append(False)
                if sum(history_state[-patience:]) == 0:
                    print(f"🛑 Early stopping triggered for Agent {agent.id}")
                    break

        # Загрузка лучших весов
        # model.load_state_dict(torch.load(agent.weights_path))
        print(f"\n⭐ Agent {agent.id} Best Loss: {best_loss:.4f}\n")
        
        training_info = {
            'epochs_trained': epoch + 1,
            'loss_history': history_loss,
            'best_loss': best_loss,
            'indecaters': agent.get_indecaters(),
            "seq_len": agent.model_parameters["seq_len"],
            "pred_len": agent.model_parameters["pred_len"],
            "d_model": agent.model_parameters.get("d_model", 128),
            "n_heads": agent.model_parameters.get("n_heads", 4),
            "emb_month_size": agent.model_parameters.get("emb_month_size", 8),
            "emb_weekday_size": agent.model_parameters.get("emb_weekday_size", 4),
            "lstm_hidden": agent.model_parameters.get("lstm_hidden", 256),
            "num_layers": agent.model_parameters.get("num_layers", 2),
            "dropout": agent.model_parameters.get("dropout", 0.2),
            'hyperparams': {
                'base_lr': base_lr,
                'batch_size': batch_size,
                'weight_decay': weight_decay
            }
        }
    
        # Сохраняем в JSON
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = data_manager["models configs"] / f"agent_{agent.id}_training_log_{timestamp}_{id(agent)}.json"
        with open(filename, 'w') as f:
            json.dump(training_info, f, indent=2)

        return history_loss

    def train_model(self, loaders: List[LoaderTimeLine], agent_manager: AgentManager, 
                    mixed: bool = True):
        
        logger.info("🚀 Starting ensemble training")
        config_train = data_manager.get_model_config(self.agent_type)
        
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
            agent.train_model(loaders, bath_size)

    def save_model(self, file_path: str):
        # Placeholder for saving the model
        logger.info(f"Saving model to: {file_path}")
        # Here you would typically save the model to disk
        # For example:
        # torch.save(self.model, file_path)
        pass