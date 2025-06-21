import copy
from typing import List

from .agents import *

import logging

logger = logging.getLogger("MMM.AgentManager")

class AgentManager:
    """
    AgentManager is a class that manages the loading and configuration of agents for time series analysis.
    
    Attributes:
        agent_type (str): The type of agent to be loaded.=
        config (dict): A dictionary containing the configuration for the agent.
        count_agents (int): The number of agents to be loaded.
        schema_RP (dict): A dictionary containing the schema for the agent.
    """
    
    type_agents = {
        "AgentPredTime": AgentPredTime,
        "AgentTradeTime": AgentTradeTime,
        "AgentNews": AgentNews,
        # "AgentPReadTimeMulti": AgentPReadTimeMulti,
        # "AgentPReadTimeMultiRP": AgentPReadTimeMultiRP
    }        

    def __init__(self, config: dict = {}, count_agents: int = 1, schema_RP: dict = {},
                 RP_I: bool = False):
        
        # self.agent_type = agent_type
        self.agent = {}
        self._multi_agent = False
        self.RP_I = RP_I

        self._init_config(count_agents, config, schema_RP)

    def _init_config(self, count_agents, config: dict, schema_RP: dict = {}):

        if len(config.get("agents", 0)) > 1:
            self._multi_agent = True

        elif not config.get("agents"):
            raise ValueError("Agents not found in config")

        if self._multi_agent:
            self.agent = self.load_multi_agent(count_agents, config, schema_RP)
        else:
            agent_config = config.get("agents")[0]

            if count_agents == 1:
                self.agent = self.create_agent(copy.deepcopy(agent_config), schema_RP)
                self.agent.set_id(1)
                self.agent.set_mode(agent_config.get("mod", "test"))
            else:
                self._multi_agent = True
                self.agent = []
                for i in range(count_agents):
                    agent = self.create_agent(copy.deepcopy(agent_config), schema_RP)
                    agent.set_id(i + 1)
                    agent.set_mode(agent_config.get("mod", "test"))
                    self.agent.append(agent)

    def create_agent(self, agent: dict, schema_RP: dict = {}) -> Agent:

        if agent["type"] == "AgentTradeTime":

            return self.get_agent(agent["type"])(
                name=agent["name"],
                indecaters=agent["indecaters"],
                timetravel=agent["timetravel"],
                discription=agent.get("discription", ""),
                model_parameters=agent.get("model_parameters", {}),
                data_normalize=agent.get("data_normalize", True),
                shema_RP=schema_RP,
                RP_I=self.RP_I,
                proffit_preddict_for_buy=agent.get("proffit_preddict_for_buy", 0.9),
                proffit_preddict_for_sell=agent.get("proffit_preddict_for_sell", 0.9)
            )
        
        return self.get_agent(agent["type"])(
                name=agent["name"],
                indecaters=agent["indecaters"],
                timetravel=agent["timetravel"],
                discription=agent.get("discription", ""),
                model_parameters=agent.get("model_parameters", {}),
                data_normalize=agent.get("data_normalize", True),
                shema_RP=schema_RP,
                RP_I=self.RP_I
            )
    
    @classmethod
    def get_agent(cls, agent_type: str) -> Agent:
        if agent_type in cls.type_agents:
            return cls.type_agents[agent_type]
        else:
            raise ValueError(f"Agent {agent_type} not found in available models.")
        
    def get_agents(self) -> List[Agent] | Agent:
        return self.agent

    def load_multi_agent(self, count_agents: int, config_model:dict, schema_RP: dict) -> List[Agent]:
        
        # logger.info(f"Loading multi agent: {self.agent_type}")

        if len(config_model.get("agents")) > 1:
            self._multi_agent = True

        agents = {}

        for agent_config in config_model.get("agents"):
            for i in range(count_agents):
                agent = self.create_agent(copy.deepcopy(agent_config), schema_RP)
                agent.set_id(i + 1)
                agent.set_mode(agent_config.get("mod", "test"))
                agents.setdefault(agent.get_timetravel(), [])
                agents[agent.get_timetravel()].append(agent)
        
        return agents
    
    