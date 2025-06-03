import copy
from typing import List

from core import data_manager
from .agent_pread_time import AgentPReadTime, Agent
from .agent_trade_time import AgentTradeTime

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
    
    _agents = {
        "AgentPReadTime": AgentPReadTime,
        "AgentTradeTime": AgentTradeTime,
        # "AgentPReadTimeMulti": AgentPReadTimeMulti,
        # "AgentPReadTimeMultiRP": AgentPReadTimeMultiRP
    }        

    def __init__(self, agent_type: str, config: dict = {}, count_agents: int = 1, schema_RP: dict = {},
                 RM_I: bool = False):
        self.agent_type = agent_type
        self.agent = {}
        self._multi_agent = False
        self.RM_I = RM_I

        self._init_config(count_agents, config, schema_RP)

    def _init_config(self, count_agents, config: dict, schema_RP: dict = {}):

        if len(config.get("agents")) > 1:
            self._multi_agent = True

        elif not config.get("agents"):
            raise ValueError("Agents not found in config")

        if self._multi_agent:
            self.agent = self.load_multi_agent(count_agents, schema_RP)
        else:
            agent_config = config.get("agents")[0]

            if count_agents == 1:
                self.agent = self.create_agent(copy.deepcopy(agent_config), schema_RP)
            else:
                self._multi_agent = True
                self.agent = []
                for i in range(count_agents):
                    agent = self.create_agent(copy.deepcopy(agent_config), schema_RP)
                    agent.set_id(i + 1)
                    self.agent.append(agent)

    def create_agent(self, agent: dict, schema_RP: dict = {}) -> Agent:

        return self.get_agent(agent["type"])(
                name=agent["name"],
                indecaters=agent["indecaters"],
                timetravel=agent["timetravel"],
                discription=agent["discription"],
                model_parameters=agent["model_parameters"],
                shema_RP=schema_RP,
                RM_I=self.RM_I
            )
    
    @classmethod
    def get_agent(cls, agent_type: str) -> Agent:
        if agent_type in cls._agents:
            return cls._agents[agent_type]
        else:
            raise ValueError(f"Agent {agent_type} not found in available models.")
        
    def get_agents(self) -> List[Agent]:
        return self.agent
    
    def load_multi_agent(self, count_agents: int, schema_RP) -> List[Agent]:
        logger.info(f"Loading multi agent: {self.agent_type}")
        config_model = data_manager.get_model_config(self.agent_type)

        if len(config_model.get("agents")) > 1:
            self._multi_agent = True

        agents = {}

        for agent_config in config_model.get("agents"):
            for i in range(count_agents):
                agent = self.create_agent(copy.deepcopy(agent_config), schema_RP)
                agent.set_id(i + 1)
                agents.setdefault(agent.get_timetravel(), [])
                agents[agent.get_timetravel()].append(agent)
        
        return agents
    
    