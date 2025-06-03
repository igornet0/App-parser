from typing import Dict, Any

from .agent import Agent

class PricePredictorModel:
    pass

class AgentTradeTime(Agent):
    
    model = PricePredictorModel

    def _init_model(self, model_parameters: Dict[str, Any]) -> PricePredictorModel:
        """
        Initializes the model for the agent.

        Args:
            model_parameters (dict): The configuration model for the agent containing parameters such as
                input features, sequence length, prediction length, model dimension, number of heads,
                and dropout rate.

        Returns:
            PricePredictorModel: An instance of the PricePredictorModel class.

        """
        n_indicators = sum(self.get_shape_indecaters().values())
        seq_len = model_parameters.get("seq_len", 30)
        pred_len = model_parameters.get("pred_len", 5)
        d_model = model_parameters.get("d_model", 64)
        n_heads = model_parameters.get("n_heads", 4)
        dropout = model_parameters.get("dropout", 0.1)

        self.model = PricePredictorModel(n_indicators=n_indicators,
                                             seq_len=seq_len,
                                             pred_len=pred_len,
                                             d_model=d_model,
                                             n_heads=n_heads,
                                             dropout=dropout,)

        return self.model