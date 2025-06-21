from typing import List
import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, meta_input_size, meta_output_size, *models: List[nn.Module]):
        super().__init__()

        # Замораживаем базовые модели
        for model in models:
            for param in model.parameters():
                param.requires_grad = False
                
        self.meta_classifier = nn.Sequential(
            nn.Linear(meta_input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, meta_output_size)
        )
    
    def forward(self, x_ts, x_news, x_state):
        out1 = self.model1(x_ts)
        out2 = self.model2(x_news)
        out3, _ = self.model3(x_state)
        
        combined = torch.cat([out1, out2, out3], dim=1)
        return self.meta_classifier(combined)
    