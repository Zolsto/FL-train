import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torchvision.models import EfficientNet_B0_Weights
from collections import OrderedDict
from modules.federated.utils import get_weights, set_weights

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes: int=6, fine_tune_layers: bool=True, weights: list=None, premodel: str=None):
        super().__init__()
        self.model = models.efficientnet_b0(weights=None, progress=False)

        if fine_tune_layers == True:
            for block in self.model.features.parameters():
                block.requires_grad = True

        b0_out = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=b0_out, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=num_classes)
        )
        if weights is not None:
            param_names = [name for name in self.model.state_dict().keys()]
            state_dict = {}
            for name, array in zip(param_names, weights):
                state_dict[name] = torch.from_numpy(array)
            
            self.model.load_state_dict(state_dict)
        elif premodel is not None:
            new_dict = OrderedDict()
            pretrained_dict = torch.load(f"../pretrain/modelli/{premodel}/best_{premodel}.pt", weights_only=True)
            # Remove 'model.' (6 char) from weights name in the pretrained state_dict
            for k, v in pretrained_dict.items():
                if k.startswith('model.'):
                    name = k[6:]
                else:
                    name = k
                new_dict[name] = v
            self.model.load_state_dict(new_dict)
            print(f"Model {premodel} imported successfully.")

    def forward(self, x):
        #return self.model(x)
        x = self.model(x)
        return {"logits": x}

    def extract_backbone_features(self, x, layer_idx=None):
        if layer_idx is None:
            return self.model.features(x)

        for idx, layer in enumerate(self.model.features):
            x = layer(x)
            if layer_idx is not None and idx == layer_idx:
                break
        return x
