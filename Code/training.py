

# %% IMPORTS

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


# %% LOAD PRETRAINED RESNET-18

# torchvision version: 0.8.1
model_resnet18 = torch.hub.load(repo_or_dir = 'pytorch/vision:v0.8.1',
                                model = 'resnet18', pretrained = True)
# TODO Is it necessary to specify more parameters when loading the model? --> w.r.t. image size
type(model_resnet18)  # torchvision.models.resnet.ResNet
model_resnet18.train()
model_resnet18.eval()

# %% CUSTOM NEURAL NETWORK CLASS


class SpeciesClassifier(nn.Module):
    def __init__(self):
        super(SpeciesClassifier, self).__init__()

        def forward(self, image):
            pass




# %% MAIN CODE
# TODO do a train, validate, test split



