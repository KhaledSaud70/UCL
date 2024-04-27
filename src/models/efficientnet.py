import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

model_dict = {
    'efficientnet_v2_s': [models.efficientnet_v2_s, 1280],
    'efficientnet_v2_m': [models.efficientnet_v2_m, 1280],
    'efficientnet_v2_l': [models.efficientnet_v2_l, 1280],
}


class SupConEfficientNet(nn.Module):
    def __init__(self, name='efficientnet_v2_s', num_classes=10):
        super().__init__()

        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(weights='DEFAULT')
        
        if hasattr(self.encoder, 'classifier'):
            self.encoder.classifier = nn.Identity()

        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        proj = F.normalize(feat, dim=1)
        logits = self.fc(proj)
        
        return proj, feat, logits
    

if __name__ == "__main__":
    model = SupConEfficientNet()
    print(model)
    print(model(torch.randn(1, 3, 224, 224)))