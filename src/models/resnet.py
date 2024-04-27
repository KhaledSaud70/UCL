import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

model_dict = {
    'resnet18': [models.resnet18, 512],
    'resnet34': [models.resnet34, 512],
    'resnet50': [models.resnet50, 2048],
    'resnet101': [models.resnet101, 2048],
}


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet18', num_classes=10, trainable_layers=[]):
        super().__init__()

        model_fun, dim_in = model_dict[name]
        self.feat_dim = dim_in
        
        self.encoder = model_fun(weights='DEFAULT')

        if trainable_layers:
            for name, param in self.encoder.named_parameters():
                if not any(name.startswith(layer) for layer in trainable_layers):
                    param.requires_grad = False

        if hasattr(self.encoder, 'fc'):
            self.encoder.fc = nn.Identity()

        
        self.head = nn.Identity()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        proj = F.normalize(feat, dim=1)
        logits = self.fc(proj)
        
        return proj, feat, logits
    

if __name__ == "__main__":
    model = SupConResNet(trainable_layers=[])
    # print(model)
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)
