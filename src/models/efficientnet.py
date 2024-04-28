import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from torchvision import models

model_dict = {
    'efficientnet_v2_s': [models.efficientnet_v2_s, 1280],
    'efficientnet_v2_m': [models.efficientnet_v2_m, 1280],
    'efficientnet_v2_l': [models.efficientnet_v2_l, 1280],
}


class SupConEfficientNet(nn.Module):
    def __init__(self, name='efficientnet_v2_s', n_classes=10, **kwargs):
        super().__init__()

        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(weights='DEFAULT')
        
        if hasattr(self.encoder, 'classifier'):
            self.encoder.classifier = nn.Identity()

        self.head = nn.Identity()
        self.fc = nn.Linear(dim_in, n_classes)

    def forward(self, x):
        feat = self.encoder(x)
        proj = F.normalize(feat, dim=1)
        logits = self.fc(proj)
        
        return proj, feat, logits


@register_model
def efficientnet_v2_s(pretrained=False, **kwargs):
    model = SupConEfficientNet(name='efficientnet_v2_s', **kwargs)
    return model

@register_model
def efficientnet_v2_m(pretrained=False, **kwargs):
    model = SupConEfficientNet(name='efficientnet_v2_m', **kwargs)
    return model

@register_model
def efficientnet_v2_l(pretrained=False, **kwargs):
    model = SupConEfficientNet(name='efficientnet_v2_l', **kwargs)
    return model


if __name__ == "__main__":
    from timm.models import create_model

    model = create_model('efficientnet_v2_s', n_classes=100)
    print(model)
    print(model(torch.randn(1, 3, 224, 224)))