import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from timm.models.registry import register_model


model_dict = {
    'resnet18': [models.resnet18, 512],
    'resnet34': [models.resnet34, 512],
    'resnet50': [models.resnet50, 2048],
    'resnet101': [models.resnet101, 2048],
}


class SupConResNet(nn.Module):
    def __init__(self, name='resnet18', n_classes=10, trainable_layers=[], **kwargs):
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
        self.fc = nn.Linear(dim_in, n_classes)

    def forward(self, x):
        feat = self.encoder(x)
        proj = F.normalize(feat, dim=1)
        logits = self.fc(proj)
        
        return proj, feat, logits

    

@register_model
def resnet18(pretrained=False, **kwargs):
    model = SupConResNet(name='resnet18', **kwargs)
    return model


@register_model
def resnet34(pretrained=False, **kwargs):
    model = SupConResNet(name='resnet34', **kwargs)
    return model

@register_model
def resnet50(pretrained=False, **kwargs):
    model = SupConResNet(name='resnet50', **kwargs)
    return model

@register_model
def resnet101(pretrained=False, **kwargs):
    model = SupConResNet(name='resnet101', **kwargs)
    return model


if __name__ == "__main__":
    from timm.models import create_model
    import torch

    model = create_model('resnet18', num_classes=100)
    print(model)
    
    input_size = (1, 3, 224, 224)

    sample_input = torch.randn(input_size)
    proj, feat, logits = model(sample_input)
    print("Projection Shape:", proj.shape)
    print("Feature Shape:", feat.shape)
    print("Logits Shape:", logits.shape)
