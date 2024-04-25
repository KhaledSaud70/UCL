import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

model_dict = {
    'resnet18': [models.resnet18, 512],
    'resnet34': [models.resnet34, 512],
    'resnet50': [models.resnet50, 2048],
    'resnet101': [models.resnet101, 2048],
    # TODO add more backbones
}


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet18', head='mlp', feat_dim=128, num_classes=10,
                 train_on_head=True):
        super().__init__()

        model_fun, dim_in = model_dict[name]
        feat_dim = min(dim_in, feat_dim)
        self.feat_dim = feat_dim
        
        self.encoder = model_fun(weights='DEFAULT')

        if hasattr(self.encoder, 'fc'):
            self.encoder.fc = nn.Identity()

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

        self.fc = nn.Linear(dim_in, num_classes)
        if train_on_head:
            self.head = nn.Identity()
            # self.fc = nn.Linear(feat_dim, num_classes)
        
        self.train_on_head = train_on_head

    def forward(self, x):
        feat = self.encoder(x)
        proj = F.normalize(self.head(feat), dim=1)
        
        if self.train_on_head:
            logits = self.fc(proj)
        else:
            logits = self.fc(feat)
        
        return proj, feat, logits


class CEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10, normalize=False):
        super().__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if hasattr(self.encoder, 'fc'):
            self.encoder.fc = nn.Identity()
        self.fc = nn.Linear(dim_in, num_classes)
        self.normalize = normalize

    def forward(self, x):
        feats = self.encoder(x)
        if self.normalize:
            feats = F.normalize(feats, dim=1)
        return self.fc(feats), feats