import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import timm
from timm.models.registry import register_model


class SupConCLIP(nn.Module):
    def __init__(self, n_classes=10, **kwargs):
        super().__init__()
        self.encoder = timm.create_model("hf_hub:timm/vit_base_patch32_clip_224.openai", pretrained=True, num_classes=0)
        print(self.encoder)
        for name, param in self.encoder.named_parameters():
            if 'attn' not in name:
                param.requires_grad = False
        
        self.encoder.head = nn.Identity()
        self.fc = nn.Linear(768, n_classes)

    def forward(self, x):
        feat = self.encoder(x)
        print(feat.shape, self.encoder(x).shape)
        proj = F.normalize(feat, dim=-1)
        logits = self.fc(feat)
        return proj, feat, logits


@register_model
def clip(pretrained=False, **kwargs):
    model = SupConCLIP(**kwargs)
    return model


if __name__ == "__main__":
    from timm.models import create_model

    model = create_model('clip', n_classes=100)
    print(model)
    print("Model parameters:", f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    # print(transforms)

    from PIL import Image
    img = Image.open('/home/khaled/workspace/projects/shelf-monitoring/src/data/camera_1/images/ref1-mis1_jpg.rf.d1ecd4f81f8c882b0372add0dfc9ea7f.jpg')
    print(transforms(img).unsqueeze(0).shape)
    output = model(transforms(img).unsqueeze(0))
    for o in output:
        print(o.shape)
