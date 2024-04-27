import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import timm

class SupConCLIP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = timm.create_model("hf_hub:timm/vit_base_patch32_clip_224.openai", pretrained=True, num_classes=0)
        for n, p in self.model.named_parameters():
            if 'attn' not in n:
                p.requires_grad = False

        print("Model parameters:", f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
        self.model.head = nn.Identity()
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        feat = self.model(x)
        proj = F.normalize(feat, dim=-1)
        logits = self.fc(feat)
        return proj, feat, logits
    

if __name__ == "__main__":
    model = SupConCLIP()
    print("Model parameters:", f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    # print(transforms)

    from PIL import Image
    img = Image.open('/home/khaled/workspace/projects/shelf-monitoring/src/data/camera_1/images/ref1-mis1_jpg.rf.d1ecd4f81f8c882b0372add0dfc9ea7f.jpg')

    output = model(transforms(img).unsqueeze(0))
    for o in output:
        print(o.shape)
