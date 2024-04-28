import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os 

from .backbones import BACKBONES


def create_encoder(backbone):
    try:
        if 'timm-' in backbone:
            backbone = backbone.split('-')[-1]
            model = timm.create_model(model_name=backbone, pretrained=True)
        else:
            model = BACKBONES[backbone](weights="DEFAULT")
    except RuntimeError or KeyError:
        raise RuntimeError('Specify the correct backbone name. Either one of torchvision backbones, or a timm backbone.'
                           'For timm - add prefix \'timm-\'. For instance, timm-resnet18')

    layers = torch.nn.Sequential(*list(model.children()))
    try:
        potential_last_layer = layers[-1]
        while not isinstance(potential_last_layer, nn.Linear):
            potential_last_layer = potential_last_layer[-1]
    except TypeError:
        raise TypeError('Can\'t find the linear layer of the model')

    features_dim = potential_last_layer.in_features
    model = torch.nn.Sequential(*list(model.children())[:-1])

    return model, features_dim


class SupConModel(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=None):
        super(SupConModel, self).__init__()
        self.encoder, self.features_dim = create_encoder(backbone)
        self.fc = nn.Linear(self.features_dim, num_classes)


    def forward(self, x):
        feat = self.encoder(x).view(x.size(0), -1)
        proj = F.normalize(feat, dim=-1)
        logits = self.fc(proj)
        
        return proj, feat, logits
    

    def load_checkpoint(self, checkpoint_file):
        """Loads the checkpoint from the given file."""
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
        # Load the checkpoint on CPU to avoid GPU mem spike
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        try:
            if 'state_dict' in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint["model"]
        except KeyError:
            state_dict = checkpoint

        model_dict = self.state_dict()

        state_dict = {k: v for k, v in state_dict.items()}
        weight_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].size() == v.size():
                print(k)

        model_dict.update(weight_dict)
        self.load_state_dict(model_dict)
    

if __name__ == "__main__":
    input_size = (1, 3, 224, 224)
    # print('EfficientNet models: ', timm.list_models('efficientnetv2*'))

    sample_input = torch.randn(input_size)
    model = SupConModel(backbone='timm-hf_hub:timm/vit_base_patch32_clip_224.openai', num_classes=100)
    print(model)
    
    proj, feat, logits = model(sample_input)
    print("Projection Shape:", proj.shape)
    print("Feature Shape:", feat.shape)
    print("Logits Shape:", logits.shape)