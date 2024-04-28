import torchvision.models as models

# for timm models we don't have such files, since it provides a simple wrapper timm.create_model. Check tools.models.py
BACKBONES = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "wide_resnet50": models.wide_resnet50_2,
    "wide_resnet101": models.wide_resnet101_2,
    "efficientnet_v2_s": models.efficientnet_v2_s,
    "efficientnet_v2_m": models.efficientnet_v2_m,
    "efficientnet_v2_l": models.efficientnet_v2_l,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "vit_b_32": models.vit_b_32,
}