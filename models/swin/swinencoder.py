from .swin.swin_transformer_backbone import SwinTransformer as STBackbone


def setup(model_name):
    if model_name =="swin-T":
        model = swinT(pretrained=True)
    elif model_name =="swin-S":
        model = swinS(pretrained=True)
    elif model_name =="swin-B":
        model = swinB(pretrained=True)
    else:
        assert model_name =="swin-L", "the swin transformer is unclear"
        model = swinL(pretrained=True)
    return model


swin_pretrain_path = ['./OSIC/models/swin/swin/swin_T.pth', './OSIC/models/swin/swin/swin_S.pth',
                      './OSIC/models/swin/swin/swin_B.pth', './OSIC/models/swin/swin/swin_L.pth']


def swinT(pretrained=False):
    """Constructs a Swin-T model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = STBackbone(img_size=384, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                       window_size=12, num_classes=1000)
    if pretrained:
        model.load_weights(swin_pretrain_path[0])
    return model


def swinS(pretrained=False):
    """Constructs a Swin-T model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = STBackbone(img_size=384, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
                       window_size=12, num_classes=1000)
    if pretrained:
        model.load_weights(swin_pretrain_path[1])
    return model


def swinB(pretrained=False):
    """Constructs a Swin-T model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = STBackbone(img_size=384, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 12, 24],
                       window_size=12, num_classes=1000)
    if pretrained:
        model.load_weights(swin_pretrain_path[2])
    return model


def swinL(pretrained=False):
    """Constructs a Swin-T model
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = STBackbone(img_size=384, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
                       window_size=12, num_classes=1000)
    if pretrained:
        model.load_weights(swin_pretrain_path[3])
    return model
