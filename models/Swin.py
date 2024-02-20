from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from models.swin.swin.swin_transformer_backbone import SwinTransformer as STBackbone

import sys
sys.path.append("..")
import opts


opt = opts.parse_opt()


class mySwin(nn.Module):
    def __init__(self):
        super(mySwin, self).__init__()
        backbone = STBackbone(img_size=384, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
                              window_size=12, num_classes=1000)
        print('load pretrained weights!')
        # backbone.load_weights('/root/autodl-tmp/osic/models/swin/swin/swin_large_patch4_window12_384_22kto1k_no_head.pth')
        backbone.load_weights(opt.swin_pretrain_path)
        # Freeze parameters
        for _name, _weight in backbone.named_parameters():
            _weight.requires_grad = False
            # print(_name, _weight.requires_grad)
        self.swin = backbone

    def forward(self, x):
        x = self.swin.patch_embed(x)
        if self.swin.ape:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.pos_drop(x)
        x_0 = self.swin.layers[0](x)
        x_1 = self.swin.layers[1](x_0)
        x_2 = self.swin.layers[2](x_1)
        x_3 = self.swin.layers[3](x_2)

        return x_0, x_1, x_2, x_3
