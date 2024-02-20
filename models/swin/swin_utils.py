import torch.nn as nn


class mySwin(nn.Module):
    def __init__(self, swin):
        super(mySwin, self).__init__()
        self.swin = swin

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



