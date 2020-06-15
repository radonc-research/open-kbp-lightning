import torch
from torch import nn
from torch.nn.functional import interpolate
from torchvision.models.video import r3d_18, r2plus1d_18
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
from kornia.losses import SSIM

class _3DLoss(nn.Module):
    def __init__(self, device, num_layers, weights, use_mask):
        super(_3DLoss, self).__init__()
        model = r3d_18(pretrained=True).to(device)
        ps = list(model.parameters())
        for p in ps: p.requires_grad=False
        self.model = []
        self.model.append(model.stem)
        self.model.append(model.layer1)
        self.model.append(model.layer2)
        self.model.append(model.layer3)

        self.layer = num_layers
        self.weights = weights
        self.use_mask = use_mask

        self.loss = nn.L1Loss()


    def weighted_loss(self, x, y, mask):
        return torch.sum(torch.abs(y - x)) / torch.sum(mask)

    def forward(self, x, y, mask):
        if self.use_mask:
            feature_loss = self.weights[0]*self.weighted_loss(x, y, mask)
        else:
            feature_loss = self.weights[0]*self.loss(x, y)

        x = x.repeat(1, 3, 1, 1, 1)
        y = y.repeat(1, 3, 1, 1, 1)

        #mask1 = interpolate(mask.repeat(1, 64, 1, 1, 1), scale_factor=(1, 0.5, 0.5), mode='nearest')
        #mask2 = interpolate(mask1.repeat(1, 2, 1, 1, 1), scale_factor=(0.5), mode='nearest')

        for w, model_loss in zip(self.weights[1:], self.model[:self.layer]):

            x = model_loss(x)
            y = model_loss(y)
            if self.use_mask:
                dim = x.shape[1]
                size = x.shape[2:]
                mask = interpolate(mask.repeat(1, dim, 1, 1, 1), size=size, mode='nearest')
                feature_loss += w * self.weighted_loss(x, y, mask)
            else:
                feature_loss += w*self.loss(x, y)

        return feature_loss
