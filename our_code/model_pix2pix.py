from torch import nn
from torch.nn import init
import torch
from our_code.Mish import Mish


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('GroupNorm') != -1:  # GroupNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('InstanceNorm') != -1:  # InstanceNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def select_norm(dim, norm='instance'):
    if norm == 'instance':
        return nn.InstanceNorm3d(dim, affine=True)
    elif norm == 'batch':
        return nn.BatchNorm3d(dim)
    elif norm == 'group':
        return nn.GroupNorm(int(dim/32), dim) #always include 32 features in one group


def select_act(act='Mish'):
    if act == 'Mish':
        return Mish(), Mish()
    elif act == 'ReLU':
        return nn.LeakyReLU(0.2, True), nn.ReLU(True)


class UnetMaxGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer='batch', use_resnet=False, used_act='Mish'):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetMaxGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetMaxSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True,use_resnet=use_resnet, used_act=used_act)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetMaxSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_resnet=use_resnet, used_act=used_act)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetMaxSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,use_resnet=use_resnet, used_act=used_act)
        unet_block = UnetMaxSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer,use_resnet=use_resnet, used_act=used_act)
        unet_block = UnetMaxSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer,use_resnet=use_resnet, used_act=used_act)
        self.model = UnetMaxSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer,use_resnet=use_resnet, used_act=used_act)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetMaxSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer='batch', use_resnet=False, used_act='Mish'):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetMaxSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = False

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)

        downrelu, uprelu = select_act(used_act)

        downnorm = select_norm(inner_nc, norm_layer)
        upnorm = select_norm(outer_nc, norm_layer)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            if use_resnet:
                downinterconv = ResnetBlock(inner_nc, norm_layer, use_bias=use_bias, act=downrelu)

                down = [downconv, downrelu, downinterconv]
                up = [uprelu, upconv, nn.Sigmoid()]
            else:
                down = [downconv]
                up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:

            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)

            if use_resnet:
                upinterconv = ResnetBlock(outer_nc, norm_layer, use_bias=use_bias, act=uprelu)
                downinterconv = ResnetBlock(inner_nc, norm_layer, use_bias=use_bias, act=downrelu)

                down = [downrelu, downconv, downnorm, downrelu, downinterconv]
                up = [uprelu, upconv, upnorm, uprelu, upinterconv]

            else:
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]

            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer='batch', kernel=3, use_bias=False, act=Mish()):
        super(ResnetBlock, self).__init__()
        conv_block = []
        p = 0

        conv_block += [nn.Conv3d(dim, 64, kernel_size=1, padding=p, bias=use_bias),
                       select_norm(64, norm_layer),
                       act]
        if kernel==3:
            conv_block += [nn.ReplicationPad3d(1)]

        conv_block += [nn.Conv3d(64, 64, kernel_size=kernel, padding=p, bias=use_bias),
                       select_norm(64, norm_layer),
                       act]

        conv_block += [nn.Conv3d(64, dim, kernel_size=1, padding=p, bias=use_bias),
                       select_norm(dim, norm_layer)]

        self.conv_block = nn.Sequential(*conv_block)


    def forward(self, x):

        out = x + self.conv_block(x)

        return out

