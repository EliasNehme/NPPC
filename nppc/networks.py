import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


## Networks auxiliaries
## ====================
def zero_weights(module, factor=1e-6):
    module.weight.data = module.weight.data * factor
    if hasattr(module, 'bias') and (module.bias is not None):
        nn.init.constant_(module.bias, 0)
    return module


class ShortcutBlock(nn.Module):
    def __init__(self, base, shortcut=None, factor=None):
        super().__init__()

        self.base = base
        self.shortcut = shortcut
        self.factor = factor

    def forward(self, x):
        shortcut = x
        x = self.base(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)
        if self.factor is not None:
            x = x * self.factor
        x = x + shortcut
        return x


class ConcatShortcutBlock(nn.Module):
    def __init__(self, base, shortcut=None, factor=None):
        super().__init__()

        self.base = base
        self.shortcut = shortcut
        self.factor = factor

    def forward(self, x):
        shortcut = x
        x = self.base(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)
        if self.factor is not None:
            x = x * self.factor
        x = torch.cat((x, shortcut), dim=1)
        return x


class RestorationWrapper(nn.Module):
    def __init__(self, net, offset=None, scale=None, mask=None, pad_base_size=None, naive_restore_func=None):
        super().__init__()
        self.net = net
        self.offset = offset
        self.scale = scale
        self.mask = mask
        self.pad_base_size = pad_base_size
        self.naive_restore_func = naive_restore_func

    def _get_padding(self, x):
        s = self.pad_base_size
        _, _, height, width = x.shape
        if (s is not None) and ((height % s != 0) or (width % s != 0)):
            pad_h = height % s
            pad_w = width % s
            padding = torch.tensor((pad_h // 2, pad_h // 2, pad_w // 2, pad_w // 2))
        else:
            padding = None
        return padding

    def forward(self, x_distorted):
        x_in = x_distorted
        x_naive = self.naive_restore_func(x_distorted)
        if self.offset is not None:
            x_distorted = x_distorted - self.offset
            x_naive = x_naive - self.offset
        if self.scale is not None:
            x_distorted = x_distorted / self.scale
            x_naive = x_naive / self.scale

        padding = self._get_padding(x_distorted)
        if padding is not None:
            x_distorted = F.pad(x_distorted, tuple(padding))

        x_restored = self.net(x_distorted)

        if padding is not None:
            x_restored = F.pad(x_restored, tuple(-padding))  # pylint: disable=invalid-unary-operand-type
        
        x_restored = x_naive + x_restored

        if self.scale is not None:
            x_restored = x_restored * self.scale
        if self.offset is not None:
            x_restored = x_restored + self.offset

        if self.mask is not None:
            x_restored = x_in * (1 - self.mask[None]) + x_restored * self.mask[None]

        return x_restored


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, n_groups=8):
        super().__init__()

        self.block = ShortcutBlock(
            nn.Sequential(
                nn.Conv2d(dim, dim_out, 3, padding=1),
                nn.GroupNorm(n_groups, dim_out),
                nn.SiLU(),
                nn.Conv2d(dim_out, dim_out, 3, padding=1),
                nn.GroupNorm(n_groups, dim_out),
                nn.SiLU(),
            ),
            shortcut= nn.Conv2d(dim, dim_out, 1) if dim != dim_out else None,
        )

    def forward(self, x):
        return self.block(x)


class Attention(nn.Module):
    def __init__(
        self,
        in_channels,
        embedding_channels=None,
        heads=4,
    ):
        super().__init__()
        self.heads = heads

        if embedding_channels is None:
            embedding_channels = in_channels

        self.conv_in = nn.Conv1d(in_channels, 3 * embedding_channels, 1, bias=False)
        self.conv_out = zero_weights(nn.Conv1d(embedding_channels, in_channels, 1))

    def forward(self, x):
        x_in = x
        shape = x.shape
        x = x.flatten(2)

        x = self.conv_in(x)
        x = x.unflatten(1, (3, self.heads, -1))
        q, k, v = x[:, 0], x[:, 1], x[:, 2]

        attn = torch.einsum(f"bhki,bhka->bhia", q, k)
        attn = attn * attn.shape[1] ** -0.5
        attn = attn.softmax(dim=-1)
        x = torch.einsum(f"bhia,bhda->bhdi", attn, v)

        x = x.flatten(1, 2)
        x = self.conv_out(x)

        x = x.unflatten(2, shape[2:])
        x = x + x_in
        return x


## Networks
## ========
class UNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=None,
            channels_list=(32, 64, 128, 256),
            bottleneck_channels=512,
            downsample_list=(False, True, True, True),
            n_blocks=2,
            n_blocks_bottleneck=2,
            min_channels_decoder=64,
            upscale_factor=1,
            n_groups=8,
        ):

        super().__init__()
        self.max_scale_factor = 2 ** np.sum(downsample_list)

        if out_channels is None:
            out_channels = in_channels

        ch = in_channels

        ## Encoder
        ## =======
        self.encoder_blocks = nn.ModuleList([])
        ch_hidden_list = []

        layers = []
        ch_ = channels_list[0]
        layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
        ch = ch_
        self.encoder_blocks.append(nn.Sequential(*layers))
        ch_hidden_list.append(ch)

        for i_level in range(len(channels_list)):
            ch_ = channels_list[i_level]
            downsample = downsample_list[i_level]

            layers = []
            if downsample:
                layers.append(nn.MaxPool2d(2))
            for _ in range(n_blocks):
                layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
                ch = ch_
                layers.append(nn.GroupNorm(n_groups, ch))
                layers.append(nn.LeakyReLU(0.1))
            self.encoder_blocks.append(nn.Sequential(*layers))
            ch_hidden_list.append(ch)

        ## Bottleneck
        ## ==========
        ch_ = bottleneck_channels
        layers = []
        for _ in range(n_blocks_bottleneck):
            layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
            ch = ch_
            layers.append(nn.GroupNorm(n_groups, ch))
            layers.append(nn.LeakyReLU(0.1))
        self.bottleneck = nn.Sequential(*layers)

        ## Decoder
        ## =======
        self.decoder_blocks = nn.ModuleList([])
        for i_level in reversed(range(len(channels_list))):
            ch_ = max(channels_list[i_level], min_channels_decoder)
            downsample = downsample_list[i_level]
            ch = ch + ch_hidden_list.pop()
            layers = []

            for _ in range(n_blocks):
                layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
                ch = ch_
                layers.append(nn.GroupNorm(n_groups, ch))
                layers.append(nn.LeakyReLU(0.1))
            if downsample:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.decoder_blocks.append(nn.Sequential(*layers))

        ch = ch + ch_hidden_list.pop()
        ch_ = channels_list[0]
        layers = []
        if upscale_factor != 1:
            factors = (2,) *  int(np.log2(upscale_factor))
            assert (np.prod(factors) == upscale_factor), 'Upscale factor must be a power of 2'
            for f in factors:
                layers.append(nn.Conv2d(ch, ch_ * f ** 2, kernel_size=3, padding=1))
                layers.append(nn.PixelShuffle(f))
                ch = ch_
        layers.append(zero_weights(nn.Conv2d(ch, out_channels, 1)))
        self.decoder_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        h = []
        for block in self.encoder_blocks:
            x = block(x)
            h.append(x)
        
        x = self.bottleneck(x)
        for block in self.decoder_blocks:
            x = torch.cat((x, h.pop()), dim=1)
            x = block(x)

        return x


class ResUNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=None,
            channels_list=(128, 128, 256, 256, 512, 512),
            bottleneck_channels=512,
            downsample_list=(False, True, True, True, True, True),
            attn_list=(False, False, False, False, True, False),
            n_blocks=2,
            min_channels_decoder=1,
            upscale_factor=1,
            n_groups=8,
            attn_heads=1,
        ):

        super().__init__()
        self.max_scale_factor = 2 ** np.sum(downsample_list)

        if out_channels is None:
            out_channels = in_channels

        ch = in_channels

        ## Encoder
        ## =======
        self.encoder_blocks = nn.ModuleList([])
        ch_hidden_list = []

        layers = []
        ch_ = channels_list[0]
        layers.append(nn.Conv2d(ch, ch_, 3, padding=1))
        ch = ch_
        self.encoder_blocks.append(nn.Sequential(*layers))
        ch_hidden_list.append(ch)

        for i_level in range(len(channels_list)):
            ch_ = channels_list[i_level]
            downsample = downsample_list[i_level]
            attn = attn_list[i_level]

            if downsample:
                layers = []
                layers.append(nn.Conv2d(ch, ch, 3, padding=1, stride=2))
                self.encoder_blocks.append(nn.Sequential(*layers))
                ch_hidden_list.append(ch)

            for _ in range(n_blocks):
                layers = []
                layers.append(ResBlock(ch, ch_, n_groups=n_groups))
                ch = ch_
                if attn:
                    layers.append(Attention(ch, heads=attn_heads))
                self.encoder_blocks.append(nn.Sequential(*layers))
                ch_hidden_list.append(ch)

        ## Bottleneck
        ## ==========
        ch_ = bottleneck_channels
        layers = []
        layers.append(ResBlock(ch, ch_, n_groups=n_groups))
        ch = ch_
        layers.append(Attention(ch, heads=attn_heads))
        layers.append(ResBlock(ch, ch, n_groups=n_groups))
        self.bottleneck = nn.Sequential(*layers)

        ## Decoder
        ## =======
        self.decoder_blocks = nn.ModuleList([])
        for i_level in reversed(range(len(channels_list))):
            ch_ = max(channels_list[i_level], min_channels_decoder)
            downsample = downsample_list[i_level]
            attn = attn_list[i_level]

            for _ in range(n_blocks):
                layers = []
                layers.append(ResBlock(ch + ch_hidden_list.pop(), ch_, n_groups=n_groups))
                ch = ch_
                if attn:
                    layers.append(Attention(ch, heads=attn_heads))
                self.decoder_blocks.append(nn.Sequential(*layers))

            if downsample:
                layers = []
                layers.append(ResBlock(ch + ch_hidden_list.pop(), ch, n_groups=n_groups))
                if attn:
                    layers.append(Attention(ch, heads=attn_heads))
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
                layers.append(nn.Conv2d(ch, ch, 3, padding=1))
                self.decoder_blocks.append(nn.Sequential(*layers))

        layers = []
        layers.append(ResBlock(ch + ch_hidden_list.pop(), ch, n_groups=n_groups))
        layers.append(nn.GroupNorm(n_groups, ch))
        layers.append(nn.SiLU())
        if upscale_factor != 1:
            factors = (2,) *  int(np.log2(upscale_factor))
            assert (np.prod(factors) == upscale_factor), 'Upscale factor must be a power of 2'
            for f in factors:
                layers.append(nn.Conv2d(ch, ch * f ** 2, kernel_size=3, padding=1))
                layers.append(nn.PixelShuffle(f))
        layers.append(zero_weights(nn.Conv2d(ch, out_channels, 1)))
        self.decoder_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        h = []
        for block in self.encoder_blocks:
            x = block(x)
            h.append(x)
        
        x = self.bottleneck(x)
        for block in self.decoder_blocks:
            x = torch.cat((x, h.pop()), dim=1)
            x = block(x)

        return x


class ResCNN(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out=None,
            channels_hidden=64,
            n_blocks=16,
            upscale_factor=None,
        ):

        super().__init__()
        self.max_scale_factor = 1

        if channels_out is None:
            channels_out = channels_in

        ch = channels_in

        net = []

        ## Input block
        ## ===========
        net += [nn.Conv2d(ch, channels_hidden, 3, padding=1)]
        ch = channels_hidden

        ## Main block
        ## ==========
        block = []
        for _ in range(n_blocks):
            block += [
                ShortcutBlock(
                    nn.Sequential(
                        nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                    ),
                ),
            ]
        block += [
            nn.Conv2d(ch, ch, 3, padding=1),
        ]
        net += [ShortcutBlock(nn.Sequential(*block))]

        ## Output block
        ## ============
        if upscale_factor is not None:
            factors = (2,) *  int(np.log2(upscale_factor))
            assert (np.prod(factors) == upscale_factor), 'Upscale factor must be a power of 2'
            for f in factors:
                net += [
                    nn.Conv2d(ch, ch * f ** 2, kernel_size=3, padding=1),
                    nn.PixelShuffle(f),
                ]
        net += [
            nn.Conv2d(ch, channels_out, kernel_size=3, padding=1),
        ]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        # x_in = x

        x = self.net(x)

        # if self.upscale_factor is not None:
        #     x_in = F.interpolate(x_in, scale_factor=self.upscale_factor, mode='nearest')
        # x = x_in + x

        return x