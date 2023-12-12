import numpy as np
import torch
import torch.nn as nn

## Networks auxiliaries
## ====================
def factor_weights(module, factor=None, bias_factor=None):
    if factor is not None:
        module.weight.data = module.weight.data * factor
        if hasattr(module, 'bias') and (module.bias is not None):
            if bias_factor is None:
                bias_factor = factor
            module.bias.data = module.bias.data * factor
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
        self.conv_out = factor_weights(nn.Conv1d(embedding_channels, in_channels, 1), factor=1e-6)

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
            output_factor=None,
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
        ch_ = max(channels_list[0], min_channels_decoder)
        layers = []
        if upscale_factor != 1:
            factors = (2,) *  int(np.log2(upscale_factor))
            assert (np.prod(factors) == upscale_factor), 'Upscale factor must be a power of 2'
            for f in factors:
                layers.append(nn.Conv2d(ch, ch_ * f ** 2, kernel_size=3, padding=1))
                layers.append(nn.PixelShuffle(f))
                ch = ch_
        layers.append(factor_weights(nn.Conv2d(ch, out_channels, 1), factor=output_factor))
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
            output_factor=None,
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
                ch = ch + ch_hidden_list.pop()
                layers.append(ResBlock(ch, ch_, n_groups=n_groups))
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
        ch_ = max(channels_list[0], min_channels_decoder)
        ch = ch + ch_hidden_list.pop()
        layers.append(ResBlock(ch, ch_, n_groups=n_groups))
        ch = ch_
        layers.append(nn.GroupNorm(n_groups, ch))
        layers.append(nn.SiLU())
        if upscale_factor != 1:
            factors = (2,) *  int(np.log2(upscale_factor))
            assert (np.prod(factors) == upscale_factor), 'Upscale factor must be a power of 2'
            for f in factors:
                layers.append(nn.Conv2d(ch, ch * f ** 2, kernel_size=3, padding=1))
                layers.append(nn.PixelShuffle(f))
        layers.append(factor_weights(nn.Conv2d(ch, out_channels, 1), factor=output_factor))
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
            in_channels,
            out_channels=None,
            hidden_channels=64,
            n_blocks=16,
            upscale_factor=1,
            output_factor=None,
        ):

        super().__init__()
        self.max_scale_factor = 1

        if out_channels is None:
            out_channels = in_channels

        ch = in_channels
        layers = []

        ## Input block
        ## ===========
        layers.append(nn.Conv2d(ch, hidden_channels, 3, padding=1))
        ch = hidden_channels

        ## Main block
        ## ==========
        main_layers = []
        for _ in range(n_blocks):
            layers.append(ShortcutBlock(nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            )))
        layers.append(nn.Conv2d(ch, ch, 3, padding=1))
        layers += [ShortcutBlock(nn.Sequential(*main_layers))]

        ## Output block
        ## ============
        if upscale_factor != 1:
            factors = (2,) *  int(np.log2(upscale_factor))
            assert (np.prod(factors) == upscale_factor), 'Upscale factor must be a power of 2'
            for f in factors:
                layers.append(nn.Conv2d(ch, ch * f ** 2, kernel_size=3, padding=1))
                layers.append(nn.PixelShuffle(f))
        layers.append(factor_weights(nn.Conv2d(ch, out_channels, kernel_size=3, padding=1), factor=output_factor))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x