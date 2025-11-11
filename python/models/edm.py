import torch
import torch.nn as nn
import torch.nn.functional as F
import tomllib
import numpy as np

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': 
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal': 
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform':
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal': 
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


#----------------------------------------------------------------------------
# Fully-connected layer.
# class Linear(torch.nn.Module):
#     def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
#         self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
#         self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

#     def forward(self, x):
#         x = x @ self.weight.to(x.dtype).t()
#         if self.bias is not None:
#             x = x.add_(self.bias.to(x.dtype))
#         return x

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    
#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.
class Conv1d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], init_mode='kaiming_normal', init_weight=1.0, init_bias=0.0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel, fan_out=out_channels*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.unsqueeze(0) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0

        #assert isinstance(f, torch.Tensor)
        assert isinstance(f, torch.Tensor) or f is None
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.up:
            assert f is not None
            x = torch.nn.functional.conv_transpose1d(x, f.mul(4).tile([self.in_channels, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
        if self.down:
            assert f is not None
            x = torch.nn.functional.conv1d(x, f.tile([self.in_channels, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
        
        if w is not None:
            x = torch.nn.functional.conv1d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1))

        return x
    
#----------------------------------------------------------------------------
# Group normalization.
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

    
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        assert isinstance(self.freqs, torch.Tensor)

        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    
#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, 
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        skip_scale=1,
        eps=1e-5,
        resample_filter=[1,1],
        resample_proj=False,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.skip_scale = skip_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = nn.Linear(in_features=emb_channels, out_features=out_channels)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv1d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv1d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv1d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

        #print(f"in={in_channels}, out={out_channels}, attn={self.num_heads > 0}, type={"up" if up else "down" if down else "mid"}")

    def forward(self, x, emb):
        orig = x
        x = self.conv0(F.silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).to(x.dtype)

        x = F.silu(self.norm1(x.add_(params)))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads != 0:
            y = self.qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2])
            q, k, v = normalize(y, dim=2).unbind(3)  # pixel norm & split
            w = torch.einsum("nhcq,nhck->nhqk", q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum("nhqk,nhck->nhcq", w, v)
            y = self.proj(y.reshape(*x.shape))
            x += y

        return x

class UNet(nn.Module):
    def __init__(self,
        resolution,  # Image resolution.
        in_channels,  # Image channels.
        condition_dim,  # Class label dimensionality. 0 = unconditional.
        base_channels=64,  # Base multiplier for the number of channels.
        channel_mult=[1, 2, 3, 4, 5],  # Per-resolution multipliers for the number of channels.
        channel_mult_noise=None,  # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb=None,  # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks=3,  # Number of residual blocks per resolution.
        attn_resolutions=[16, 8],  # List of resolutions with self-attention.
    ):
        super().__init__()
        block_channels = [base_channels * x for x in channel_mult]

        #print(f"block_channels={block_channels}")
        noise_channels = base_channels * channel_mult_noise if channel_mult_noise is not None else block_channels[0] 
        embed_channels = base_channels * channel_mult_emb if channel_mult_emb is not None else max(block_channels)

        init = {"init_mode": 'kaiming_uniform', "init_weight": np.sqrt(1/3), "init_bias": np.sqrt(1/3)}
        init_zero = {"init_mode": 'kaiming_uniform', "init_weight" : 0.0, "init_bias" : 0.0}
        block_kwargs = {"emb_channels": embed_channels, "channels_per_head": 64, "init": init, "init_zero": init_zero}

        # Embedding
        self.emb_noise = FourierEmbedding(num_channels=noise_channels)
        self.emb_layer0 = nn.Linear(in_features=base_channels, out_features=embed_channels)#, **init)
        self.emb_layer1 = nn.Linear(in_features=embed_channels, out_features=embed_channels)#, **init)
        self.emb_condition = nn.Linear(in_features=condition_dim, out_features=embed_channels, bias=False)# init_mode='kaiming_normal', init_weight=np.sqrt(condition_dim)) if condition_dim else None

       # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level,  channels in enumerate(block_channels):
            res = resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = Conv1d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(block_channels))):
            res = resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv1d(in_channels=cout, out_channels=in_channels, kernel=3, **init_zero)
        
    
    def forward(self, x, noise_labels, class_labels):
        # Embed noise
        emb = self.emb_noise(noise_labels)
        emb = F.silu(self.emb_layer0(emb))
        emb = self.emb_layer1(emb)

        # Embed conditioning information (class labels)
        if self.emb_condition is not None:
            emb = emb + self.emb_condition(class_labels)        
        emb = F.silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(F.silu(self.out_norm(x)))
        return x
    
class Denoiser(nn.Module):
    def __init__(self,
        resolution,     # Image resolution.
        in_channels,    # Image channels.
        condition_dim,  # Class label dimensionality. 0 = unconditional.
        data_std=0.5,   # Expected standard deviation of the training data.
        return_logvar=False,  # Does nothing
        **unet_kwargs,  # Keyword arguments for UNet.
    ):
        super().__init__()
        
        self.img_resolution = resolution
        self.img_channels = in_channels
        self.label_dim = condition_dim
        self.data_std = data_std
        self.unet = UNet(resolution=resolution, in_channels=in_channels, condition_dim=condition_dim, **unet_kwargs)
    
    def forward(self, x, noise_std, condition_vector=None, return_logvar=False, **unet_kwargs):

        x = x.to(torch.float32)
        noise_std = noise_std.to(torch.float32).reshape(-1, 1, 1)
        
        if self.label_dim == 0:
            # No conditioning information for this network
            condition_vector = None
        elif condition_vector is None:
            # Unconditional denoising
            condition_vector = torch.zeros([1, self.label_dim], device=x.device)
        else:
            condition_vector = condition_vector.to(torch.float32).reshape(-1, self.label_dim)

        # Preconditioning weights
        c_skip = self.data_std**2 / (noise_std**2 + self.data_std**2)
        c_out = noise_std * self.data_std / (noise_std**2 + self.data_std**2).sqrt()
        c_in = 1 / (self.data_std**2 + noise_std**2).sqrt()
        c_noise = noise_std.flatten().log() / 4

        # Run the model.
        x_in = c_in * x
        F_x = self.unet(x_in, c_noise, condition_vector, **unet_kwargs)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        return D_x

    @staticmethod
    @torch.no_grad
    def from_config(config):
        return Denoiser(
            in_channels=config["in_channels"],
            resolution=config["resolution"],
            base_channels=config["base_channels"],
            channel_mult=config["channel_mult"],
            num_blocks=config["num_blocks"],
            condition_dim=config["label_dim"],
        )

    @staticmethod
    @torch.no_grad
    def from_config_file(config_file):
        with open(config_file, "rb") as fp:
            config = tomllib.load(fp)
        return Denoiser.from_config(config["model"])
    

    @staticmethod
    @torch.no_grad
    def check_shape(resolution, channels, base_channels):
        batch_size = 1024
        label_dim = 8
        for dim in (1,):
            model = Denoiser(resolution=resolution, in_channels=channels, data_std=1.0, condition_dim=label_dim, base_channels=base_channels)
            cond_vec = torch.randn((batch_size, label_dim))
            x = torch.rand((batch_size, channels) + (resolution,) * dim)
            noise_std = torch.rand((batch_size, 1) + (1,) * dim)
            out = model(x, noise_std, condition_vector=cond_vec)
            print(f"{dim}D: input = {x.shape=}, output = {out.shape=}")
            assert x.shape == out.shape

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    Denoiser.check_shape(resolution=128, channels=14, base_channels = 64)
