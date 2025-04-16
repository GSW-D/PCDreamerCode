import torch
from torch import nn
import math
from einops import rearrange
from einops.layers.torch import Rearrange


class MLP(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Linear(last_channel, out_channel))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Linear(last_channel, layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class CrossViewViT(nn.Module):
    def __init__(self,
                 image_size = 224,
                 patch_size = 16,
                 dim = 1024,
                 depth = 6,
                 heads = 16,
                 mlp_dim = 2048,
                 channels=1,
                 dim_head=64,
                 dropout = 0.1,
                 emb_dropout = 0.1,
                 view_distance = 1.5,
                 init_view = 'right'):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.view_distance = view_distance
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        if init_view == 'right':
            self.pos_embedding = torch.tensor([0, 0, self.view_distance,
                                       self.view_distance / math.sqrt(2.0), 0, self.view_distance/math.sqrt(2.0),
                                       self.view_distance, 0, 0,
                                       0, 0, -self.view_distance,
                                       -self.view_distance, 0, 0,
                                       -self.view_distance/math.sqrt(2.0), 0, self.view_distance/math.sqrt(2.0)],
                                      dtype=torch.float32).view(-1, 6, 3)
        elif init_view == 'svd':
            angles = (torch.linspace(0, 360, 21 + 1) % 360)[[0, 2, 5, 11, 16, 19]]  # selected 6 frames
            self.pos_embedding = torch.tensor([0, 0, self.view_distance,
                                       self.view_distance * math.sin(angles[1] / 180 * math.pi), 0, self.view_distance * math.cos(angles[1] / 180 * math.pi),
                                       self.view_distance * math.sin(angles[2] / 180 * math.pi), 0, self.view_distance * math.cos(angles[2] / 180 * math.pi),
                                       self.view_distance * math.sin(angles[3] / 180 * math.pi), 0, self.view_distance * math.cos(angles[3] / 180 * math.pi),
                                       self.view_distance * math.sin(angles[4] / 180 * math.pi), 0, self.view_distance * math.cos(angles[4] / 180 * math.pi),
                                       self.view_distance * math.sin(angles[5] / 180 * math.pi), 0, self.view_distance * math.cos(angles[5] / 180 * math.pi)],
                                      dtype=torch.float32).view(-1, 6, 3)
        elif init_view == 'sn55':
            self.pos_embedding = torch.tensor([self.view_distance, 0, 0,
                                               self.view_distance / math.sqrt(2.0), 0, -self.view_distance / math.sqrt(2.0),
                                               0, 0, -self.view_distance,
                                               -self.view_distance, 0, 0,
                                               0, 0, self.view_distance,
                                               self.view_distance / math.sqrt(2.0), 0, self.view_distance / math.sqrt(2.0)],
                                              dtype=torch.float32).view(-1, 6, 3)
        elif init_view == 'sn55_svd':
            angles = (torch.linspace(0, 360, 21 + 1) % 360)[[0, 2, 5, 11, 16, 19]]  # selected 6 frames
            self.pos_embedding = torch.tensor([self.view_distance, 0, 0,
                                       self.view_distance * math.cos(angles[1] / 180 * math.pi), 0, self.view_distance * math.sin(angles[1] / 180 * math.pi),
                                       self.view_distance * math.cos(angles[2] / 180 * math.pi), 0, self.view_distance * math.sin(angles[2] / 180 * math.pi),
                                       self.view_distance * math.cos(angles[3] / 180 * math.pi), 0, self.view_distance * math.sin(angles[3] / 180 * math.pi),
                                       self.view_distance * math.cos(angles[4] / 180 * math.pi), 0, self.view_distance * math.sin(angles[4] / 180 * math.pi),
                                       self.view_distance * math.cos(angles[5] / 180 * math.pi), 0, self.view_distance * math.sin(angles[5] / 180 * math.pi)],
                                      dtype=torch.float32).view(-1, 6, 3)
        else:
            self.pos_embedding = torch.tensor([self.view_distance, 0, 0,
                                               self.view_distance / math.sqrt(2.0), 0, -self.view_distance / math.sqrt(2.0),
                                               0, 0, -self.view_distance,
                                               -self.view_distance, 0, 0,
                                               0, 0, self.view_distance,
                                               self.view_distance / math.sqrt(2.0), 0, self.view_distance / math.sqrt(2.0)],
                                              dtype=torch.float32).view(-1, 6, 3)

        self.pos_mlp = MLP(in_channel=3, layer_dims=[128, 512, 1024])
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, 128)

    def forward(self, x):
        x = self.to_patch_embedding(x)  # [B, 6, 1024]
        b, n, _ = x.shape

        x += self.pos_mlp(self.pos_embedding.repeat(b, 1, 1).to(x.device))
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.mlp_head(x)


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn

class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)