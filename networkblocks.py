import torch
from torch import nn
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def Upsample(dim, output_padding = 0):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1,output_padding=output_padding)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, mult=2, norm=True, groups=4):
        super(ConvBlock, self).__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, in_channels))
            if not time_emb_dim == None else nn.Identity()
        )

        self.ds_conv = nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels)
        # put this in self.net = nn.Sequential() and then call it in forward
        self.net = nn.Sequential(
            #nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels),
            nn.GroupNorm(groups, in_channels) if norm else nn.Identity(),
            nn.Conv2d(in_channels, out_channels * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(groups, out_channels * mult),
            nn.Conv2d(out_channels * mult, out_channels, 3, padding=1),
            nn.GELU()
        )

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if not time_emb == None:
            time = self.mlp(time_emb)
            #h =  h + einops.rearrange(time, "b c -> b c 1 1")
            time = time.unsqueeze(2).unsqueeze(3)  # Shape: (batch_size, out_channels, 1, 1)
            time = time.expand(-1, -1, x.shape[2], x.shape[3])  # Expand to match the spatial dimensions of x
            h = h + time
        # Concatenate time embedding with x
        #x = torch.cat([x, time_embed], dim=1)

        h = self.net(h)
        return h
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads."
        self.Norm = nn.GroupNorm(num_heads, in_channels)
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, input):
        input = self.Norm(input)
        batch_size, C, width, height = input.size()
        query = self.query(input).view(batch_size, self.num_heads, self.head_dim, width*height)#.permute(0, 2, 1, 3)
        key = self.key(input).view(batch_size, self.num_heads, self.head_dim, width*height)
        energy = torch.einsum("bnqd,bnkd->bnqk", [query, key])
        attention = self.softmax(energy)
        value = self.value(input).view(batch_size, self.num_heads, self.head_dim, width*height)
        out = torch.einsum("bnqk,bnkd->bnqd", [attention, value])
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, C, width, height)
        out = self.out_proj(out)
        out = self.gamma*out + input
        return out
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, in_channels, num_heads,emd_dim):
        super(MultiHeadCrossAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads."
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)


    def forward(self, query, key_value):
        batch_size, C, width, height = query.size()
        query = self.query(query).view(batch_size, self.num_heads, self.head_dim, width*height)
        key = self.key(key_value).view(batch_size, self.num_heads, self.head_dim, width*height)
        energy = torch.einsum("bnqd,bnkd->bnqk", [query, key])
        attention = self.softmax(energy)
        value = self.value(key_value).view(batch_size, self.num_heads, self.head_dim, width*height)
        out = torch.einsum("bnqk,bnkd->bnqd", [attention, value])
        out = out.contiguous().view(batch_size, C, width, height)
        out = self.out_proj(out)
        out = self.gamma*out + query
        return out

class UNetAttn(nn.Module):
    def __init__(self, firstconv_channels, img_channels,output_channel,channel_multiplier=(1, 2, 4, 8), time_emb_dim=None,class_emb_dim=None):
        super(UNetAttn, self).__init__()
        self.time_embed = SinusoidalPositionEmbeddings(time_emb_dim) if not time_emb_dim == None else nn.Identity()
        self.time_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        ) if not time_emb_dim == None else nn.Identity()
        self.class_embed = SinusoidalPositionEmbeddings(class_emb_dim)if not class_emb_dim == None else nn.Identity()
        self.class_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(class_emb_dim, class_emb_dim))
        ) if not class_emb_dim == None else nn.Identity()
        self.firstconv = nn.Conv2d(img_channels, firstconv_channels, 7, padding=3)

        self.channels_list = [firstconv_channels * mult for mult in channel_multiplier]

        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])

        for i in range(len(self.channels_list)-1):
            self.encoder.append(
                nn.ModuleList(
                    [
                        ConvBlock(self.channels_list[i], self.channels_list[i+1],time_emb_dim),
                        ConvBlock(self.channels_list[i+1], self.channels_list[i + 1],time_emb_dim),
                        MultiHeadSelfAttention(self.channels_list[i + 1],num_heads=4),
                        #MultiHeadCrossAttention(self.channels_list[i + 1],num_heads=4) if self.classcond else nn.Identity(),
                        Downsample(self.channels_list[i+1]) if not i == len(self.channels_list)-2 else nn.Identity() ,
                    ]
                )
            )


        for i in reversed(range(len(self.channels_list)-1)):
            self.decoder.append(
                nn.ModuleList(
                    [
                        ConvBlock(self.channels_list[i+1] * 2, self.channels_list[i],time_emb_dim),
                        ConvBlock(self.channels_list[i], self.channels_list[i],time_emb_dim),
                        MultiHeadSelfAttention(self.channels_list[i], num_heads=4),
                        #MultiHeadCrossAttention(self.channels_list[i + 1], num_heads=4) if self.classcond else nn.Identity(),
                        Upsample(self.channels_list[i]) if not i == 0 else nn.Identity(),
                    ]
                )
            )




        self.bottleneckBlock0 = ConvBlock(self.channels_list[-1], self.channels_list[-1],time_emb_dim)
        self.bottleneckAttn = MultiHeadSelfAttention(self.channels_list[-1],num_heads=4)
        self.bottleneckBlock1 = ConvBlock(self.channels_list[-1], self.channels_list[-1],time_emb_dim)

        self.final_conv = nn.Conv2d(self.channels_list[0], output_channel, 1)

    def forward(self, x , step=None, classes=None):
        # Create time embedding

        embed = None
        if not step == None:
            time_embed = self.time_embed(step)  # Shape: (batch_size, time_emb_dim)
            time_embed = self.time_mlp(time_embed)  # Transform time embedding
            embed = time_embed
        else:
            time_embed = None

        if not classes == None:
            class_embed = self.class_embed(classes)
            class_embed = self.class_mlp(class_embed)
            embed = time_embed + class_embed if not embed ==None else class_embed
        else :
            class_embed = None # unnessary but for clarity
        # We could also contactenate the time and class embedings with the input.
        # Concatenate time embedding with x
        # x = torch.cat([x, time_embed,class_embed], dim=1)


        x = self.firstconv(x)


        enc = []
        for block0,block1,attn,downsample in self.encoder:
            x = block0(x,embed)
            x = block1(x,embed)
            x = attn(x)
            enc.append(x)
            x = downsample(x)


        x = self.bottleneckBlock0(x,embed)
        x = self.bottleneckAttn(x)
        x = self.bottleneckBlock1(x, embed)



        for  block0,block1,attn,upsample in self.decoder:
            x = torch.cat([x, enc.pop()], dim=1)
            x = block0(x,embed)
            x = block1(x,embed)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)

def plot_grad_flow(path,named_parameters, model_name, epoch):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.cpu().detach().abs().mean())
            max_grads.append(p.grad.cpu().detach().abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.0001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow of " + str(model_name) )
    plt.tight_layout()
    plt.savefig(path + "Grad of model " + str(model_name) + "First batch Epoch " + str(epoch) + ".png")
    plt.close()