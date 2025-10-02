import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ConBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))


class Resblock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, same_shape=True):
        super().__init__()
        self.same_shape = same_shape
        if not same_shape:
            strides = 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channel, in_channel, 1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        out = self.block(x)
        if not self.same_shape:
            x = self.bn3(self.conv3(x))
        return F.relu(out + x)


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.gap(x).view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CrossAttention(nn.Module):
    """
    Cross attention between x (b, c, h, w) and context (b, c, h, w).
    Returns: out (b, c, h, w), att_weights (b, heads, seq_q, seq_k)
    """
    def __init__(self, channels, emb_dim, num_heads, num_feature):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.depth = emb_dim // num_heads
        self.scale = self.depth ** -0.5
        self.num_feature = num_feature
        # map input channels -> embedding dim (optional conv)
        self.proj_in = nn.Conv2d(channels, emb_dim, kernel_size=1, bias=False)

        # simple linear projections (operate on last dim after flattening spatial dims)
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim* self.num_feature, emb_dim)
        self.Wv = nn.Linear(emb_dim* self.num_feature, emb_dim)

        # project back to original channel if desired
        self.proj_out = nn.Conv2d(emb_dim, channels, kernel_size=1, bias=False)

    def forward(self, x, context, pad_mask=None):
        """
        x: (b, c, h, w)
        context: (b, c_ctx, h_ctx, w_ctx)  -- typically same spatial dims in our use
        returns: out (b, c, h, w), att_weights (b, num_heads, seq_q, seq_k)
        """
        b, c, h, w = x.shape

        # map to embedding dim
        # x_emb = self.proj_in(x)              # (b, emb_dim, h, w)
        x_flat = rearrange(x, 'b d h w -> b (h w) d')    # (b, seq_q, emb_dim)
        # ctx_emb = self.proj_in(context) if context.shape[1] == c else self.proj_in(context)
        ctx_flat = rearrange(context, 'b d h w -> b (h w) d') # (b, seq_k, emb_dim)

        # Q K V
        Q = self.Wq(x_flat)   # (b, seq_q, emb_dim)
        K = self.Wk(ctx_flat) # (b, seq_k, emb_dim)
        V = self.Wv(ctx_flat) # (b, seq_k, emb_dim)

        # split heads: (b, heads, seq, depth)
        Q = Q.view(b, -1, self.num_heads, self.depth).transpose(1, 2)
        K = K.view(b, -1, self.num_heads, self.depth).transpose(1, 2)
        V = V.view(b, -1, self.num_heads, self.depth).transpose(1, 2)

        # attention scores
        att = torch.einsum('bnid,bnjd->bnij', Q, K) * self.scale  # (b, heads, seq_q, seq_k)

        if pad_mask is not None:
            # pad_mask expected shape (b, seq_k) or (b, seq_q, seq_k)
            if pad_mask.dim() == 2:
                mask = pad_mask.unsqueeze(1).unsqueeze(2)  # (b,1,1,seq_k)
                att = att.masked_fill(mask, float('-inf'))
            else:
                att = att.masked_fill(pad_mask.unsqueeze(1), float('-inf'))

        att_weights = F.softmax(att, dim=-1)  # (b, heads, seq_q, seq_k)

        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)  # (b, heads, seq_q, depth)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.emb_dim)  # (b, seq_q, emb_dim)

        # reshape back to spatial
        # out_proj = self.proj_out(out_spatial)  # (b, channels, h, w)

        return out, att_weights


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.15):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))

class classifier(nn.Module):
    def __init__(self, num_classes, num_features):
        super(classifier, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
       # self.fc = nn.Linear(512*num_features, num_classes)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x_ = self.global_pool(x.permute(0, 2, 1))
        # x_ = self.global_pool(x.flatten(2))
        x = x_.view(x_.size(0), -1)
        x = self.fc(x)  #
        return x
