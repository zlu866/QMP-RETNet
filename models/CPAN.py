import torch
import torch.nn as nn
from einops import rearrange
import torchvision.models as tvmodels

from backbone import Resblock, SE_Block, Mlp, CrossAttention, classifier

from pretrain_FFA import Pix2PixFFAModel
from DRCR import DRCR_net
from pretrain_sm import UNet

class FeatureSelection1(nn.Module):
    def __init__(self, inchannel, outchannel, num_inputs):
        super().__init__()
        self.blocks = nn.ModuleList([Resblock(inchannel, outchannel) for _ in range(num_inputs)])
        self.SE = SE_Block(outchannel * num_inputs, ratio=16)

    def forward(self, x_list):
        feats = [block(x) for block, x in zip(self.blocks, x_list)]
        return self.SE(torch.cat(feats, dim=1))


class FeatureSelection2(nn.Module):
    def __init__(self, inchannel, outchannel, num_inputs):
        super().__init__()
        self.SE = SE_Block(inchannel, ratio=16)
        self.out = nn.Conv2d(inchannel, outchannel // 2, 1, bias=False)

    def forward(self, x):
        return self.out(self.SE(x))


class FeatureSelection3(nn.Module):
    def __init__(self, inchannel, outchannel, num_inputs):
        super().__init__()
        self.SE = SE_Block(inchannel, ratio=16)
        self.out = nn.Conv2d(inchannel, outchannel // 6, 1, bias=False)

    def forward(self, x):
        return self.out(self.SE(x))



class Featurefusion1(nn.Module):
    def __init__(self, channal1, channal2, num_heads, num_feature):
        super(Featurefusion1, self).__init__()
        self.CrossAttention = CrossAttention(channal1, channal1, num_heads, num_feature)
        self.Norm = nn.LayerNorm(channal1, channal1)
        # self.MLP = Mlp(channal1, channal1, channal1)

    def forward(self, x, feature):
        y_att, att_w = self.CrossAttention(x, feature)
        f_att = self.Norm(y_att + x.flatten(2).permute(0, 2, 1))
        # out = (f_att + y_att + x).flatten(2).permute(0, 2, 1)
        f_att = rearrange(f_att, 'b (h w) d -> b d h w', h=16, w=16)  # (b, emb_dim, h, w)
        return f_att, att_w  #out.permute(0, 2, 1)


class Featurefusion2(nn.Module):
    def __init__(self, channal1, channal2, num_heads, num_feature):
        super(Featurefusion2, self).__init__()
        self.CrossAttention = CrossAttention(channal1, channal1, num_heads, num_feature)
        self.Norm = nn.LayerNorm(channal1, channal1)
        # self.MLP = Mlp(channal1, channal1, channal1)

    def forward(self, x, feature):
        y_att, att_w = self.CrossAttention(x, feature)
        f_att = self.Norm(y_att + x.flatten(2).permute(0, 2, 1))
        # out = (f_att + y_att + x).flatten(2).permute(0, 2, 1)
        f_att = rearrange(f_att, 'b (h w) d -> b d h w', h=16, w=16)  # (b, emb_dim, h, w)
        return f_att, att_w  #out.permute(0, 2, 1)

class Featurefusion3(nn.Module):
    def __init__(self, channal1, channal2, num_heads, num_feature):
        super(Featurefusion3, self).__init__()
        self.CrossAttention = CrossAttention(channal1, channal1, num_heads, num_feature)
        self.Norm = nn.LayerNorm(channal1, channal1)

        # self.MLP = Mlp(channal1, channal1, channal1)

    def forward(self, x, feature):
        y_att, att_w = self.CrossAttention(x, feature)
        f_att = self.Norm(y_att + x.flatten(2).permute(0, 2, 1))
        # out = (f_att + y_att + x).flatten(2).permute(0, 2, 1)
        # f_att = rearrange(f_att, 'b (h w) d -> b d h w', h=16, w=16)  # (b, emb_dim, h, w)
        return f_att, att_w  #out.permute(0, 2, 1)


class MultiFeatureProcessor(nn.Module):
    def __init__(self, img_channel: int, num_features: int = 6):
        super().__init__()

        self.selectors = nn.ModuleList([
            FeatureSelection1(img_channel, img_channel, num_features),
            FeatureSelection2(img_channel * num_features, img_channel * num_features, num_features),
            FeatureSelection3(img_channel * num_features // 2, img_channel * num_features, num_features)
        ])

        self.fusion_levels = nn.ModuleList([
            nn.ModuleList([
                Featurefusion1(img_channel, img_channel, num_heads=1, num_feature=6),
                Featurefusion2(img_channel, img_channel, num_heads=1, num_feature=3),
                Featurefusion3(img_channel, img_channel, num_heads=1, num_feature=1)
            ]) for _ in range(num_features)
        ])

    def forward(self, z_list):
        z_hwc = []
        for z in z_list:
            b, hw, c = z.shape
            side = int(hw ** 0.5)
            assert side * side == hw, f"expect HW to be square, got {hw}"
            z_ = z.permute(0, 2, 1).contiguous().view(b, c, side, side)  # (B, C, H, W)
            z_hwc.append(z_)

        feature_maps = self.selectors[0](z_hwc)  # (B, C * num_inputs, H, W)
        f = [feature_maps]
        for selector in self.selectors[1:]:
            feature_maps = selector(feature_maps)
            f.append(feature_maps)

        outputs, attentions = [], []
        for i, fusion_modules in enumerate(self.fusion_levels):
            y, att_w = fusion_modules[0](z_hwc[i], f[0])
            for level in range(1, 3):
                y, att_w = fusion_modules[level](y.permute(0, 2, 1).view(y.size(0), -1, 1).permute(0,2,1) if False else y, f[level])

            outputs.append(y)
            attentions.append(att_w)

        return outputs, attentions


# ---------------------- Classifier & CPAN ----------------------
class ClassifierHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.global_pool(x.permute(0, 2, 1)).view(x.size(0), -1)
        return self.fc(x)


class CPAN(nn.Module):
    def __init__(self, in_channels, outputs, num_features,

                 device, pretrained=True, heads=16, dropout=0.15):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.outputs = outputs
        self.heads = heads
        self.num_features = num_features
        self.pretrained = pretrained
        self.dropout = dropout
        ckpt_path={
            "ckpt_pathG_A":" ",
            "ckpt_pathD_A":" ",
            "ckpt_pathG_B":" ",
            "ckpt_pathD_B":" ",
            "ckpt_pathc":" ",
            "ckpt_pathd":"",
            "ckpt_pathe":" "
        }
        self.pix2pix1 = Pix2PixFFAModel(device, ckpt_path["ckpt_pathG_A"], ckpt_path["ckpt_pathD_A"])
        self.pix2pix2 = Pix2PixFFAModel(device, ckpt_path["ckpt_pathG_B"], ckpt_path["ckpt_pathD_B"])

        self.DRCR1 = DRCR_net(device, ckpt_path["ckpt_pathe"])
        self.pre_img_net = tvmodels.resnet18(pretrained=pretrained)
        self.pre_img = nn.Sequential(
            self.pre_img_net.conv1, self.pre_img_net.bn1, self.pre_img_net.relu, self.pre_img_net.maxpool,
            self.pre_img_net.layer1, self.pre_img_net.layer2, self.pre_img_net.layer3, self.pre_img_net.layer4
        )

        # UNet for c/d
        pre_c = UNet(in_channels, 4)
        pre_d = UNet(in_channels, 3)
        pre_c.load_state_dict(torch.load(ckpt_path["ckpt_pathc"]))

        state_dict = torch.load(ckpt_path["ckpt_pathd"])
        state_dict.pop('outc.conv.weight')
        state_dict.pop('outc.conv.bias')
        pre_d.load_state_dict(state_dict, strict=False)
        self.reduce_feature_maps = nn.Conv2d(1024, 512, kernel_size=1, stride=2, bias=False)

        self.pre_c = nn.Sequential(pre_c.inc, pre_c.down1, pre_c.down2, pre_c.down3, pre_c.down4)
        self.pre_d = nn.Sequential(pre_d.inc, pre_d.down1, pre_d.down2, pre_d.down3, pre_d.down4)

        for m in [self.pre_c, self.pre_d, self.pre_img]:
            for param in m.parameters():
                param.requires_grad = False

        # feature fusion blocks
        img_channel = 512
        self.feature_f = MultiFeatureProcessor(img_channel=img_channel, num_features=num_features)
        self.MLP = Mlp(img_channel * num_features, img_channel, img_channel)
        self.layernorm = nn.LayerNorm(img_channel * num_features)
        # self.classifier = classifier(outputs, num_features)
        self.classifier = ClassifierHead(outputs)

    def forward(self, img):

        z_a, fake_A = self.pix2pix1(img)
        z_b, fake_B = self.pix2pix2(img)

        z_img = self.pre_img(img)  # (B, C_img, H', W')

        z_c = self.pre_c(img)
        z_c = self.reduce_feature_maps(z_c)  # -> (B, 512, H', W')
        z_d = self.pre_d(img)
        z_d = self.reduce_feature_maps(z_d)

        z_e, RGB_out = self.DRCR1(img[:, :2, :, :])

        # ---------------- flatten -> (B, HW, C) ----------------
        def flatten(z):
            b, c, h, w = z.size()
            return z.view(b, c, -1).permute(0, 2, 1)

        # [z_a, z_b, z_img, z_c, z_d, z_e]
        z_a_f, z_b_f, z_img_f, z_c_f, z_d_f, z_e_f = map(flatten, [z_a, z_b, z_img, z_c, z_d, z_e])

        outputs, attentions = self.feature_f([z_a_f, z_b_f, z_img_f, z_c_f, z_d_f, z_e_f])

        fused = torch.cat(outputs, dim=2)   # (B, HW, 512 * num_features)
        fused = self.MLP(fused)
        return self.classifier(fused), fake_A, fake_B, RGB_out

import PIL
import numpy as np
import torchvision
if __name__ == '__main__':
    feature_maps = []
    gradients = []
    ckpt_path1 ='/mnt/home/zlu/code/Retinal_CPANet/pre-trained model/600_net_G_A.pth'
    ckpt_path2 ='/mnt/home/zlu/code/Retinal_CPANet/pre-trained model/600_net_D_A.pth'
    ckpt_path3 ='/mnt/home/zlu/code/Retinal_CPANet/pre-trained model/600_net_G_V.pth'
    ckpt_path4 = '/mnt/home/zlu/code/Retinal_CPANet/pre-trained model/600_net_D_V.pth'
    ckpt_path5 = '/mnt/home/zlu/code/Retinal_CPANet/pre-trained model/checkpoint_epoch_lesion.pth'
    ckpt_path6 = '/mnt/home/zlu/code/Retinal_CPANet/pre-trained model/checkpoint_epoch_optic.pth'
    ckpt_path7 = '/mnt/home/zlu/code/Retinal_CPANet/pre-trained model/net_153epoch.pth'
    in_channels = 3
    outputs = 20
    num_features = 6
    device = torch.device(f'cuda:{0}')
    # device = torch.device('cpu')
    image = PIL.Image.open('/mnt/home/zlu/code/MuReD/img_512/437.png').convert('RGB')
    image = np.array(image, dtype=np.uint8)
    t = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = t(image).unsqueeze(0)
    mask = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    mask = torch.tensor(mask).unsqueeze(0)
    model = CPAN(in_channels, outputs, num_features, ckpt_path1, ckpt_path2, ckpt_path3, ckpt_path4, ckpt_path5,ckpt_path6, ckpt_path7, device=device)
    model = model.to(device)
    for k in model.state_dict():
        print(k)
    for name, para in model.named_parameters():
        print(name, ':', para.size())
    output = model(img.to(device))