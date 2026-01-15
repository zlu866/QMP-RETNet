import torch
from torch import nn
from torch.nn import functional as F
import warnings
import torchvision
import h5py
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class Conv2D(nn.Module):
    def __init__(self, in_channel=256, out_channel=8):
        super(Conv2D, self).__init__()
        self.guide_conv2D = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

    def forward(self, x):
        spatial_guidance = self.guide_conv2D(x)
        return spatial_guidance

class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            pass
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class NPM(nn.Module):
    def __init__(self, in_channel):
        super(NPM, self).__init__()
        self.in_channel = in_channel
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv0_33 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.conv0_11 = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.conv_0_cat = nn.Conv2d(in_channel * 2, in_channel, 3, 1, 1)
        self.conv2_33 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.conv2_11 = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.conv_2_cat = nn.Conv2d(in_channel * 2, in_channel, 3, 1, 1)
        self.conv4_33 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.conv4_11 = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.conv_4_cat = nn.Conv2d(in_channel * 2, in_channel, 3, 1, 1)
        self.conv_cat = nn.Conv2d(in_channel * 3, in_channel, 3, 1, 1)

    def forward(self, x):
        x_0 = x
        x_2 = F.avg_pool2d(x, 2, 2)
        x_4 = F.avg_pool2d(x_2, 2, 2)

        x_0 = torch.cat([self.conv0_33(x_0), self.conv0_11(x_0)], 1)
        x_0 = self.activation(self.conv_0_cat(x_0))

        x_2 = torch.cat([self.conv2_33(x_2), self.conv2_11(x_2)], 1)
        x_2 = F.interpolate(self.activation(self.conv_2_cat(x_2)), scale_factor=2, mode='bilinear')

        x_4 = torch.cat([self.conv2_33(x_4), self.conv2_11(x_4)], 1)
        x_4 = F.interpolate(self.activation(self.conv_4_cat(x_4)), scale_factor=4, mode='bilinear')

        x = x + self.activation(self.conv_cat(torch.cat([x_0, x_2, x_4], 1)))
        return x


class CRM(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CRM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DRCR_Block(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size=3, stride=1, padding=1, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=False):
        super(DRCR_Block, self).__init__()
        # dense convolutions
        self.conv1 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv2 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv3 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv4 = Conv2dLayer(in_channels * 2, in_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv5 = Conv2dLayer(in_channels * 2, in_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv6 = Conv2dLayer(in_channels * 2, in_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        # self.cspn2_guide = GMLayer(in_channels)
        # self.cspn2 = Affinity_Propagate_Channel()
        self.se1 = CRM(in_channels)
        self.se2 = CRM(in_channels)


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # guidance2 = self.cspn2_guide(x3)
        # x3_2 = self.cspn2(guidance2, x3)
        x3_2 = self.se1(x)
        x4 = self.conv4(torch.cat((x3, x3_2), 1))
        x5 = self.conv5(torch.cat((x2, x4), 1))
        x6 = self.conv6(torch.cat((x1, x5), 1)) + self.se2(x3_2)
        return x6

class down_sample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_sample, self).__init__()

        self.do = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.do(x)


class up_sample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_sample, self).__init__()

        self.do = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(p=0.5)
        )
    def forward(self, x):
        return self.do(x)


class DRCR_Block_1(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(DRCR_Block_1, self).__init__()

        self.down1 = down_sample(input_nc, 64)
        self.down2 = down_sample(64, 128)
        self.down3 = down_sample(128, 256)
        self.down4 = down_sample(256, 512)
        self.down5 = down_sample(512, 512)
        self.down6 = down_sample(512, 512)
        # self.down7 = down_sample(512, 512)
        #  self.down8 = down_sample(512, 512)
        self.up1 = up_sample(512, 512)
        self.up2 = up_sample(1024, 512)
        self.up3 = up_sample(1024, 512)
        #  self.up4 = up_sample(1024, 512)
        self.up4 = up_sample(512 + 256, 256)  # up3 + x3
        self.up5 = up_sample(256 + 128, 128)  # up4 + x2
        # self.up6 = up_sample(128 + 64, 64)  # up5 + x1
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(self.up1, self.up2, self.up3, self.up4, self.up5, self.final)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.down6(x5)
        #   x = self.down8(x7)
        # x8 = torch.cat((x5, self.up1(x)), dim=1)
        u1 = self.up1(x)
        u2 = self.up2(torch.cat([u1, x5], dim=1))
        u3 = self.up3(torch.cat([u2, x4], dim=1))
        u4 = self.up4(torch.cat([u3, x3], dim=1))
        u5 = self.up5(torch.cat([u4, x2], dim=1))
     #   u6 = self.up6(torch.cat([u5, x1], dim=1))
        out = self.final(u5)

        return out, x5


class DRCR(nn.Module):
    def __init__(self, inplanes=2, planes=16, channels=140, n_DRBs=8):
        super(DRCR, self).__init__()
        self.input_conv2D = Conv3x3(inplanes, channels, 3, 1)
        self.input_prelu2D = nn.PReLU()
        self.head_conv2D = Conv3x3(channels, channels, 3, 1)
        self.denosing = NPM(channels)
        self.backbone = nn.ModuleList(
            [DRCR_Block(channels, channels) for _ in range(n_DRBs)])
        self.backbone_1 = DRCR_Block_1(channels, channels)
        self.tail_conv2D = Conv3x3(channels, channels, 3, 1)
        self.output_prelu2D = nn.PReLU()
        self.output_conv2D = Conv3x3(channels, planes, 3, 1)
        self.w_gen_R = W_Generator(channels, planes = planes)
        self.w_gen_G = W_Generator(channels, planes = planes)
        # Read QE curves
        # mat = h5py.File('../Dataset/QE_curves/rgb_QE.mat')  # hdf5.loadmat
        # tmp = torch.tensor(np.float32(np.array(mat['tmp']))).cuda()
        # self.QE_curves = tmp[:,15:]

    def forward(self, x):
        x = x[:, :2, :, :]
        img_out, feature, w_out_R, w_out_G = self.DRN2D(x)
        return img_out, feature, w_out_R, w_out_G

    def DRN2D(self, x):
        out = self.input_prelu2D(self.input_conv2D(x))
        out = self.head_conv2D(out)
        denosing_out = self.denosing(out)

        out = denosing_out
        for i, block in enumerate(self.backbone):
            out = block(out)
        out, feature = self.backbone_1(out)
        img_out = self.tail_conv2D(out)
        img_out = self.output_conv2D(self.output_prelu2D(img_out))

        w_out_R = self.w_gen_R(denosing_out) #+ self.QE_curves[0,:].repeat(denosing_out.size(0),1,1,1).permute(0,3,1,2)
        w_out_G = self.w_gen_G(denosing_out) #+ self.QE_curves[1,:].repeat(denosing_out.size(0),1,1,1).permute(0,3,1,2)
        return img_out, feature, w_out_R, w_out_G


class DRCR_net(nn.Module):
    def __init__(self, device, ckpt):
        super(DRCR_net, self).__init__()
        self.device = device
        self.net_G = DRCR(2, 7, 21, 1)
        checkpoint = torch.load(ckpt)
        self.net_G.load_state_dict(checkpoint['state_dict'], strict=False)
        self.freeze_by_name(self.net_G,
                            module_names=[
                                "w_gen_R","w_gen_G","backbone",
                                "input_conv2D","input_prelu2D",
                                "head_conv2D", "denosing","tail_conv2D",
                                "output_prelu2D","output_conv2D",
                                "backbone_1.decoder"])
        self.criterion = Loss_train().to(self.device)

    def freeze_by_name(self, model, module_names):
        for name, module in model.named_modules():
            if name in module_names:
                for p in module.parameters():
                    p.requires_grad = False

    def forward(self, x):
        self.real = x[:, :2, :, :]
        self.img_out, out, self.w_out_R, self.w_out_G = self.net_G(x)
        RGB_out = self.backward_DRCR()
        return out, RGB_out#, self.img_out

    def backward_DRCR(self):
        QE_R_0 = self.w_out_R[:, :6, :, :]
        QE_G_0 = self.w_out_G[:, :6, :, :]

        QE_R = torch.tensor(QE_R_0.expand_as(self.img_out[:, :6, :, :])).to(device)
        QE_G = torch.tensor(QE_G_0.expand_as(self.img_out[:, :6, :, :])).to(device)
        R_out = torch.sum(self.img_out[:, :6, :, :] * QE_R, dim=1, keepdim=True)
        G_out = torch.sum(self.img_out[:, :6, :, :] * QE_G, dim=1, keepdim=True)
        RGB_out = torch.cat([R_out, G_out], dim=1)
        return RGB_out
        # self.loss_cy = 0.2*self.criterion(RGB_out, self.real) + loss
        # self.loss_cy.backward(retain_graph=True)


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self, x, loss, optimizer):
        self.forward(x)                   # compute fake images: G(A)
        # update D
        self.set_requires_grad([self.net_G], False)  # D requires no gradients when optimizing G
        optimizer.zero_grad()        # set G's gradients to zero
        self.backward_DRCR(loss)                   # calculate graidents for G
        # print(self.k[0].conv2d.weight.grad)



class W_Generator(nn.Module):
    def __init__(self, channel, reduction=8, planes=10):
        super(W_Generator, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, planes, bias=False),
            nn.Sigmoid()
        )
        self.planes = planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.planes, 1, 1)
        return y

class Loss_train(nn.Module):
    def __init__(self):
        super(Loss_train, self).__init__()

    def forward(self, outputs, label):
        #label = torch.where(label == 0, 1e-8, label)
        # if label.min() <= 0:
        #     # print('Found pixel=0 in label images!')
        #     label[label<=0]=1e-2
        error = torch.abs(outputs - label) / (label + 1e-2)#label.size(0)#
        mrae = torch.mean(error.view(-1))
        return mrae


#### Define Network used for Perceptual Loss
class LossNetwork(nn.Module):
    def __init__(self, use_bn=False, use_input_norm=True, device=torch.device('cuda')):
        super(LossNetwork, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        self.vgg_layers = model
        self.layer_name_mapping = {
            # '3': "relu1_2",
            # '8': "relu2_2",
            '13': "relu3_2",
            '22': "relu4_2",
            '31': "relu5_2"
        }
        self.weight = [0.8, 1.0]

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                # print("vgg_layers name:", name, module)
                output[self.layer_name_mapping[name]] = x
        # print(output.keys())
        return list(output.values())

    def forward(self, output, gt):
        output_features = []
        gt_features = []
        for channel in range(output.size(1)):
            f_output = output[:, channel, ].unsqueeze(1)
            f_output = (f_output - self.mean) / self.std
            output_features.append(self.output_features(f_output))
            f_gt = gt[:, channel, ].unsqueeze(1)
            f_gt = (f_gt - self.mean) / self.std
            gt_features.append(self.output_features(f_gt))
        # aaa=torch.sum(gt_features[0][1][0,], dim=0)
        # plt.imshow(aaa.cpu().detach().numpy())
        # plt.show()
        loss = []
        for m in range(len(output_features)):
            for n, (dehaze_feature, gt_feature, loss_weight) in enumerate(
                    zip(output_features[m], gt_features[m], self.weight)):
                # loss.append(F.mse_loss(dehaze_feature, gt_feature) * loss_weight)
                error = torch.abs(dehaze_feature - gt_feature) / (gt_feature + 1e-3)  # .size(0)
                mrae = torch.mean(error.view(-1))
                loss.append(mrae * loss_weight)
        return sum(loss), gt_features  # /len(loss)




if __name__ == "__main__":
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    input_tensor = torch.rand(1, 3, 128, 128)
    model = DRCR(3, 31, 100, 10)
    # model = nn.DataParallel(model).cuda()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    print(output_tensor.size())
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    print(torch.__version__)





