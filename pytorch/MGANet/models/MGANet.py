import torch
from torch import nn, cat
import numpy as np
from models.DenseBiasNet_state_b import DenseBiasNet
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self, features, n_atten):
        super(VGG, self).__init__()
        self.features = features
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, n_atten),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = self.sigmoid(x)
        return x

def make_layers(cfg, in_channels, gn=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if gn:
                layers += [conv3d, nn.GroupNorm(v//4, v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [8, 'M', 16, 'M', 32, 'M', 64, 'M', 128, 'M']
}

def Meta_perceiver(n_atten, in_channels):
    return VGG(make_layers(cfg['A'], in_channels, gn=True), n_atten)

class MGANet(nn.Module):
    def __init__(self, n_channels, n_classes, num_learners, checkpoint_dir, model_name):
        super(MGANet, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.num_learners = num_learners
        self.model_name = model_name
        self.meta_perceiver = Meta_perceiver(16*4*5+5, 16*4+1)
        self.wwwc = [[768, 1280],
                    [1280, 1024],
                    [512, 1280],
                    [1280, 1280]]
        self.densebisanet_0 = DenseBiasNet(n_channels, n_classes)
        self.densebisanet_1 = DenseBiasNet(n_channels, n_classes)
        self.densebisanet_2 = DenseBiasNet(n_channels, n_classes)
        self.densebisanet_3 = DenseBiasNet(n_channels, n_classes)
        for p in self.densebisanet_0.parameters():  # reset requires_grad
            p.requires_grad = False  # they are set to False below in netG update
        for p in self.densebisanet_1.parameters():  # reset requires_grad
            p.requires_grad = False  # they are set to False below in netG update
        for p in self.densebisanet_2.parameters():  # reset requires_grad
            p.requires_grad = False  # they are set to False below in netG update
        for p in self.densebisanet_3.parameters():  # reset requires_grad
            p.requires_grad = False  # they are set to False below in netG update

        self.out_conv = nn.Conv3d(16*4, n_classes, 1)
        self.softmax = nn.Softmax(dim=1)

    def load(self):
        ww = self.wwwc[0][0]
        wc = self.wwwc[0][1]
        self.densebisanet_0.load_state_dict(torch.load(
            '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, self.model_name + "_" + str(ww) + "_" + str(wc)+'_aug',
                                           "800")))
        ww = self.wwwc[1][0]
        wc = self.wwwc[1][1]
        self.densebisanet_1.load_state_dict(torch.load(
            '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, self.model_name + "_" + str(ww) + "_" + str(wc)+'_aug',
                                           "800")))
        ww = self.wwwc[2][0]
        wc = self.wwwc[2][1]
        self.densebisanet_2.load_state_dict(torch.load(
            '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, self.model_name + "_" + str(ww) + "_" + str(wc)+'_aug',
                                           "800")))
        ww = self.wwwc[3][0]
        wc = self.wwwc[3][1]
        self.densebisanet_3.load_state_dict(torch.load(
            '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, self.model_name + "_" + str(ww) + "_" + str(wc)+'_aug',
                                           "800")))
    def forward(self, x):
        x = x * 2048
        Z = x.size()[2]
        Y = x.size()[3]
        X = x.size()[4]
        diffZ = (16 - x.size()[2] % 16) % 16
        diffY = (16 - x.size()[3] % 16) % 16
        diffX = (16 - x.size()[4] % 16) % 16

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        with torch.no_grad():
            ww = self.wwwc[0][0]
            wc = self.wwwc[0][1]
            top = wc + ww // 2
            down = wc - ww // 2
            x_ = torch.where(x < down, torch.full_like(x, down), x)
            x_ = torch.where(x_ > top, torch.full_like(x, top), x_)
            x_ = (x_ - down)/ww
            f_0 = self.densebisanet_0.densebisanet(x_)

            ww = self.wwwc[1][0]
            wc = self.wwwc[1][1]
            top = wc + ww // 2
            down = wc - ww // 2
            x_ = torch.where(x < down, torch.full_like(x, down), x)
            x_ = torch.where(x_ > top, torch.full_like(x, top), x_)
            x_ = (x_ - down)/ww
            f_1 = self.densebisanet_1.densebisanet(x_)

            ww = self.wwwc[2][0]
            wc = self.wwwc[2][1]
            top = wc + ww // 2
            down = wc - ww // 2
            x_ = torch.where(x < down, torch.full_like(x, down), x)
            x_ = torch.where(x_ > top, torch.full_like(x, top), x_)
            x_ = (x_ - down)/ww
            f_2 = self.densebisanet_2.densebisanet(x_)

            ww = self.wwwc[3][0]
            wc = self.wwwc[3][1]
            top = wc + ww // 2
            down = wc - ww // 2
            x_ = torch.where(x < down, torch.full_like(x, down), x)
            x_ = torch.where(x_ > top, torch.full_like(x, top), x_)
            x_ = (x_ - down)/ww
            f_3 = self.densebisanet_3.densebisanet(x_)

            x = x/2048
            f = cat([f_0, f_1, f_2, f_3, x], dim=1)

        x_0 = self.densebisanet_0.out_conv(f[:, 0:16, :, :, :])
        x_1 = self.densebisanet_1.out_conv(f[:, 16:32, :, :, :])
        x_2 = self.densebisanet_2.out_conv(f[:, 32:48, :, :, :])
        x_3 = self.densebisanet_3.out_conv(f[:, 48:64, :, :, :])

        atten_vector = self.meta_perceiver(f)
        atten_vector = atten_vector[:, :, np.newaxis, np.newaxis, np.newaxis]
        f = f[:, 0:64, :, :, :]
        f_0 = torch.sum(f * atten_vector[:, 64*0:64*1, :, :, :], dim=1, keepdim=True)+atten_vector[:, 64*5:64*5+1, :, :, :]
        f_1 = torch.sum(f * atten_vector[:, 64*1:64*2, :, :, :], dim=1, keepdim=True)+atten_vector[:, 64*5+1:64*5+2, :, :, :]
        f_2 = torch.sum(f * atten_vector[:, 64*2:64*3, :, :, :], dim=1, keepdim=True)+atten_vector[:, 64*5+2:64*5+3, :, :, :]
        f_3 = torch.sum(f * atten_vector[:, 64*3:64*4, :, :, :], dim=1, keepdim=True)+atten_vector[:, 64*5+3:64*5+4, :, :, :]
        f_4 = torch.sum(f * atten_vector[:, 64*4:64*5, :, :, :], dim=1, keepdim=True)+atten_vector[:, 64*5+4:64*5+5, :, :, :]
        f_x = cat([f_0, f_1, f_2, f_3, f_4], dim=1)

        x = x_0 + x_1 + x_2 + x_3 + f_x
        x = self.softmax(x)
        return x[:, :, diffZ//2: Z+diffZ//2, diffY//2: Y+diffY//2, diffX // 2:X + diffX // 2]

