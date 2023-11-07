
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

def extract_image_patches(images, ksizes, strides):
    unfold = torch.nn.Unfold(kernel_size=ksizes, padding=0, stride=strides)
    patches = unfold(images)
    return patches

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False)
        )

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        midplanes = int(inplanes//2)
        self.conv1 = conv3x3(inplanes, midplanes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv1x1(midplanes, planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out

class SPA(nn.Module):
    def __init__(self, ksize=7, stride_1=4, stride_2=4, in_channels=6, inter_channels=16):
        super(SPA, self).__init__()
        self.ksize = ksize
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        N = int(((128 - ksize) / stride_1 + 1) ** 2)
        self.conv = BasicBlock(in_channels, in_channels)
        self.sigmoid = nn.Sigmoid()
        self.g = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1,
                           padding=0)
        )
        self.W = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1,
                           padding=0)
        )
        self.theta = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        self.fc1 = nn.Linear(in_features=ksize ** 2 * inter_channels, out_features=(ksize ** 2 * inter_channels) // 4)
        self.fc2 = nn.Linear(in_features=ksize ** 2 * inter_channels, out_features=(ksize ** 2 * inter_channels) // 4)

        self.M_conv = nn.Conv1d(N, N, kernel_size=1, stride=1, padding=0)
        self.M_fc = nn.Linear(N, N)

    def forward(self, x):
        residual = x
        x1 = self.g(x)
        x2 = self.W(x)
        x3 = self.theta(x1)
        raw_int_bs = list(x1.size())

        patch_6 = extract_image_patches(x1, ksizes=[self.ksize, self.ksize],
                                                    strides=[self.stride_1, self.stride_1])
        patch_6 = patch_6.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_6 = patch_6.permute(0, 4, 1, 2, 3)
        patch_6_group = torch.split(patch_6, 1, dim=0)

        patch_4 = extract_image_patches(x2, ksizes=[self.ksize, self.ksize],
                                                      strides=[self.stride_2, self.stride_2])
        patch_4 = patch_4.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
        patch_4 = patch_4.permute(0, 4, 1, 2, 3)
        patch_4_group = torch.split(patch_4, 1, dim=0)

        patch_6_2 = extract_image_patches(x3, ksizes=[self.ksize, self.ksize],
                                        strides=[self.stride_1, self.stride_1])
        patch_6_2 = patch_6_2.view(raw_int_bs[0], raw_int_bs[1]*2, self.ksize, self.ksize, -1)
        patch_6_2 = patch_6_2.permute(0, 4, 1, 2, 3)
        patch_6_2_group = torch.split(patch_6_2, 1, dim=0)

        y = []
        spa_map = []

        for xi, wi, pi in zip(patch_6_group, patch_4_group, patch_6_2_group):

            n_s = pi.shape[1]
            wi = self.fc1(wi.view(wi.shape[1], -1))
            xi = self.fc1(xi.view(xi.shape[1], -1)).permute(1, 0)
            Ai = torch.matmul(wi, xi)
            A = Ai.unsqueeze(0)
            Ti1 = self.M_conv(A)
            Ti2 = self.M_fc(A.permute(0, 2, 1)).permute(0, 2, 1)
            Ti = (Ti1 + Ti2).squeeze(0)
            mask = self.sigmoid(Ti)
            mask_b = (mask != 0.).float()
            Ai = Ai * mask
            scale = wi.shape[1] ** -0.5
            Ai = F.softmax(Ai*scale, dim=1)
            Ai = Ai * mask_b
            pi = pi.view(n_s, -1)
            yi = torch.mm(Ai, pi)
            zi = yi.view(1, n_s, -1).permute(0, 2, 1)
            zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize),
                                          padding=0, stride=self.stride_1)
            y.append(zi)
            Ai = Ai.view(1, n_s, -1)
            spa_map.append(Ai)
        out = torch.cat(y, dim=0)
        spa_map = torch.cat(spa_map, dim=0)
        out = self.conv(out)
        out = residual + out
        return spa_map, out

class SPE(nn.Module):
    def __init__(self,in_channels, ratio):
        super(SPE, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.conv = BasicBlock(in_channels, in_channels)
        self.sigmoid = nn.Sigmoid()
        self.edge = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                              padding=0)
        self.node = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                              padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1,
                      padding=0)
        )
        self.M_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.M_fc = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        x = self.conv(x)
        residual = x
        n, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        bn = self.node(x)
        be = self.edge(x)

        bn = bn.view(n, c, h * w)
        bn_group = torch.split(bn, 1, dim=0)
        be = be.view(n, c, h * w)
        be_group = torch.split(be, 1, dim=0)
        y = []
        score_map = []
        for xi, wi, pi in zip(be_group, bn_group, be_group):

            wi = wi.mean(2).view(c, 1).detach()
            xi = xi.mean(2).view(1, c).detach()
            score = torch.matmul(wi,xi)
            A = score.unsqueeze(0)
            Ti1 = self.M_conv(A)
            Ti2 = self.M_fc(A.permute(0, 2, 1)).permute(0, 2, 1)
            Ti = (Ti1 + Ti2).squeeze(0)
            mask = self.sigmoid(Ti)
            mask_b = (mask != 0.).float()
            yi = score * mask
            scale = (h*w) ** -0.5
            yi = F.softmax(yi*scale, dim=1)
            Ai = yi * mask_b
            pi = pi.view(c, -1)
            out = torch.mm(Ai, pi)
            out = out.view(1, c, h, w)
            Ai = Ai.view(1, c, -1)
            score_map.append(Ai)
            y.append(out)
        y = torch.cat(y, dim=0)
        y = self.out(y)
        spe_map = torch.cat(score_map, dim=0)
        y = residual+y
        return spe_map, y

class iAFF(nn.Module):

    def __init__(self, channels=64, r=4, ksize=16, stride2=8):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)
        self.ksize = ksize
        self.stride_2 = stride2

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.sigmoid = nn.Sigmoid()
        self.conv = BasicBlock(channels, channels)

    def forward(self, fea, spa_graph, spe_graph):
        bt, c, h, w = fea.shape
        fea = self.conv(fea)
        fea_spe = fea
        fea_spa = fea
        fea_spe = fea_spe.view(bt, c, h * w)
        fea_spe_group = torch.split(fea_spe, 1, dim=0)
        patch_4 = extract_image_patches(fea_spa, ksizes=[self.ksize, self.ksize],
                                                      strides=[self.stride_2, self.stride_2])
        fea_spa_group = torch.split(patch_4, 1, dim=0)
        spa_graph_group = torch.split(spa_graph, 1, dim=0)
        spe_graph_group = torch.split(spe_graph, 1, dim=0)
        spa_y = []
        spe_y = []

        for fea_spe1, fea_spa1, spa_graph1, spe_graph1 in zip(fea_spe_group, fea_spa_group, spa_graph_group, spe_graph_group):
            spa_graph1 = spa_graph1.squeeze(0)
            spe_graph1 = spe_graph1.squeeze(0)
            fea_spa1 = fea_spa1.squeeze(0).permute(1, 0)
            fea_spe1 = fea_spe1.squeeze(0)
            out_spa1 = torch.mm(spa_graph1, fea_spa1)
            out_spe1 = torch.mm(spe_graph1, fea_spe1)
            out_spa1 = out_spa1.permute(1, 0)
            out_spa1 = out_spa1.unsqueeze(0)
            out_spa1 = torch.nn.functional.fold(out_spa1, (h, w), (self.ksize, self.ksize), padding=0, stride=self.stride_2)
            out_spe1 = out_spe1.view(1, c, h, w)
            spa_y.append(out_spa1)
            spe_y.append(out_spe1)
        x = torch.cat(spa_y, dim=0)
        residual = torch.cat(spe_y, dim=0)
        xa = (x + residual)*0.5
        xl = self.local_att(xa)
        wei = self.sigmoid(xl)
        out = x * wei + residual * (1 - wei) + fea

        return out

class model(nn.Module):

    def __init__(self, ksize=8, stride1=4, stride2=4, embed_dim=32, inchannels=256, num_layers=3, ratio=64):
        super(model, self).__init__()
        self.num_layers = num_layers
        self.hs_conv = BasicBlock(191, inchannels)
        self.ms_conv = BasicBlock(6, embed_dim)
        self.conv = BasicBlock(197, inchannels)

        encoder1 = [SPA(ksize=ksize, stride_1=stride1, stride_2=stride2, in_channels=embed_dim,
                        inter_channels=embed_dim//2) for _ in range(self.num_layers)]
        encoder2 = [SPE(in_channels=inchannels, ratio=ratio) for _ in range(self.num_layers)]
        self.encoder1 = nn.Sequential(*encoder1)
        self.encoder2 = nn.Sequential(*encoder2)

        aggra_encoder = [iAFF(channels=inchannels, r=4, ksize=ksize, stride2=stride2) for _ in range(self.num_layers)]
        self.aggra_encoder = nn.Sequential(*aggra_encoder)
        self.conv_out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=inchannels, out_channels=191, kernel_size=3, stride=1,
                      padding=0, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.conv_hs = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=inchannels, out_channels=191, kernel_size=3, stride=1,
                      padding=0, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.conv_ms = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=embed_dim, out_channels=6, kernel_size=3, stride=1,
                      padding=0, bias=False),
            nn.LeakyReLU(0.2)
        )
    def forward(self, ms, hs):

        fea = self.conv(torch.cat([ms, torch.nn.functional.interpolate(hs, size=(ms.shape[2], ms.shape[3]), mode='bilinear')], 1))
        hs_fea = self.hs_conv(hs)
        ms_fea = self.ms_conv(ms)

        for i in range(self.num_layers):
            spa_graph, ms_fea = self.encoder1[i](ms_fea)
            spe_graph, hs_fea = self.encoder2[i](hs_fea)
            fea = self.aggra_encoder[i](fea, spa_graph, spe_graph)
        hs_out = self.conv_hs(hs_fea)
        ms_out = self.conv_ms(ms_fea)
        out = self.conv_out(fea)

        return out, hs_out, ms_out

