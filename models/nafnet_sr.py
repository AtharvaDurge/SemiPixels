import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel)
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0)
        )

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1)

        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        inp = x

        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = y.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma

class NAFNetSR(nn.Module):
    def __init__(self, img_channel=1, width=32, middle_blk_num=2, enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2], scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        curr_width = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(curr_width) for _ in range(num)]))
            self.downs.append(nn.Conv2d(curr_width, curr_width * 2, kernel_size=2, stride=2))
            curr_width *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(curr_width) for _ in range(middle_blk_num)])

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(curr_width, curr_width * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2)
            ))
            curr_width = curr_width // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(curr_width) for _ in range(num)]))

        self.sr_head = nn.Sequential(
            nn.Conv2d(width, width * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(width, img_channel, kernel_size=3, padding=1)
        )

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

        x_feat = self.intro(x)
        
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x_feat = encoder(x_feat)
            encs.append(x_feat)
            x_feat = down(x_feat)

        x_feat = self.middle_blks(x_feat)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, reversed(encs)):
            x_feat = up(x_feat)
            x_feat = x_feat + enc_skip
            x_feat = decoder(x_feat)

        res = self.sr_head(x_feat)
        return base + res