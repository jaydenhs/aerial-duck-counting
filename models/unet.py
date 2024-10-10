import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels=64):
        super(UNet, self).__init__()

        # Down-sampling path (Encoder)
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)

        # Up-sampling path (Decoder)
        self.upconv4 = self.upconv(base_channels * 16, base_channels * 8)
        self.dec4 = self.conv_block(base_channels * 16, base_channels * 8)
        self.upconv3 = self.upconv(base_channels * 8, base_channels * 4)
        self.dec3 = self.conv_block(base_channels * 8, base_channels * 4)
        self.upconv2 = self.upconv(base_channels * 4, base_channels * 2)
        self.dec2 = self.conv_block(base_channels * 4, base_channels * 2)
        self.upconv1 = self.upconv(base_channels * 2, base_channels)
        self.dec1 = self.conv_block(base_channels * 2, base_channels)

        # Final output: number of output channels is equal to number of classes
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        up4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat((up4, enc4), dim=1))
        up3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))
        up2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))

        return self.final_conv(dec1)  # Output shape: (N, num_classes, H, W)