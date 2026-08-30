import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV2_UNet(nn.Module):
    def __init__(self, out_channels=1):
        super(MobileNetV2_UNet, self).__init__()
        
        # Load MobileNetV2 pretrained on ImageNet
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        features = mobilenet.features
        
        # Extract specific layers for skip connections
        self.enc1 = features[0:2]   # 1/2 resolution, 16 channels
        self.enc2 = features[2:4]   # 1/4 resolution, 24 channels
        self.enc3 = features[4:7]   # 1/8 resolution, 32 channels
        self.enc4 = features[7:14]  # 1/16 resolution, 96 channels
        self.enc5 = features[14:]   # 1/32 resolution, 1280 channels (or 320 depending on cut)
        
        # Up-convolutions
        self.up4 = nn.ConvTranspose2d(1280, 96, kernel_size=2, stride=2)
        self.dec4 = self._make_dec_block(96 + 96, 96)
        
        self.up3 = nn.ConvTranspose2d(96, 32, kernel_size=2, stride=2)
        self.dec3 = self._make_dec_block(32 + 32, 32)
        
        self.up2 = nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2)
        self.dec2 = self._make_dec_block(24 + 24, 24)
        
        self.up1 = nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2)
        self.dec1 = self._make_dec_block(16 + 16, 16)
        
        self.up0 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        
    def _make_dec_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Decoder with skip connections
        d4 = self.up4(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        d0 = self.up0(d1)
        
        out = self.final_conv(d0)
        return torch.sigmoid(out) # 1 channel output, P(pixel belongs to lane)
