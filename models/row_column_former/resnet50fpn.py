from mmcv.cnn import ConvModule
import torch
import torch.nn as nn
from timm import create_model

class ResNet50FPN(torch.nn.Module):
    def __init__(self, 
                 out_channels=256):
        """
        ResNet50 with Feature Pyramid Network (FPN).
        
        Args:
            out_channels (int): Number of output channels for the FPN layers.
        """
        super(ResNet50FPN, self).__init__()
        
        # ResNet Backbone
        self.backbone = create_model("resnet50.a1_in1k", pretrained=True, features_only=True)

        # FPN Layers
        self.fpn_convs = nn.ModuleList()
        for i in range(len(self.backbone.feature_info.channels())):
            self.fpn_convs.append(
                ConvModule(
                    in_channels=self.backbone.feature_info.channels()[i],
                    out_channels=out_channels,
                    kernel_size=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    act_cfg=dict(type='ReLU'),
                )
            )
        
        # Upsample module for FPN
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Forward pass through ResNet50 and FPN.

        Args:
            x (Tensor): Input image tensor (batch_size, in_channels, height, width).
        
        Returns:
            list[Tensor]: List of feature maps from the FPN.
        """
        # Extract features from ResNet stages
        features = self.backbone(x)  # Returns a list of feature maps
        
        # Apply FPN layers
        fpn_features = []
        for i in range(len(features)):
            fpn_features.append(self.fpn_convs[i](features[i]))
        
        # Upsample and combine feature maps
        for i in range(len(fpn_features) - 1, 0, -1):
            fpn_features[i - 1] += self.upsample(fpn_features[i])
        
        return fpn_features[-1]


if __name__ == "__main__":
    model = ResNet50FPN()
    input_tensor = torch.randn(1, 3, 640, 384)
    output = model(input_tensor)
