import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization
from torch.ao.quantization import QuantStub, DeQuantStub

class QuantizedResNet18(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # Set up quantization configuration
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Copy the original model structure
        self.conv1 = nn.Conv2d(original_model.conv1.in_channels, 
                              original_model.conv1.out_channels,
                              kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = original_model.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = original_model.maxpool
        
        self.layer1 = self._copy_layer(original_model.layer1)
        self.layer2 = self._copy_layer(original_model.layer2)
        self.layer3 = self._copy_layer(original_model.layer3)
        self.layer4 = self._copy_layer(original_model.layer4)
        
        self.avgpool = original_model.avgpool
        self.fc = original_model.fc
        
        # Copy weights
        self.conv1.weight.data = original_model.conv1.weight.data
        
    def _copy_layer(self, layer):
        return nn.Sequential(*[self._copy_basic_block(b) for b in layer])
        
    def _copy_basic_block(self, block):
        return nn.Sequential(
            nn.Conv2d(block.conv1.in_channels, block.conv1.out_channels,
                     kernel_size=3, stride=block.conv1.stride,
                     padding=1, bias=False),
            block.bn1,
            nn.ReLU(inplace=True),
            nn.Conv2d(block.conv2.in_channels, block.conv2.out_channels,
                     kernel_size=3, stride=1, padding=1, bias=False),
            block.bn2,
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.quant(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        x = self.dequant(x)
        return x

def prepare_and_quantize_model(model, calibration_data_loader):
    """
    Prepare and quantize the model using PyTorch's built-in quantization
    """
    # Set model to evaluation mode
    model.eval()
    
    # Setup quantization configuration
    model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model for quantization
    model_prepared = torch.ao.quantization.prepare(model)
    
    # Calibrate with training data
    with torch.no_grad():
        for batch, _ in calibration_data_loader:
            model_prepared(batch)
    
    # Convert to quantized model
    model_quantized = torch.ao.quantization.convert(model_prepared)
    
    return model_quantized

# Example usage
def quantize_resnet18(original_model, calibration_data_loader):
    """
    Quantize ResNet18 model using PyTorch's native 8-bit quantization
    """
    # Create quantization-ready model
    quantized_model = QuantizedResNet18(original_model)
    
    # Fuse layers for better quantization
    model_fused = torch.ao.quantization.fuse_modules(quantized_model, 
        [['conv1', 'bn1', 'relu']], inplace=False)
    
    # Prepare and quantize the model
    model_quantized = prepare_and_quantize_model(model_fused, calibration_data_loader)
    
    return model_quantized