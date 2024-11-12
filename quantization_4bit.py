import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

class Custom4BitQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, is_activation=False):
        ctx.save_for_backward(input)
        ctx.is_activation = is_activation
        if is_activation:
            # Activation: 0에서 1 사이를 16단계로 양자화
            output = torch.clamp(input, 0, 1)
            output = torch.round(output * 15) / 15
        else:
            # 가중치: -1에서 1 사이를 16단계로 양자화
            output = torch.clamp(input, -1, 1)
            output = torch.round(output * 7.5) / 7.5
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        if ctx.is_activation:
            grad_input[(input < 0) | (input > 1)] = 0
        else:
            grad_input[input.abs() > 1] = 0
        return grad_input, None

class QuantizedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize = Custom4BitQuantization.apply

    def forward(self, x):
        quantized_weight = self.quantize(self.weight, False)
        return F.conv2d(x, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QuantizedReLU(nn.ReLU):
    def __init__(self, inplace=False):
        super().__init__(inplace)
        self.quantize = Custom4BitQuantization.apply

    def forward(self, x):
        return self.quantize(F.relu(x), True)

class QuantizedResNet18FeatureExtractor_4bit(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.conv1 = QuantizedConv2d(original_model.conv1.in_channels, original_model.conv1.out_channels, 
                                     kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = original_model.bn1
        self.relu = QuantizedReLU(inplace=True)
        self.maxpool = original_model.maxpool
        self.layer1 = self._quantize_layer(original_model.layer1)
        self.layer2 = self._quantize_layer(original_model.layer2)
        self.layer3 = self._quantize_layer(original_model.layer3)
        self.layer4 = self._quantize_layer(original_model.layer4)
        self.avgpool = original_model.avgpool
        self.fc = original_model.fc  # fc layer도 복사

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, QuantizedConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _quantize_layer(self, layer):
        return nn.Sequential(*[self._quantize_basic_block(b) for b in layer])

    def _quantize_basic_block(self, block):
        return nn.Sequential(
            QuantizedConv2d(block.conv1.in_channels, block.conv1.out_channels, 
                            kernel_size=3, stride=block.conv1.stride, padding=1, bias=False),
            block.bn1,
            QuantizedReLU(inplace=True),
            QuantizedConv2d(block.conv2.in_channels, block.conv2.out_channels, 
                            kernel_size=3, stride=1, padding=1, bias=False),
            block.bn2,
            QuantizedReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # flatten
        x = self.fc(x)  # fc layer 통과
        return x

def feature_to_distribution(features):
    # Convert features to probability distribution
    features_pos = F.relu(features)  # Ensure non-negativity
    return features_pos / features_pos.sum(dim=1, keepdim=True)
