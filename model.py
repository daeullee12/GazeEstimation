import torch
import torch.nn as nn 
import numpy as np
import math
import copy
from torchvision.models import resnet18  # Import ResNet18 from torchvision
from efficientnet_pytorch import EfficientNet  # Import EfficientNet
from torchvision.models import mobilenet_v2  # Import MobileNetV2

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        if pos.dim() == 1:
            pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        elif pos.dim() == 2:
            pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + pos
        

    def forward(self, src, pos):
        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Model(nn.Module):
    def __init__(self, config, backbone='mobilenet_v2'):
        super(Model, self).__init__()
        maps = 32
        nhead = 8
        dim_feature = 7*7
        dim_feedforward=512
        dropout = 0.1
        num_layers=6


        if backbone == 'efficientnet-b0':
            self.base_model = EfficientNet.from_pretrained(backbone)  # Initialize EfficientNet-B0 with pre-trained weights
            self.base_model._fc = nn.Identity()  # Remove the final fully connected layer
            self.base_model_output_size = 1280  # EfficientNet-B0 output size
            self.conv = nn.Sequential(
                nn.Conv2d(self.base_model_output_size, maps, 1),  # 1x1 convolution to reduce the number of channels to `maps`
                nn.BatchNorm2d(maps),                             # Batch normalization
                nn.ReLU(inplace=True)                             # ReLU activation
            )
        elif backbone == 'mobilenet_v2':
            self.base_model = mobilenet_v2(pretrained=True)  # Initialize MobileNetV2 with pre-trained weights
            self.base_model.classifier = nn.Identity()  # Remove the final fully connected layer
            self.base_model_output_size = 1280  # MobileNetV2 output size
            self.conv = nn.Sequential(
                nn.Conv2d(self.base_model_output_size, maps, 1),  # 1x1 convolution to reduce the number of channels to `maps`
                nn.BatchNorm2d(maps),                             # Batch normalization
                nn.ReLU(inplace=True)                             # ReLU activation
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        encoder_layer = TransformerEncoderLayer(
                  maps, 
                  nhead, 
                  dim_feedforward, 
                  dropout)

        encoder_norm = nn.LayerNorm(maps) 

        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

        self.pos_embedding = nn.Embedding(dim_feature+1, maps)

        self.feed = nn.Linear(maps, 2)
            
        self.loss_op = nn.L1Loss()

        # Check if transformer_weights_path is provided
        if config.train.pretrain.enable:
            self.load_transformer_weights(config.train.pretrain.path)

        # if hasattr(config.train, 'freeze_transformer') and config.train.freeze_transformer:
        #     self.freeze_transformer_layers()

        # Add a convolutional layer to match the output size of the backbone to the input size of the transformer encoder
        if backbone != 'resnet18':
            self.conv = nn.Conv2d(self.base_model_output_size, maps, kernel_size=1)

        # Freeze early layers of the backbone
        # self.freeze_early_layers()

        # print the number of parameters
        print(f"Total parameters: {sum(p.numel() for p in self.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def load_transformer_weights(self, transformer_weights_path):
        if isinstance(transformer_weights_path, dict):
            state_dict = transformer_weights_path
        else:
            state_dict = torch.load(transformer_weights_path, map_location='cpu')
        
        if not isinstance(state_dict, dict):
            raise ValueError(f"Loaded state_dict is not a dictionary. Got type: {type(state_dict)}")

        # Filter the state_dict to include only transformer weights
        transformer_state_dict = {k: v for k, v in state_dict.items() if 'encoder' in k or 'cls_token' in k or 'pos_embedding' in k}

        self.encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in transformer_state_dict.items() if 'encoder' in k})
        if 'cls_token' in transformer_state_dict:
            self.cls_token.data.copy_(transformer_state_dict['cls_token'])
        if 'pos_embedding' in transformer_state_dict:
            self.pos_embedding.load_state_dict({k.replace('pos_embedding.', ''): v for k, v in transformer_state_dict.items() if 'pos_embedding' in k})

    def load_backbone_weights(self, backbone_weights_path):
        if isinstance(backbone_weights_path, dict):
            state_dict = backbone_weights_path
        else:
            state_dict = torch.load(backbone_weights_path, map_location='cpu')
        
        if not isinstance(state_dict, dict):
            raise ValueError(f"Loaded state_dict is not a dictionary. Got type: {type(state_dict)}")

        # Filter the state_dict to include only backbone weights
        backbone_state_dict = {k: v for k, v in state_dict.items() if 'base_model' in k}

        self.base_model.load_state_dict({k.replace('base_model.', ''): v for k, v in backbone_state_dict.items()})

    def forward(self, x_in):
        feature = self.base_model(x_in["face"])
        if hasattr(self, 'conv'):
            feature = feature.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
            feature = self.conv(feature)  # Match the output size to the input size of the transformer encoder
        else:
            feature = feature.view(feature.size(0), feature.size(1), 7, 7)  # Ensure feature has the correct shape for resnet18
        batch_size = feature.size(0)
        feature = feature.view(batch_size, feature.size(1), -1)  # Ensure feature has the correct shape
        feature = feature.permute(2, 0, 1)
        
        cls = self.cls_token.expand(1, batch_size, -1)  # Ensure cls_token has the same batch size
        feature = torch.cat([cls, feature], 0)
        
        position = torch.arange(0, feature.size(0), device=feature.device).unsqueeze(1).repeat(1, batch_size)

        pos_feature = self.pos_embedding(position)

        # feature is [HW, batch, channel]
        feature = self.encoder(feature, pos_feature)
  
        feature = feature.permute(1, 2, 0)

        feature = feature[:,:,0]

        gaze = self.feed(feature)
        
        return gaze


    def loss(self, x_in, label):
        # Unpack the inputs
        gaze = self.forward(x_in)
        loss = self.loss_op(gaze, label) 
        return loss
