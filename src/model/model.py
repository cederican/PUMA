import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torchinfo import summary
from ctran import ctranspath

from src.dataset.HistoDataset import HistoDataset
from src.modules.config import DatasetConfig, SegmentationModelConfig
from src.modules.plotting import plot_images



class ConvFeatureExtractor(nn.Module):
    def __init__(
            self,
            model_path: str,
            reset_encoder_parameters: bool,
    ):
        super().__init__()
        self.reset_encoder_parameters = reset_encoder_parameters 
        self.model = self._load_model(model_path)
        self.stages = [0,1,2,3]
        
    def _load_model(self, model_path):
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(model_path, weights_only=True)
        model.load_state_dict(td['model'], strict=True)
        if self.reset_encoder_parameters:
            self.reset_parameters(model)
        model.train()
        return model
    
    def reset_parameters(self, model):
        # Iterate through all layers in the model
        for name1, module in model.named_children():
            if name1 == "patch_embed":
                for name2, layer in module.named_children():
                    if name2 == "norm":
                        print(f"Resetting parameters for layer: {layer}")
                        layer.reset_parameters()
                    elif name2 == "proj":
                        print(f"Resetting parameters for layer: {layer}")
                        for i in range(len(layer)):
                            if hasattr(layer[i], "reset_parameters"):
                                layer[i].reset_parameters()
                                print(f"Resetting parameters for layer: {layer[i]}")
            elif name1 == "layers":
                for name2, layer in module.named_children():
                    for name3, sublayer in layer.named_children():
                        for name4, subsublayer in sublayer.named_children():
                            if name4 == "reduction" or name4 == "norm":
                                print(f"Resetting parameters for layer: {subsublayer}")
                                subsublayer.reset_parameters()
                            elif name4 == "0" or name4 == "1" or name4 == "2" or name4 == "3" or name4 == "4" or name4 == "5":
                                print(f"Resetting parameters for layer: {subsublayer}")
                                for name5, subsubsublayer in subsublayer.named_children():
                                    if name5 == "norm1" or name5 == "norm2":
                                        print(f"Resetting parameters for layer: {subsubsublayer}")
                                        subsubsublayer.reset_parameters()
                                    elif name5 == "attn":
                                        for name6, subsubsubsublayer in subsubsublayer.named_children():
                                            if name6 == "qkv":
                                                print(f"Resetting parameters for layer: {subsubsubsublayer}")
                                                subsubsubsublayer.reset_parameters()
                                            elif name6 == "proj":
                                                print(f"Resetting parameters for layer: {subsubsubsublayer}")
                                                subsubsubsublayer.reset_parameters()
                                    elif name5 == "mlp":
                                        for name6, subsubsubsublayer in subsubsublayer.named_children():
                                            if name6 == "fc1":
                                                print(f"Resetting parameters for layer: {subsubsubsublayer}")
                                                subsubsubsublayer.reset_parameters()
                                            elif name6 == "fc2":
                                                print(f"Resetting parameters for layer: {subsubsubsublayer}")
                                                subsubsubsublayer.reset_parameters()
            elif name1 == "norm":
                print(f"Resetting parameters for layer: {module}")
                module.reset_parameters()
                
        print("All parameters reset!")
        
                                                   
    
    def forward(self, x):
        #features = self.model(x) # shape (num_instances, 768)
        features = []
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        for i, layer in enumerate(self.model.layers):
            x = layer(x)
            if i in self.stages:
                features.append(x)
        x = self.model.norm(x)
        #x = x.mean(dim=1)
        x = self.model.head(x)
        features.append(x)
        return features

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc1(y)
        y = nn.ReLU(inplace=True)(y)
        y = self.fc2(y)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y

class SegmentationModel(nn.Module):
    def __init__(self, config):
        super(SegmentationModel, self).__init__()
        
        self.config = config
        self.mode = config.mode
        
        self.encoder = ConvFeatureExtractor(
            model_path=config.feature_extractor_path,
            reset_encoder_parameters=config.reset_encoder_parameters,
        ) 
        self.dec1 = self._decoder_block(768, 768, 768)   # From (49, 768) to (7x7, 512)
        self.dec2 = self._decoder_block(768, 768, 384)  # From (196, 384) to (14x14, 256)
        self.dec3 = self._decoder_block(384, 384, 192)  # From (192, 768) to (28x28, 128)
        self.dec4 = self._decoder_block(192, 192, 64)         # From (128, 64) to (56x56, 64)

        if self.mode == "tissue":
            self.num_classes = 6
            self.segmentation_head = nn.ConvTranspose2d(64, self.num_classes, kernel_size=4, stride=2, padding=1)  # Final upsample to 224x224
        elif self.mode == "nuclei1":
            self.num_classes = 4
            self.segmentation_head = nn.ConvTranspose2d(64, self.num_classes, kernel_size=4, stride=2, padding=1)
        elif self.mode == "nuclei2":
            self.num_classes = 11
            self.segmentation_head = nn.ConvTranspose2d(64, self.num_classes, kernel_size=4, stride=2, padding=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Upsampling by a factor of 2

    def _decoder_block(self, in_channels, skip_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels + skip_channels, out_channels, kernel_size=4, stride=2, padding=1),  # Upsample
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Fusion with skip
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
        )
        
        
    def forward(self, x):
       
        features = self.encoder(x)  # Shape: list of features at different stages [(B, 768, 192), (B, 196, 384), (B, 49, 768), (B, 49, 768), (B, 49, 768)]
        f0, f1, f2, f3, f4 = features
        
        x = f4.view(f4.size(0), 768, 7, 7)
        
        for i, dec_block in enumerate([self.dec1, self.dec2, self.dec3, self.dec4]):
    
            
            if i == 0:
                skip = f3.view(f3.size(0), 768, 7, 7) 
            elif i == 1:
                skip = f2.view(f2.size(0), 768, 7, 7)  
            elif i == 2:
                skip = f1.view(f1.size(0), 384, 14, 14)
            else:
                skip = f0.view(f0.size(0), 192, 28, 28) 
                
            if i > 0:
                skip = self.upsample(skip)
                
            x = torch.cat([x, skip], dim=1)  
            
            x = dec_block[0](x)  
            x = dec_block[1](x)  
            x = dec_block[2](x)  

            x = dec_block[3](x)  
            x = dec_block[4](x)  
            x = dec_block[5](x)  
        
        logits = self.segmentation_head(x)
        
        
        # u = self.dec1(features)  # Shape: (B, 7*7*768)
        # u = u.view(batch_size, 768, 7, 7)  # Reshape to (B, 768, 7, 7)
        # u = self.dec2(u)  # Shape: (B, 512, 14, 14)
        # u = self.dec3(u)  # Shape: (B, 256, 28, 28)
        # u = self.dec4(u)  # Shape: (B, 128, 56, 56)
        # u = self.dec5(u)  # Shape: (B, 64, 112, 112)
        
        #logits = self.segmentation_head(u)  # Shape: (B, num_classes, 224, 224)
        
        probs = torch.softmax(logits, dim=1)  # Shape: (B, num_classes, 224, 224)
        predicted_classes = torch.argmax(probs, dim=1)  # Shape: (B, 224, 224)
        
        return logits, probs, predicted_classes, features


# test functionality of the different model approaches
if __name__ == "__main__":
    """
    test just one model approach
    """
    train_config = SegmentationModelConfig(
        device="cuda",
        batch_size=1,
        dataset_config=DatasetConfig(
            image_dir="data/01_training_dataset_tif_ROIs",
            geojson_dir_tissue="data/01_training_dataset_geojson_tissue",
            geojson_dir_nuclei="data/01_training_dataset_geojson_nuclei",
            transform=True,
            color_norm=None,
            preprocess=False,   
        ),
        num_classes=6,
        feature_extractor_path="/home/cederic/dev/puma/models/ctranspath.pth",
        reset_encoder_parameters=True,
        dropout=0.5,
    )
    
    train_data_loader = data_utils.DataLoader(
        HistoDataset(**train_config.dataset_config.__dict__),
        batch_size=train_config.batch_size,
        shuffle=False,
    )
    
    model = SegmentationModel(config=train_config).to(train_config.device)
    summary(model,
           verbose=1,
           input_data={"x": torch.rand(1,3,224,224).to(train_config.device)},
    )
    model.train()
    for batch_idx, (image, tissue_seg, nuclei_seg, label) in enumerate(train_data_loader):
            #print(f"Batch {batch_idx}:")
            #print(f"Features shape: {features.shape}")  
            #print(f"Labels: {label}")                 
            #print(f"Classes: {cls}")
            #print(f"Dict: {dict}")
            image, tissue_seg, nuclei_seg = image.to(train_config.device), tissue_seg.to(train_config.device), nuclei_seg.to(train_config.device)
            logits, probs, predicted_classes, features = model(image)
            print(f"Features shape: {features[4].shape}")
            print(f"Segmentation map shape: {predicted_classes.shape}")
            #plot_images(image[0].permute(1,2,0).cpu().detach().numpy(), predicted_classes[0].cpu().detach().numpy(), nuclei_seg[0].cpu().detach().numpy(), batch_idx)

            if batch_idx == 2:
                break
