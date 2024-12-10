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
    ):
        super().__init__() 
        self.model = self._load_model(model_path)
        #self.model.eval()
        
    def _load_model(self, model_path):
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(model_path)
        model.load_state_dict(td['model'], strict=True)
        return model
    
    def forward(self, x):
        features = self.model(x) # shape (num_instances, 768)
        return features

class SegmentationModel(nn.Module):
    def __init__(self, config):
        super(SegmentationModel, self).__init__()
        
        self.config = config
        
        self.encoder = ConvFeatureExtractor(
            model_path=config.feature_extractor_path,
        )  
        self.num_classes = config.num_classes
        
        self.dec1 = nn.Sequential(
            nn.Linear(768, 7 * 7 * 768),               # Map (768) to (7x7x768)
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),  # Upsample to 14x14
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Upsample to 28x28
            nn.ReLU(inplace=True),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Upsample to 56x56
            nn.ReLU(inplace=True),
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # Upsample to 112x112
            nn.ReLU(inplace=True),
        )
        self.segmentation_head = torch.softmax(nn.ConvTranspose2d(64, self.num_classes, kernel_size=4, stride=2, padding=1), dim=1)  # Final upsample to 224x224
        
        self.encoder = self.encoder.to(config.device)
        self.dec1 = self.dec1.to(config.device)
        self.dec2 = self.dec2.to(config.device)
        self.dec3 = self.dec3.to(config.device)
        self.dec4 = self.dec4.to(config.device)
        self.dec5 = self.dec5.to(config.device)
        self.segmentation_head = self.segmentation_head.to(config.device)
        
    def forward(self, x):
       
        features = self.encoder(x)  # Shape: (B, 768)
        batch_size = features.shape[0]
        
        u = self.dec1(features)  # Shape: (B, 7*7*768)
        u = u.view(batch_size, 768, 7, 7)  # Reshape to (B, 768, 7, 7)
        u = self.dec2(u)  # Shape: (B, 512, 14, 14)
        u = self.dec3(u)  # Shape: (B, 256, 28, 28)
        u = self.dec4(u)  # Shape: (B, 128, 56, 56)
        u = self.dec5(u)  # Shape: (B, 64, 112, 112)
        
        segmentation_map = self.segmentation_head(u)  # Shape: (B, num_classes, 224, 224)
        predicted_classes = torch.argmax(segmentation_map, dim=1)  # Shape: (B, 224, 224)
        
        return predicted_classes, features


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
        ),
        num_classes=6,
        feature_extractor_path="/home/cederic/dev/puma/models/ctranspath.pth",
    )
    
    train_data_loader = data_utils.DataLoader(
        HistoDataset(**train_config.dataset_config.__dict__),
        batch_size=train_config.batch_size,
        shuffle=False,
    )
    
    model = SegmentationModel(config=train_config)
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
            segmentation_map, features = model(image)
            print(f"Features shape: {features.shape}")
            print(f"Segmentation map shape: {segmentation_map.shape}")
            plot_images(image[0].permute(1,2,0).cpu().detach().numpy(), segmentation_map[0].cpu().detach().numpy(), nuclei_seg[0].cpu().detach().numpy(), batch_idx)

            if batch_idx == 2:
                break
