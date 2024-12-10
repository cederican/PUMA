import torch
import torch.nn as nn
from PIL import Image
from ctran import ctranspath
from src.dataset.HistoDataset import HistoDataset
from torch.utils.data import DataLoader



train_dataset = HistoDataset(
        image_dir="data/01_training_dataset_tif_ROIs",
        geojson_dir_tissue="data/01_training_dataset_geojson_tissue",
        geojson_dir_nuclei="data/01_training_dataset_geojson_nuclei",
        transform=True,
        color_norm=None,
    )

train_data_loader = DataLoader(
        train_dataset, 
        batch_size=1,      # Define batch size
        shuffle=False,      # Shuffle data during loading
        num_workers=0,     # Number of worker threads for loading data
        pin_memory=True    # Optimize for GPU if available
    )
    
model = ctranspath()
model.head = nn.Identity()
td = torch.load("/home/cederic/dev/puma/models/ctranspath.pth")
model.load_state_dict(td['model'], strict=True)


model.eval()
with torch.no_grad():
    for batch_idx, (image, tissue_seg, nuclei_seg, label) in enumerate(train_data_loader):
        features = model(image)
        features = features.cpu().numpy()