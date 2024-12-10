import os
import tifffile
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
import random
import json
from shapely.geometry import shape
from skimage.draw import polygon
from src.modules.plotting import plot_images
from src.modules.utils import tissue_label_to_logits, tissue_logits_to_label, nuclei_label_to_logits, nuclei_logits_to_label
from torch.utils.data import DataLoader


class TransformImageAndMask:
    def __init__(self):
        self.resize = T.Resize(224)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, image, tissue_seg, nuclei_seg):
        if random.random() < 0.5:
            image = F.hflip(image)
            tissue_seg = F.hflip(tissue_seg)
            nuclei_seg = F.hflip(nuclei_seg)

        if random.random() < 0.5:
            image = F.vflip(image)
            tissue_seg = F.vflip(tissue_seg)
            nuclei_seg = F.vflip(nuclei_seg)

        angle = random.choice([0, 90, 180, 270])
        image = F.rotate(image, angle)
        tissue_seg = F.rotate(tissue_seg, angle)
        nuclei_seg = F.rotate(nuclei_seg, angle)

        image = self.resize(image)
        tissue_seg = self.resize(tissue_seg)
        nuclei_seg = self.resize(nuclei_seg)

        image = self.to_tensor(image)
        tissue_seg = torch.tensor(np.array(tissue_seg), dtype=torch.long)
        nuclei_seg = torch.tensor(np.array(nuclei_seg), dtype=torch.long)

        image = self.normalize(image)

        return image, tissue_seg, nuclei_seg


class HistoDataset(Dataset):
    def __init__(
        self,
        image_dir,
        geojson_dir_tissue,
        geojson_dir_nuclei,
        transform=None,
        color_norm="macenko"
    ):
        """
        Args:
            image_dir (str): Path to folder containing .tif images.
            geojson_dir_tissue (str): Path to folder containing tissue segmentation .geojson files.
            geojson_dir_nuclei (str): Path to folder containing nuclei segmentation .geojson files.
            transform (callable, optional): Transform to apply to images (e.g., augmentations).
            color_norm (str): Color normalization method ('macenko' or 'reinhard').
        """
        self.image_dir = image_dir
        self.geojson_dir_tissue = geojson_dir_tissue
        self.geojson_dir_nuclei = geojson_dir_nuclei
        if transform:
            self.transform = TransformImageAndMask()
        else:
            self.transform = transform
        self.color_norm = color_norm

        self.image_paths = sorted(
            [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(".tif")]
        )

        self.labels = [1 if "metastatic" in os.path.basename(path) else 0 for path in self.image_paths] # metastatic = 1, primary = 0

        self.tissue_paths = sorted(
            [os.path.join(geojson_dir_tissue, fname) for fname in os.listdir(geojson_dir_tissue) if fname.endswith(".geojson")]
        )
        self.nuclei_paths = sorted(
            [os.path.join(geojson_dir_nuclei, fname) for fname in os.listdir(geojson_dir_nuclei) if fname.endswith(".geojson")]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
       
        img_path = self.image_paths[idx]
        
        name = os.path.splitext(os.path.basename(img_path))[0]
        name = name.split("_", maxsplit=2)[-1]
        image = tifffile.imread(img_path)
        image_shape = image.shape
        image = Image.fromarray(image)
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # if self.color_norm == "macenko":
        #     image = macenko_normalization(image)
        # # elif self.color_norm == "reinhard":
        # #     image = reinhard_normalization(image)
        # elif self.color_norm is None:
        #     print("No color normalization applied.")

        tissue_seg = self._load_geojson(self.tissue_paths[idx], image_shape[:2], "tissue")
        tissue_seg = Image.fromarray(tissue_seg)
        nuclei_seg = self._load_geojson(self.nuclei_paths[idx], image_shape[:2], "nuclei")
        nuclei_seg = Image.fromarray(nuclei_seg)

        label = self.labels[idx]
        
        if self.transform:
            image, tissue_seg, nuclei_seg = self.transform(image, tissue_seg, nuclei_seg)
            
        # visualize
        plot_images(image.permute(1,2,0), tissue_seg, nuclei_seg, name)
        

        # ToDo - Add data augmentation here
        # patchify the image and the masks

        return image, tissue_seg, nuclei_seg, label

    def _load_geojson(
        self,
        geojson_path,
        image_shape,
        mode,
    ):
        mask = np.zeros(image_shape, dtype=np.uint8)

        with open(geojson_path, "r") as f:
            data = json.load(f)

        for feature in data["features"]:
            geom = shape(feature["geometry"])
            
            class_label = feature["properties"].get("classification", 0)  # Default to 0 if not specified
            if mode == "tissue":
                class_logit = tissue_label_to_logits(class_label['name'])
            elif mode == "nuclei":
                class_logit = nuclei_label_to_logits(class_label['name'], which_track=1)
            else:
                raise ValueError("Invalid mode. Choose 'tissue' or 'nuclei'.")
            
            if geom.geom_type == "Polygon":
                # Handle a single polygon
                exterior_coords = np.array(geom.exterior.coords)
                x, y = exterior_coords[:, 0], exterior_coords[:, 1]
                rr, cc = polygon(y, x, mask.shape)
                mask[rr, cc] = class_logit
            elif geom.geom_type == "MultiPolygon":
                # Handle multiple polygons
                for sub_geom in geom.geoms:
                    exterior_coords = np.array(sub_geom.exterior.coords)
                    x, y = exterior_coords[:, 0], exterior_coords[:, 1]
                    rr, cc = polygon(y, x, mask.shape)
                    mask[rr, cc] = class_logit

        return mask



        
    
if __name__ == "__main__":
    train_dataset = HistoDataset(
        image_dir="data/01_training_dataset_tif_ROIs",
        geojson_dir_tissue="data/01_training_dataset_geojson_tissue",
        geojson_dir_nuclei="data/01_training_dataset_geojson_nuclei",
        transform=True,
        color_norm=None,
    )
    val_dataset = HistoDataset(
        image_dir="/home/cederic/dev/puma/data/01_training_dataset_tif_ROIs",
        geojson_dir_tissue="/home/cederic/dev/puma/data/01_training_dataset_geojson_tissue",
        geojson_dir_nuclei="/home/cederic/dev/puma/data/01_training_dataset_geojson_nuclei",
        transform=True,
        color_norm=None,
    )

    #train_data = train_dataset[0]
    #val_data = val_dataset[0]

    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=1,      # Define batch size
        shuffle=False,      # Shuffle data during loading
        num_workers=0,     # Number of worker threads for loading data
        pin_memory=True    # Optimize for GPU if available
    )

    val_data_loader = DataLoader(
        val_dataset, 
        batch_size=1,      # Define batch size
        shuffle=False,      # Shuffle data during loading
        num_workers=0,     # Number of worker threads for loading data
        pin_memory=True    # Optimize for GPU if available
    )

    for batch_idx, (image, tissue_seg, nuclei_seg, label) in enumerate(train_data_loader):
        print(f"Batch {batch_idx}:")
        print(f"Image shape: {image.shape}")
        print(f"Tissue segmentation shape: {tissue_seg.shape}")
        print(f"Nuclei segmentation shape: {nuclei_seg.shape}")
        print(f"Label: {label}")
    
        if batch_idx == 1:
            break
    
    for batch_idx, (image, tissue_seg, nuclei_seg, label) in enumerate(val_data_loader):
        print(f"Batch {batch_idx}:")
        print(f"Image shape: {image.shape}")
        print(f"Tissue segmentation shape: {tissue_seg.shape}")
        print(f"Nuclei segmentation shape: {nuclei_seg.shape}")
        print(f"Label: {label}")
    
        if batch_idx == 1:
            break