import os
import torch
from tqdm import tqdm
from src.dataset.HistoDataset import HistoDataset

def preprocess_data():
    # Define directories to save preprocessed data
    output_dir_images = "/home/cederic/dev/puma/data/preprocessed_data/images_nn"
    output_dir_tissue = "/home/cederic/dev/puma/data/preprocessed_data/tissue_seg"
    output_dir_nuclei = "/home/cederic/dev/puma/data/preprocessed_data/nuclei_seg"

    # Create directories if they don't exist
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_tissue, exist_ok=True)
    os.makedirs(output_dir_nuclei, exist_ok=True)


    dataset = HistoDataset(
        image_dir="data/01_training_dataset_tif_ROIs",
        geojson_dir_tissue="data/01_training_dataset_geojson_tissue",
        geojson_dir_nuclei="data/01_training_dataset_geojson_nuclei",
        transform=True,  # Enable TransformImageAndMask
        color_norm=None,
        preprocess=True
    )

    # Iterate through the dataset and save preprocessed data
    print("Processing and saving data...")
    for idx in tqdm(range(len(dataset)), desc="Processing dataset"):
        image, tissue_seg, nuclei_seg, label = dataset[idx]

        # Save tensors to respective directories
        torch.save(image, os.path.join(output_dir_images, f"{idx:05d}.pt"))
        torch.save(tissue_seg, os.path.join(output_dir_tissue, f"{idx:05d}.pt"))
        torch.save(nuclei_seg, os.path.join(output_dir_nuclei, f"{idx:05d}.pt"))

    print("Preprocessing and saving complete!")

    # test to load the preprocessed data, get the first 5 samples 
    print("Loading preprocessed data...")
    for idx in range(5):
        image = torch.load(os.path.join(output_dir_images, f"{idx:05d}.pt"))
        tissue_seg = torch.load(os.path.join(output_dir_tissue, f"{idx:05d}.pt"))
        nuclei_seg = torch.load(os.path.join(output_dir_nuclei, f"{idx:05d}.pt"))

        print(f"Image shape: {image.shape}")
        print(f"Tissue segmentation shape: {tissue_seg.shape}")
        print(f"Nuclei segmentation shape: {nuclei_seg.shape}")

if __name__ == "__main__":
    
    preprocess_data()

