import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage.io import imread


def optical_density_conversion(image):
    """Convert RGB image to optical density (OD) space."""
    image = image.astype(np.float32) + 1  # Avoid division by zero
    od = -np.log(image / 255.0)
    return np.clip(od, a_min=1e-6, a_max=None)  # Avoid log(0)

def macenko_normalization(image, target_mean=[0.5, 0.5, 0.5], target_std=[0.2, 0.2, 0.2], alpha=1.0, beta=0.15):
    """
    Apply Macenko color normalization to a histopathology image.
    
    Args:
        image (np.array): RGB image (H, W, 3) in the range [0, 255].
        target_mean (list): Target mean optical density for each channel (default: [0.5, 0.5, 0.5]).
        target_std (list): Target standard deviation for each channel (default: [0.2, 0.2, 0.2]).
        alpha (float): Percentile for robust PCA (default: 1.0).
        beta (float): OD threshold for background pixels (default: 0.15).
        
    Returns:
        np.array: Color-normalized image (H, W, 3) in the range [0, 255].
    """
    # Step 1: Convert to OD space
    od = optical_density_conversion(image)
    h, w, c = image.shape

    # Step 2: Remove background pixels
    od_flat = od.reshape((-1, c))
    od_non_bg = od_flat[np.any(od_flat > beta, axis=1)]  # Pixels above threshold

    # Step 3: Perform PCA to find the stain matrix
    pca = PCA(n_components=2)
    stain_matrix = pca.fit(od_non_bg).components_.T  # Transpose to shape (3, 2)

    # Normalize the stain matrix columns
    stain_matrix /= np.linalg.norm(stain_matrix, axis=0)

    # Step 4: Project OD values onto stain matrix to get stain concentrations
    # Ensure stain_matrix has shape (3, 2) and od_flat.T has shape (3, num_pixels)
    stain_concentrations = np.linalg.lstsq(stain_matrix, od_flat.T, rcond=None)[0]

    # Step 5: Rescale stain concentrations
    max_concentration = np.percentile(stain_concentrations, 100 * alpha, axis=1, keepdims=True)
    stain_concentrations /= (max_concentration + 1e-6)

    # Step 6: Reconstruct the image using the target stain intensities
    target_stain_matrix = np.array(target_mean).reshape(3, -1) * np.array(target_std).reshape(3, -1)
    normalized_od = np.dot(target_stain_matrix, stain_concentrations).T
    normalized_od = np.clip(normalized_od, a_min=0, a_max=None)  # Clip to avoid negatives

    # Step 7: Convert back to RGB space
    normalized_image = np.exp(-normalized_od).reshape((h, w, c))
    normalized_image = np.clip(normalized_image * 255, 0, 255).astype(np.uint8)

    return normalized_image


if __name__ == "__main__":
    image_path = "/home/cederic/dev/puma/data/01_training_dataset_tif_ROIs/training_set_metastatic_roi_001.tif"
    image = imread(image_path)

    # Apply Macenko normalization
    normalized_image = macenko_normalization(image)

    # Visualize the original and normalized images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Macenko Normalized Image")
    plt.imshow(normalized_image)
    plt.axis("off")
    plt.show()

