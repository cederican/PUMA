import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.modules.utils import get_tumor_annotation
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches


nuclei_colors_track1 = {
    0: ('nuclei_white_background', '#f0f0f0'),
    1: ('nuclei_tumor', '#fbb4ae'),
    2: ('nuclei_TILs', '#b3cde3'),
    3: ('nuclei_other', '#ccebc5')
}

nuclei_colors_track2 = {
    0: ('nuclei_white_background', '#f0f0f0'),
    1: ('nuclei_tumor', '#fbb4ae'),
    2: ('nuclei_lymphocyte', '#b3cde3'),
    3: ('nuclei_plasma_cell', '#decbe4'),
    4: ('nuclei_apoptosis', '#fed9a6'),
    5: ('nuclei_stroma', '#ffffcc'),
    6: ('nuclei_endothelium', '#e5d8bd'),
    7: ('nuclei_histiocyte', '#fddaec'),
    8: ('nuclei_melanophage', '#f2f2f2'),
    9: ('nuclei_neutrophil', '#ccebc5'),
    10: ('nuclei_epithelium', '#ffed6f')
}

tissue_colors = {
    0: ('tissue_white_background', '#f0f0f0'),
    1: ('tissue_stroma', '#b3cde3'),
    2: ('tissue_blood_vessel', '#ccebc5'),
    3: ('tissue_tumor', '#fbb4ae'),
    4: ('tissue_epidermis', '#decbe4'),
    5: ('tissue_necrosis', '#fed9a6')
}

def create_colored_mask(mask, color_dict):
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, (name, color) in color_dict.items():
        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        colored_mask[mask == label] = rgb
    return colored_mask


def plot_images(
    image,
    tissue_segmentation,
    nuclei_segmentation,
    name,
):
    nuclei_colored = create_colored_mask(nuclei_segmentation, nuclei_colors_track1)
    tissue_colored = create_colored_mask(tissue_segmentation, tissue_colors)

    fig, ax = plt.subplots(1, 3, figsize=(16, 7))

    # 1. Original Image
    ax[0].imshow(image)
    ax[0].set_title("Melanoma Region of Interest")
    ax[0].axis("off")

    # Plot tissue segmentation
    ax[1].imshow(nuclei_colored)
    ax[1].set_title("Ground Truth (Nuclei)")
    ax[1].axis("off")

    # 3. Tissue Segmentation Mask
    ax[2].imshow(tissue_colored)
    ax[2].set_title("Ground Truth (Tissue)")
    ax[2].axis("off")
    
    nuclei_legend = [mpatches.Patch(color=color, label=name) for label, (name, color) in nuclei_colors_track1.items()]
    tissue_legend = [mpatches.Patch(color=color, label=name) for label, (name, color) in tissue_colors.items()]

    fig.legend(handles=nuclei_legend, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize="small")
    fig.legend(handles=tissue_legend, loc="lower center", bbox_to_anchor=(0.83, 0.05), ncol=3, fontsize="small")

    plt.tight_layout()
    plt.show()

    plt.savefig(f"logs/misc/visualize_data_{name}.png")



# def visualize_histo_patches(
#         model,
#         batch: tuple,
#         misc_save_path: str
# ):
#     features, label, cls, dict = batch
#     positions = [dict[i][1] for i in range(len(dict)-2)]
#     patch_size_abs = [dict[i][2] for i in range(len(dict)-2)]
#     original_shape = dict['original_shape']

#     figsize = ((original_shape[0].item() / 100), (original_shape[1].item() / 100))

#     fig, ax = plt.subplots(figsize=figsize)
#     fig.patch.set_facecolor("black")
#     ax.set_xlim(0, original_shape[0].item())
#     ax.set_ylim(0, original_shape[1].item())
#     ax.invert_yaxis()

#     for i, position in enumerate(positions):
#         patch_path = os.path.join("/home/space/datasets/camelyon16/patches/20x", dict['case_name'][0], f"{dict[i][0].item()}.jpg")
#         if os.path.exists(patch_path):
#             patch = Image.open(patch_path)
#             patch_array = np.asarray(patch)
#             patch_array = np.flipud(patch_array) 
#             x, y = position
#             x = x.item()/16
#             y = y.item()/16

#             patch_width = patch_size_abs[i].item()/16
#             patch_height = patch_size_abs[i].item()/16
#             ax.imshow(patch_array, extent=(x, (x + patch_width), y, (y + patch_height)))

#             #value = y_instance_pred[i]
#             #value = (value - min_ak) / (max_ak - min_ak)
#             #color = plt.cm.Reds(value)
#             #rect = patches.Rectangle((x.item()/16, y.item()/16), patch_size_abs[i].item()/16, patch_size_abs[i].item()/16, linewidth=0, edgecolor=None, facecolor=color)
#             #ax.add_patch(rect)
            
    
#     ax.axis('off')
#     if misc_save_path:
#         batch_name = dict['case_name'][0]
#         batch_dir = os.path.join(misc_save_path, batch_name)
#         if not os.path.exists(batch_dir):
#             os.makedirs(batch_dir)
#         image_save_path = f"{batch_dir}/all_cell_patches.png"
#         if not os.path.exists(image_save_path):
#             plt.savefig(image_save_path, dpi=100, bbox_inches='tight', pad_inches=0)
#             img = Image.open(image_save_path)
#             img = img.resize((original_shape[0].item(), original_shape[1].item()), Image.ANTIALIAS)
#             img.save(image_save_path)
#     plt.close()


# def visualize_histo_gt(
#         model,
#         batch: tuple,
#         misc_save_path: str
# ):
#     features, label, cls, dict = batch
#     positions = [dict[i][1] for i in range(len(dict)-2)]
#     patch_size_abs = [dict[i][2] for i in range(len(dict)-2)]
#     original_shape = dict['original_shape']
#     annotation_array = get_tumor_annotation(dict['case_name'][0])

#     figsize = ((original_shape[0].item() / 100), (original_shape[1].item() / 100))

#     fig, ax = plt.subplots(figsize=figsize)
#     fig.patch.set_facecolor("black")
#     ax.set_xlim(0, original_shape[0].item())
#     ax.set_ylim(0, original_shape[1].item())
#     ax.invert_yaxis()

#     for i, position in enumerate(positions):
#         patch_path = os.path.join("/home/space/datasets/camelyon16/patches/20x", dict['case_name'][0], f"{dict[i][0].item()}.jpg")
#         if os.path.exists(patch_path):
#             patch = Image.open(patch_path)
#             patch_array = np.asarray(patch)
#             patch_array = np.flipud(patch_array) 
#             x, y = position
#             x = x.item()/16
#             y = y.item()/16

#             patch_width = patch_size_abs[i].item()/16
#             patch_height = patch_size_abs[i].item()/16

#             annotation_region = annotation_array[
#                 int(y):int(y + patch_height),
#                 int(x):int(x + patch_width),
#                 :
#             ]
#             is_red = np.any(
#                 (annotation_region[:, :, 0] == 255) & 
#                 (annotation_region[:, :, 1] == 0) & 
#                 (annotation_region[:, :, 2] == 0)   
#             )
#             if is_red:
#                 ax.imshow(patch_array, extent=(x, (x + patch_width), y, (y + patch_height)))


#     ax.axis('off')
#     if misc_save_path:
#         batch_name = dict['case_name'][0]
#         batch_dir = os.path.join(misc_save_path, batch_name)
#         if not os.path.exists(batch_dir):
#             os.makedirs(batch_dir)
#         image_save_path = f"{batch_dir}/gt_tumor_patches.png"
#         if not os.path.exists(image_save_path):
#             plt.savefig(image_save_path, dpi=100, bbox_inches='tight', pad_inches=0)
#             img = Image.open(image_save_path)
#             img = img.resize((original_shape[0].item(), original_shape[1].item()), Image.ANTIALIAS)
#             img.save(image_save_path)
#     plt.close()
    
    


if __name__ == "__main__":
    print("Visualizing AUC results...")
    #visualize_auc_results(10, 2, "./logs", False, True)
    #visualize_auc_results(50, 10, "./logs", False, True)
    #visualize_auc_results(100, 20, "./logs", False, True)