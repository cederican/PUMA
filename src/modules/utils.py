import torch as th
import os
from PIL import Image
import numpy as np

def move_to_device(nested_list, device):
    """
    Moves a nested list of tensors to the specified device.

    Args:
        nested_list (list): The nested list of tensors.
        device (torch.device): The device to which the tensors should be moved.
    """
    if isinstance(nested_list, th.Tensor):
        return nested_list.to(device)
    elif isinstance(nested_list, list):
        return [move_to_device(item, device) for item in nested_list]
    elif isinstance(nested_list, dict):
        return {key: move_to_device(value, device) for key, value in nested_list.items()}
    elif isinstance(nested_list, str):
        return nested_list
    else:
        raise TypeError("All elements must be either tensors or lists of tensors.")
    

def tissue_label_to_logits(
        label: str,
    ):
        class_to_index = {
                "tissue_white_background": 0,
                "tissue_stroma": 1,
                "tissue_blood_vessel": 2,
                "tissue_tumor": 3,
                "tissue_epidermis": 4,
                "tissue_necrosis": 5,
        }
        return class_to_index[label]
    
def tissue_logits_to_label(
        logits: int,
    ):
        index_to_class = {
                0: "tissue_white_background",
                1: "tissue_stroma",
                2: "tissue_blood_vessel",
                3: "tissue_tumor",
                4: "tissue_epidermis",
                5: "tissue_necrosis",
        }
        return index_to_class[logits]
    
def nuclei_label_to_logits(
        label: str,
        which_track: int,
    ):
        if which_track == 1:
            class_to_index = {
                "nuclei_tumor": 1,
                "nuclei_lymphocyte": 2,
                "nuclei_plasma_cell": 2,
                "nuclei_apoptosis": 3,
                "nuclei_stroma": 3,
                "nuclei_neutrophil": 3,
                "nuclei_endothelium": 3,
                "nuclei_histiocyte": 3,
                "nuclei_melanophage": 3,
                "nuclei_epithelium": 3
            }
        elif which_track == 2:
            class_to_index = {
                "nuclei_tumor": 1,
                "nuclei_lymphocyte": 2,
                "nuclei_plasma_cell": 3,
                "nuclei_apoptosis": 4,
                "nuclei_stroma": 5,
                "nuclei_neutrophil": 6,
                "nuclei_endothelium": 7,
                "nuclei_histiocyte": 8,
                "nuclei_melanophage": 9,
                "nuclei_epithelium": 10
            }
        return class_to_index[label]
    
def nuclei_logits_to_label(
        logits: int,
        which_track: int,
    ):
        if which_track == 1:
            index_to_class = {
                1: "nuclei_tumor",
                2: "nuclei_lymphocyte",
                2: "nuclei_plasma_cell",
                3: "nuclei_apoptosis",
                3: "nuclei_stroma",
                3: "nuclei_neutrophil",
                3: "nuclei_endothelium",
                3: "nuclei_histiocyte",
                3: "nuclei_melanophage",
                3: "nuclei_epithelium",
            }
        elif which_track == 2:
            index_to_class = {
                1: "nuclei_tumor",
                2: "nuclei_lymphocyte",
                3: "nuclei_plasma_cell",
                4: "nuclei_apoptosis",
                5: "nuclei_stroma",
                6: "nuclei_neutrophil",
                7: "nuclei_endothelium",
                8: "nuclei_histiocyte",
                9: "nuclei_melanophage",
                10: "nuclei_epithelium",
            }
        return index_to_class[logits]
        

def get_tumor_annotation(
        case_name: str,
):
    annotations_path = "/home/space/datasets/camelyon16/annotations"
    annotations_path = f"{annotations_path}/{case_name}.png"
    if not os.path.isfile(annotations_path):
        print(f"Image {case_name} not found in {annotations_path}")
        return None
    with Image.open(annotations_path) as img:
        rgba_array = np.array(img)
        r_channel = rgba_array[:, :, 0]
        g_channel = rgba_array[:, :, 1]
        b_channel = rgba_array[:, :, 2]

        red_mask = (r_channel == 255) & (g_channel == 0) & (b_channel == 0)
        positions = np.argwhere(red_mask)
        red_positions = {i: (x, y) for i, (y, x) in enumerate(positions)}
    return rgba_array 

def cut_off(
        y_instance_pred,
        vis_mode,
        top_k=10,
        threshold=0.9,
):
    if not isinstance(y_instance_pred, th.Tensor):
        y_instance_pred = th.tensor(y_instance_pred)

    if vis_mode == "raw":
        return y_instance_pred

    elif vis_mode == "log":
        y_instance_pred = np.clip(y_instance_pred, a_min=1e-10, a_max=None)
        y_instance_pred = th.tensor(y_instance_pred)

        return y_instance_pred
    
    elif vis_mode == "percentile":
        percentile_min = np.percentile(y_instance_pred, 1)
        percentile_max = np.percentile(y_instance_pred, 99)
        value_clipped = th.clamp(y_instance_pred, percentile_min, percentile_max)
        y_instance_pred = (value_clipped - percentile_min) / (percentile_max - percentile_min)

        return y_instance_pred




    
    # top_values, top_indices = th.topk(y_instance_pred, top_k)
    # cumulative_sum = th.sum(top_values)
    # if cumulative_sum*100 > threshold:
    #     modified_array = y_instance_pred.clone()
    #     modified_array[top_indices] = 0.0
    #     print(f"Attention Map cut off at {top_k} positions")
    # else:
    #     modified_array = y_instance_pred
    #     print(f"Attention Map not cut off at {top_k} positions")
    # return modified_array
    