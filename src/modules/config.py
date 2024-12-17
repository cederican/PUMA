from types import SimpleNamespace
from typing import Union

class DatasetConfig(SimpleNamespace):
    image_dir: str
    geojson_dir_tissue: str
    geojson_dir_nuclei: str
    transform: bool
    color_norm: str
    preprocess: bool

class SegmentationModelConfig(SimpleNamespace):
    device: str
    epochs: int
    batch_size: int
    split: list[float]
    dataset_config: DatasetConfig
    mode: str
    num_classes: int
    feature_extractor_path: str
    reset_encoder_parameters: bool
    ckpt_path: str
    lr: float
    dropout: float
    betas: tuple[float, float]
    weight_decay: float
    T_0: int
    T_mult: int
    eta_min: float
    step_size: int
    gamma: float
    ckpt_save_path: str
    misc_save_path: str
    val_every: int
    save_max: int

