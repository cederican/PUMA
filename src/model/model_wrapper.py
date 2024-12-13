import torch as th
import torch.utils.data as data_utils
import torch.nn.functional as F
from torchinfo import summary

from src.modules.ModelWrapperABC import ModelWrapper
from src.model.model import SegmentationModel

from src.dataset.HistoDataset import HistoDataset
from src.modules.config import DatasetConfig, SegmentationModelConfig
from src.modules.metrics import SegmentationMetric
from src.modules.plotting import plot_images, visualize_segmentation

class SegmentationModelWrapper(ModelWrapper):
    def __init__(
            self,
            *,
            model: SegmentationModel,
            config: dict,
            epochs: int,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.epochs = epochs

        if config.ckpt_path is not None:
            self.load_model_checkpoint(config.ckpt_path)
    
    def init_val_metrics(self):
        """
        Initializes the validation metrics.
        """
        self.val_metrics = SegmentationMetric(model=self.model,  mode="val", num_classes=self.config.num_classes, device=self.config.device)
        self.test_metrics = SegmentationMetric(model=self.model, mode="test", num_classes=self.config.num_classes, device=self.config.device)
    
    
    def configure_optimizers(self):
        """
        Configures the optimizer and the learning rate scheduler.

        Returns:
            tuple: The optimizer and the learning rate scheduler.
        """
        print(f"Using base learning rate: {self.config.lr}")
        optimizer = th.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay
        )
        # optional: add lr_scheduler
        # lr_scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=self.config.T_0,
        #     T_mult=self.config.T_mult,
        #     eta_min=self.config.eta_min
        # )

        # use lr_scheduler but constant lr
        lr_scheduler = th.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.step_size,
            gamma=self.config.gamma
        )
        return optimizer, lr_scheduler
    
    def training_step(
            self,
            model: SegmentationModel,
            batch: tuple,
    ):
        image, tissue_seg, nuclei_seg, label = batch
        image, tissue_seg, nuclei_seg = image.to(self.config.device), tissue_seg.to(self.config.device), nuclei_seg.to(self.config.device)
        
        loss, dice_loss, pixel_loss = self._loss(model, image, tissue_seg)
        loss.backward()

        return {"loss": loss, "dice_loss": dice_loss, "pixel_loss": pixel_loss}
    
    def validation_step(
            self,
            batch: tuple,
    ):
        #batch[0] = batch[0].squeeze(0)
        self.val_metrics.update(batch)
    
    def test_step(
            self,
            batch: tuple,
    ):
        #batch[0] = batch[0].squeeze(0)
        self.test_metrics.update(batch)

    def visualize_step(
            self,
            model: SegmentationModel,
            batch: tuple,
            misc_save_path: str,
            global_step: int,
            mode: str,
    ):
        if mode == "train":
            visualize_segmentation(model, batch, misc_save_path, global_step, mode)
            #visualize_histo_att(model, batch, misc_save_path, global_step, mode, "raw")
            #visualize_histo_gt(model, batch, misc_save_path)
            #visualize_histo_patches(model, batch, misc_save_path)
        #elif mode == "test":
            #visualize_histo_att(model, batch, misc_save_path, global_step, mode, "raw")
            #visualize_histo_att(model, batch, misc_save_path, global_step, mode, "log")
            #visualize_histo_att(model, batch, misc_save_path, global_step, mode, "percentile")
        pass
    
    def _loss(
            self,
            model: SegmentationModel,
            image: th.Tensor,
            tissue_seg: th.Tensor,
    ):
        logits, probs, predicted_classes, features = model(image)
        num_classes = logits.shape[1]
        tissue_seg_one_hot = F.one_hot(tissue_seg, num_classes=num_classes).permute(0, 3, 1, 2).float()  # Shape: (B, num_classes, H, W)

        intersection = th.sum(probs * tissue_seg_one_hot, dim=(2, 3))  # Shape: (B, num_classes)
        union = th.sum(probs + tissue_seg_one_hot, dim=(2, 3))  # Shape: (B, num_classes)
        
        dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)  # Shape: (B, num_classes)

        # Average Dice score across all classes (macro Dice)
        macro_dice = th.mean(dice_score, dim=1)  # Shape: (B,)

        dice_loss = 1.0 - macro_dice.mean()
        
        #pixel loss
        class_weights = [57.13917497, 0.62771381, 13.91278002, 0.24365293, 6.96976719, 14.31341772]  # based on data statistics          # 0: "tissue_white_background",1: "tissue_stroma",2: "tissue_blood_vessel",3: "tissue_tumor",4: "tissue_epidermis",5: "tissue_necrosis",
        #class_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        criterion = th.nn.CrossEntropyLoss(weight=th.tensor(class_weights).to(self.config.device))
        pixel_loss = criterion(logits, tissue_seg)
        
        loss = dice_loss
        

        return loss, dice_loss, pixel_loss

    
    def load_model_checkpoint(self, ckpt_path):
        try:
            model_state = th.load(ckpt_path) 
            self.model.load_state_dict(model_state["model"])
            print(f"Model loaded from {ckpt_path}")
        except Exception as e:
            print(f"Error loading model from {ckpt_path}: {e}")


if __name__ == "__main__":

    train_config = SegmentationModelConfig(
        device="cuda",
        epochs=2,
        batch_size=8,
        dataset_config=DatasetConfig(
            image_dir="data/01_training_dataset_tif_ROIs",
            geojson_dir_tissue="data/01_training_dataset_geojson_tissue",
            geojson_dir_nuclei="data/01_training_dataset_geojson_nuclei",
            transform=True,
            color_norm=None,    
        ),
        num_classes=6,
        feature_extractor_path="/home/cederic/dev/puma/models/ctranspath.pth",
        ckpt_path=None,
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-3,
        step_size=10000000,
        gamma=0.1,
    )

    train_data_loader = data_utils.DataLoader(
        HistoDataset(**train_config.dataset_config.__dict__),
        batch_size=train_config.batch_size,
        shuffle=False
    )

    model = SegmentationModel(config=train_config)
    wrapper = SegmentationModelWrapper(model=model, config=train_config, epochs=train_config.epochs)

    summary(model,
           verbose=1,
           input_data={"x": th.rand(1,3,224,224).to(train_config.device)},
    )

    optimizer, lr_scheduler = wrapper.configure_optimizers()
    wrapper.init_val_metrics()

    for batch_idx, batch in enumerate(train_data_loader):
        result = wrapper.training_step(
            model=model,
            batch=batch,
        )
        wrapper.validation_step(
            batch=batch,
        )
        print(f"Training Step {batch_idx} - Loss: {result['loss']}")
        if batch_idx >= 2:
            break