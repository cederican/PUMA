import torch as th
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from sklearn.metrics import roc_auc_score

class SegmentationMetric(Metric):
    """
    Custom metric for calculating loss, error, accuracy, precision, recall, F1 score, and AUC.

    Attributes:
        model (nn.Module): The model being evaluated.
        device (str): The device on which computations are performed.
    """
    def __init__(self, model, mode, num_classes, device):
        self.model = model
        self.mode = mode
        self.device = device
        self.num_classes = num_classes
        self._loss_sum = 0
        self._error_sum = 0
        self._accuracy_sum = 0
        self._true_positives = 0
        self._false_positives = 0
        self._false_negatives = 0
        self._dice_sum = 0
        self._iou_sum = 0
        self._num_examples = 0
        super().__init__(device=device)

    def reset(self):
        """
        Resets all metric counters to zero.
        """
        self._loss_sum = 0
        self._error_sum = 0
        self._accuracy_sum = 0
        self._true_positives = 0
        self._false_positives = 0
        self._false_negatives = 0
        self._dice_sum = 0
        self._iou_sum = 0
        self._num_examples = 0
        super().reset()

    def update(self, batch):
        """
        Updates the metric counters based on the current batch.

        Args:
            batch (tuple): A tuple containing the input data and labels.
        """
        image, tissue_seg, nuclei_seg, label = batch
        image, tissue_seg, nuclei_seg = image.to(self.device), tissue_seg.to(self.device), nuclei_seg.to(self.device)
        
        logits, probs, predicted_classes, features = self.model(image)
        tissue_seg = tissue_seg.to(logits.device)
        
        tissue_seg_one_hot = th.nn.functional.one_hot(tissue_seg, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Loss (Dice Loss)
        intersection = th.sum(probs * tissue_seg_one_hot, dim=(2, 3))
        union = th.sum(probs + tissue_seg_one_hot, dim=(2, 3))
        dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
        macro_dice = th.mean(dice_score, dim=1)
        dice_loss = 1.0 - macro_dice.mean()

        # Pixel Accuracy
        correct_pixels = (predicted_classes == tissue_seg).sum().item()
        total_pixels = predicted_classes.numel()
        pixel_accuracy = correct_pixels / total_pixels

        # IoU and mIoU
        iou_per_class = (intersection + 1e-6) / (union - intersection + 1e-6)
        mean_iou = th.mean(iou_per_class).item()

        # Update cumulative metrics
        self._loss_sum += dice_loss.item()
        self._accuracy_sum += pixel_accuracy
        self._dice_sum += macro_dice.mean().item()
        self._iou_sum += mean_iou
        self._num_examples += 1

    def compute(self):
        """
        Computes the final metric values.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        if self._num_examples == 0:
            raise NotComputableError("SegmentationMetrics must have at least one example before it can be computed.")
        
        avg_loss = self._loss_sum / self._num_examples
        avg_accuracy = self._accuracy_sum / self._num_examples
        avg_dice = self._dice_sum / self._num_examples
        avg_iou = self._iou_sum / self._num_examples
        
        if self.mode == "val":
            return {
                "val/loss": avg_loss,
                "val/accuracy": avg_accuracy,
                "val/dice": avg_dice,
                "val/iou": avg_iou,
            }
        elif self.mode == "test":
            return {
                "test/loss": avg_loss,
                "test/accuracy": avg_accuracy,
                "test/dice": avg_dice,
                "test/iou": avg_iou,
            }
