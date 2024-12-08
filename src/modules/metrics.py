import torch as th
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from sklearn.metrics import roc_auc_score

class LossErrorAccuracyPrecisionRecallF1Metric(Metric):
    """
    Custom metric for calculating loss, error, accuracy, precision, recall, F1 score, and AUC.

    Attributes:
        model (nn.Module): The model being evaluated.
        device (str): The device on which computations are performed.
    """
    def __init__(self, model, just_features, mode, device="cpu"):
        self.model = model
        self.just_features = just_features
        self.mode = mode
        self._loss_sum = 0
        self._error_sum = 0
        self._accuracy_sum = 0
        self._true_positives = 0
        self._false_positives = 0
        self._false_negatives = 0
        self._num_examples = 0
        self._y_true = [] 
        self._y_scores = []
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
        self._num_examples = 0
        self._y_true = []
        self._y_scores = []
        super().reset()

    def update(self, batch):
        """
        Updates the metric counters based on the current batch.

        Args:
            batch (tuple): A tuple containing the input data and labels.
        """
        if self.just_features == False:
            bag, label = batch[0], batch[1]
            y_bag_true = label[0].float()
        else:
            bag, label, cls, dict = batch
            y_bag_true = label
        y_bag_pred, _ = self.model(bag)
  
        y_bag_pred = th.clamp(y_bag_pred, min=1e-4, max=1.0 - 1e-4)
        
        loss = th.nn.BCELoss()(y_bag_pred, y_bag_true)
        
        y_bag_pred_binary = th.where(y_bag_pred > 0.5, 1, 0)

        accuracy = th.mean((y_bag_pred_binary == y_bag_true).float()).item()
        error = 1.0 - accuracy

        # Precision, Recall, F1
        self._true_positives += th.sum((y_bag_pred_binary == 1) & (y_bag_true == 1)).item()
        self._false_positives += th.sum((y_bag_pred_binary == 1) & (y_bag_true == 0)).item()
        self._false_negatives += th.sum((y_bag_pred_binary == 0) & (y_bag_true == 1)).item()

        self._loss_sum += loss.item()
        self._error_sum += error
        self._accuracy_sum += accuracy
        self._num_examples += 1

        self._y_true.extend(y_bag_true.cpu().numpy())
        self._y_scores.extend(y_bag_pred.detach().cpu().numpy())

    def compute(self):
        """
        Computes the final metric values.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        if self._num_examples == 0:
            raise NotComputableError("LossErrorAccuracyPrecisionRecallF1Metric must have at least one example before it can be computed.")
        
        avg_loss = self._loss_sum / self._num_examples
        avg_error = self._error_sum / self._num_examples
        avg_accuracy = self._accuracy_sum / self._num_examples
        
        precision = self._true_positives / (self._true_positives + self._false_positives) if (self._true_positives + self._false_positives) > 0 else 0.0
        recall = self._true_positives / (self._true_positives + self._false_negatives) if (self._true_positives + self._false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        try:
            auc = roc_auc_score(self._y_true, self._y_scores)
        except ValueError:
            auc = float('nan')

        if self.mode == "val":
            return {
                "val/loss": avg_loss,
                "val/error": avg_error,
                "val/accuracy": avg_accuracy,
                "val/precision": precision,
                "val/recall": recall,
                "val/f1": f1_score,
                "val/auc": auc
            }
        elif self.mode == "test":
            return {
                "test/loss": avg_loss,
                "test/error": avg_error,
                "test/accuracy": avg_accuracy,
                "test/precision": precision,
                "test/recall": recall,
                "test/f1": f1_score,
                "test/auc": auc
            }
