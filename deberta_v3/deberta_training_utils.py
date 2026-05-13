import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, "..", "utils"))
from nutrient_utils import evaluate_accuracy

TARGETS = ['fat_g', 'protein_g', 'saturates_g', 'sugars_g']


class NutrientMetrics:
    """compute_metrics callable for HF Trainer.

    scaler: dict with 'min' and 'range' arrays (inverse-scaled before metric).
    """

    def __init__(self, scaler=None):
        self.scaler = scaler

    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        if hasattr(predictions, 'detach'):
            predictions = predictions.detach().cpu().numpy()
        if hasattr(labels, 'detach'):
            labels = labels.detach().cpu().numpy()

        # Replace NaN/Inf to avoid sklearn crash
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

        if self.scaler:
            min_vals = self.scaler['min']
            range_vals = self.scaler['range']
            predictions = predictions * range_vals + min_vals
            labels = labels * range_vals + min_vals

        metrics = {}
        for i, target_name in enumerate(TARGETS):
            pred_vals = predictions[:, i]
            true_vals = labels[:, i]

            metrics[f"rmse_{target_name}"] = np.sqrt(mean_squared_error(true_vals, pred_vals))
            metrics[f"acc_{target_name}"] = evaluate_accuracy(true_vals, pred_vals, target_name)

        return metrics
