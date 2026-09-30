import numpy as np
from mmseg.evaluation.metrics import IoUMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class ClassIoUMetric(IoUMetric):
    def compute_metrics(self, results):
        metrics = super().compute_metrics(results)
        intersect, union, pred, label = (sum(areas).numpy() for areas in zip(*results))
        with np.errstate(invalid="ignore"):
            iou = intersect / union * 100
        self.per_class = {
            name: dict(IoU=None if np.isnan(iou[i]) else round(float(iou[i]), 2), area_intersect=int(intersect[i]),
                       area_union=int(union[i]), area_pred=int(pred[i]), area_label=int(label[i]))
            for i, name in enumerate(self.dataset_meta["classes"])}
        return metrics
