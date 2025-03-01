import torch
import torch.nn as nn
from mmseg.registry import MODELS

@MODELS.register_module()
class BoundaryDoULoss(nn.Module):
    """
    Implements loss function from "Boundary Difference Over Union Loss For Medical Image Segmentation".
    Focuses on boundary region segmentation by calculating ratio of difference set
    to union of difference and partial intersection sets.

    Features:
    - Adaptive attention to boundary regions based on target size
    - Region-based calculation without need for explicit boundary extraction
    - Stable training without additional loss terms

    Attributes:
        n_classes (int): Number of segmentation classes
    """

    def __init__(self, n_classes=2, loss_weight=1.0, use_sigmoid=False):
        super().__init__()
        self._loss_name = 'loss_boundary_dou'
        self.n_classes = n_classes
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid

    def _adaptive_size(self, score, target):
        """
        Calculate adaptive boundary loss for a single class.

        Args:
            score: Predicted segmentation map (B, H, W)
            target: Ground truth segmentation map (B, H, W)

        Returns:
            Loss value incorporating boundary attention

        Implementation:
            1. Uses 3x3 kernel to detect boundary pixels
            2. Calculates adaptive alpha based on boundary/region ratio
            3. Applies truncated alpha (max 0.8) for stability
            4. Computes final loss using intersection and union terms
        """
        kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).half()
        padding_out = torch.zeros(
            (target.shape[0], target.shape[-2] + 2, target.shape[-1] + 2)
        )
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros(
            (
                padding_out.shape[0],
                padding_out.shape[1] - h + 1,
                padding_out.shape[2] - w + 1,
            )
        ).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(
                target[i].unsqueeze(0).unsqueeze(0).half(),
                kernel.unsqueeze(0).unsqueeze(0).cuda(),
                padding=1,
            )

        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(
            alpha, 0.8
        )  # We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (
            z_sum + y_sum - (1 + alpha) * intersect + smooth
        )

        return self.loss_weight * loss

    def forward(self, pred, target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255):
        assert ignore_index == 255, 'ignore_index is not supported'
        assert reduction_override is None, 'reduction_override is not supported'
        assert avg_factor is None, 'avg_factor is not supported'
        assert weight is None, 'weight is not supported'
        
        if self.use_sigmoid:
            pred = pred.sigmoid()
        assert (
            pred.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            pred.size(), target.size()
        )

        # return self._adaptive_size(inputs, target)
        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(pred[:, i], target[:, i])
        return loss / self.n_classes
    
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name