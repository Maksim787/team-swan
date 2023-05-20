import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from constants import N_CLASSES, device

############################################################
# CrossEntropyLoss - baseline loss
############################################################

CrossEntropyLoss = nn.CrossEntropyLoss

############################################################
# ScoreLoss - task specific loss
############################################################

SCORE_MATRIX = np.array(
    [
        [3, -1, -3],
        [-1, 3, -3],
        [-3, -3, 2]
    ], dtype=np.float64
)
assert np.all(SCORE_MATRIX == SCORE_MATRIX.T)


# class ScoreLoss(nn.Module):
#     def __init__(self, alpha: float) -> None:
#         super().__init__()
#         self.alpha = alpha
#         self.cross_entropy = CrossEntropyLoss()

#     def forward(self, logits, labels):
#         probs = F.softmax(logits, dim=-1)
#         score = 0.0
#         for i in range(3):
#             for j in range(3):
#                 score += (probs[i] * (labels == j).reshape(-1, 1).repeat(1, N_CLASSES) * SCORE_MATRIX[i, j]).sum(dim=-1)
#         return -score.mean() + self.alpha * self.cross_entropy(logits, labels)
