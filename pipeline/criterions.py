import numpy as np
from torch import nn

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
