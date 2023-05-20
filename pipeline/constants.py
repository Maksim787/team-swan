import torch


############################################################
# Global constants
############################################################

N_CLASSES = 3
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
