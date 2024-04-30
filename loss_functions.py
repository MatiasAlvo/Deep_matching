from torch import nn
import torch

class L2Loss(nn.Module):
    """
    Loss that returns the sum of the rewards.
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, not_finished):
        # print(f'pred.shape: {pred.shape}')
        # print(f'target.shape: {target.shape}')
        # print(f'not_finished.shape: {not_finished.shape}')
        # calculate L2 distance of the entries of the matrices
        # loss = torch.sum(((pred - target) ** 2))

        # to each row, stack a column equals to 1 - the sum of the row (to represent 'not matching' in the last column)
        # print(f'pred: {pred[0][0]}')
        pred = torch.cat((pred, 1 - pred.sum(dim=3, keepdim=True)), dim=3)
        # print(f'post: {pred[0][0]}')
        # assert False
        target = torch.cat((target, 1 - target.sum(dim=3, keepdim=True)), dim=3)
        loss = torch.sum(((pred - target) ** 2)*(not_finished.unsqueeze(2).unsqueeze(3)))
        # print(f'loss: {loss}')
        # loss = torch.sum((pred - target) ** 2)/(pred.shape[0] * pred.shape[1] * pred.shape[2])
        # get the number of entries in pred
        # num_entries = pred.shape[0] * pred.shape[1] * pred.shape[2]
        return loss