from torch import nn
import torch

class L2Loss(nn.Module):
    """
    Loss that returns squared some of the differences between 2 vectors.
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, not_finished, args=None, proposal=True):
        # print(f'pred.shape: {pred.shape}')
        # print(f'target.shape: {target.shape}')
        # print(f'not_finished.shape: {not_finished.shape}')
        # calculate L2 distance of the entries of the matrices
        # loss = torch.sum(((pred - target) ** 2))

        # to each row, stack a column equals to 1 - the sum of the row (to represent 'not matching' in the last column)
        pred = torch.cat((pred, 1 - pred.sum(dim=3, keepdim=True)), dim=3)
        target = torch.cat((target, 1 - target.sum(dim=3, keepdim=True)), dim=3)

        # multiply the last column by the not_finished mask (if not_finished is 0, the algorithm arrived at its terminal state
        # so should not be penalized)
        idx=6
        print(f'pred.shape: {pred[0][idx]}')
        print(f'target.shape: {target[0][idx]}')
        print(f'not_finished.shape: {not_finished[0][idx]}')
        print(f'(pred - target) ** 2)*(not_finished.unsqueeze(2).unsqueeze(3)): {(((pred - target) ** 2)*(not_finished.unsqueeze(2).unsqueeze(3))).shape}')
        print(f'(pred - target) ** 2)*(not_finished.unsqueeze(2).unsqueeze(3)): {(((pred - target) ** 2)*(not_finished.unsqueeze(2).unsqueeze(3)))[0][idx]}')
        # assert False

        loss = torch.sum(((pred - target) ** 2)*(not_finished.unsqueeze(2).unsqueeze(3)))
        print(f'loss: {loss}')
        # loss = torch.sum((pred - target) ** 2)/(pred.shape[0] * pred.shape[1] * pred.shape[2])
        # get the number of entries in pred
        # num_entries = pred.shape[0] * pred.shape[1] * pred.shape[2]
        return loss

class WeightedL2Loss(nn.Module):
    """
    Loss that returns the sum of the rewards, but multiplied by preference weights.
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, not_finished, args=None, proposal=True):
        # print(f'pred.shape: {pred.shape}')
        # print(f'target.shape: {target.shape}')
        # print(f'not_finished.shape: {not_finished.shape}')
        # calculate L2 distance of the entries of the matrices
        # loss = torch.sum(((pred - target) ** 2))

        # to each row, stack a column equals to 1 - the sum of the row (to represent 'not matching' in the last column)
        # pred = torch.cat((pred, 1 - pred.sum(dim=3, keepdim=True)), dim=3)
        # target = torch.cat((target, 1 - target.sum(dim=3, keepdim=True)), dim=3)

        # multiply the last column by the not_finished mask (if not_finished is 0, the algorithm arrived at its terminal state
        # so should not be penalized)
        # idx=6
        # print(f'pred.shape: {pred[0][idx]}')
        # print(f'target.shape: {target[0][idx]}')
        # print(f'not_finished.shape: {not_finished[0][idx]}')
        # print(f'(pred - target) ** 2)*(not_finished.unsqueeze(2).unsqueeze(3)): {(((pred - target) ** 2)*(not_finished.unsqueeze(2).unsqueeze(3))).shape}')
        # print(f'(pred - target) ** 2)*(not_finished.unsqueeze(2).unsqueeze(3)): {(((pred - target) ** 2)*(not_finished.unsqueeze(2).unsqueeze(3)))[0][idx]}')
        # assert False
        if proposal:
            val = args['m_preferences']
            # print(f'pred: {pred.shape}')
            # print(f'val: {val.shape}')
            # print(f'(((pred - target)*val).sum(dim=3)): {(((pred - target)*val).sum(dim=3)).shape}')
            loss = torch.sum(((((pred - target)*val).sum(dim=3)) ** 2)*(not_finished.unsqueeze(2)))
        if not proposal:
            val = args['w_preferences'].transpose(2, 3)
            loss = torch.sum(((((pred - target)*val).sum(dim=2)) ** 2)*(not_finished.unsqueeze(2)))

        # print(f'loss: {loss}')
        # loss = torch.sum((pred - target) ** 2)/(pred.shape[0] * pred.shape[1] * pred.shape[2])
        # get the number of entries in pred
        # num_entries = pred.shape[0] * pred.shape[1] * pred.shape[2]
        return loss