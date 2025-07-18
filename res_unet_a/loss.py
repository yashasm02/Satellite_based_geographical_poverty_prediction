import torch 
from torch import nn
import torch.nn.functional as F


class TanimotoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        """
        Input shapes (one-hot encoded)
        y_pred: (b, c, h, w)
        y: (b, c, h, w)
        """
        p, l = y_pred, y
        epsilon = 1e-7

        V = torch.mean(torch.sum(l, dim=[2, 3], dtype=torch.float), dim=0)
        w = V ** -2

        inf = torch.tensor(float('inf'))
        new_weights = torch.where(w == inf, torch.zeros_like(w), w)
        w = torch.where(w == inf, torch.ones_like(w) * torch.max(new_weights), w)

        p2 = p ** 2
        l2 = l ** 2
        
        sum_prod_p_l = torch.sum(p * l, dim=[2, 3])
        numerator = torch.sum(w * sum_prod_p_l)

        sum_p2_l2 = torch.sum(p2 + l2, dim=[2, 3])
        sum_p2_l2_minus_p_l = sum_p2_l2 - sum_prod_p_l
        denominator = torch.sum(w * sum_p2_l2_minus_p_l)

        loss = numerator / (denominator + epsilon)

        return 1 - loss
    

class ComplementedTanimotoLoss(nn.Module):
    def __init__(self):
        self.loss_func = TanimotoLoss()

    def forward(self, y_pred, y):
        y_pred = F.softmax(y_pred, 1)
        loss = self.loss_func(y_pred, y)
        loss_complement = self.loss_func(1 - y_pred, 1 - y)

        return (loss + loss_complement) / 2
    

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y, epsilon=1e-7):
        """
        Input shapes
        y_pred: (b, c, h, w) in binary
        y: (b, h, w) with class indices
        """
        num_classes = y_pred.shape[1]

        y_one_hot = F.one_hot(y.long(), num_classes).permute(0, 3, 1, 2)
        y_pred_softmax = torch.softmax(y_pred, dim=1)

        dice = 0
        for index in range(num_classes):
            dice += dice_coef(y_pred_softmax[:, index, :, :], y_one_hot[:, index, :, :], epsilon)

        dice /= num_classes

        return 1 - dice


def dice_coef(y_pred_one_hot, y_one_hot, epsilon=1e-7):
    """
    Input shapes
    y_pred_one_hot: (b, h, w) in binary
    y_one_hot: (b, h, w) in binary
    """
    y_flatten = y_one_hot.view(y_one_hot.shape[0], -1)
    y_pred_flatten = y_pred_one_hot.view(y_pred_one_hot.shape[0], -1)

    intersection = torch.sum(y_flatten * y_pred_flatten, 1)
    union = torch.sum(y_flatten, 1) + torch.sum(y_pred_flatten, 1) 
    coef = (2 * intersection + epsilon) / (union + epsilon)

    return coef.mean()


if __name__ == '__main__':
    """
    label = torch.randn([4, 5, 5])
    V = torch.sum(label, dim=[1, 2])
    V_mean = torch.mean(V, dim=0)
    print(V.shape)
    print(V_mean.shape)
    """
    
    #y = torch.randint(low=0, high=6, size=(5, 512, 512))
    y_pred = torch.rand(5, 6, 512, 512)
    y = torch.argmax(y_pred.clone(), 1)
    dice_loss = DiceLoss()
    
    print(dice_loss(y_pred, y))
