import torch
import numpy as np


def random_pairs_of_minibatches(minibatches):
    '''minibatches: (x, y)
    x: a Tensor of shape [B, xxxx]
    y: a Tensor of shape [B, xxxx]
    '''
    # TODO: does not consider overlapping (same) data
    x, y = minibatches

    perm = torch.randperm(x.shape[0]).tolist()
    # xi, yi = minibatches[perm][0], minibatches[perm[i]][1]
    xj, yj = x[perm], y[perm]

    return (x, y), (xj, yj)


class Mixup_helper():
    def __init__(self, mode=None) -> None:
        '''Support Mixup and CutMix'''
        assert mode in ['Mixup', 'CutMix', 'NA']
        self.mode = mode

    def Mixup(self, xi, xj, alpha=0.2):
        '''return data and corresponding two labels, and lam,

        output objective: 
            predictions = self.predict(x)
            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)
        '''
        lam = np.random.beta(alpha, alpha)
        x = lam * xi + (1 - lam) * xj

        return x, lam
    
    def CutMix(self, xi, xj, alpha=0.2):
        '''Randomly cut a portion of xi to replace the portion of xj
        Here we assert the data dimension follows: [B, channel, length], where xxx can be total number of channels
        '''
        lam = np.random.beta(alpha, alpha)
        bbx1, bbx2 = self.generate_random_bbox(xi.shape[-1], lam)

        mixed_image = xi.clone()
        mixed_image[:, :, bbx1:bbx2] = xj[:, :, bbx1:bbx2]

        lam = 1 - ((bbx2 - bbx1) / (xi.shape[-1]))
        return mixed_image, lam

    def generate_random_bbox(self, length, lam):
        cut_rat = 1. - lam

        cut_w = np.round(length * cut_rat)
        cx = np.random.randint(length)

        bbx1 = np.clip(cx - cut_w // 2, 0, length)
        bbx2 = np.clip(cx + cut_w // 2, 0, length)

        return int(bbx1), int(bbx2)

    def forward(self, minibatches):
        if self.mode == 'NA':
            return minibatches
        
        (xi, yi), (xj, yj) = random_pairs_of_minibatches(minibatches)

        if self.mode == 'Mixup':
            x, lam = self.Mixup(xi, xj)
        elif self.mode == 'CutMix':
            x, lam = self.CutMix(xi, xj)
        else:
            raise NotImplementedError

        return x, (lam, yi, yj)

