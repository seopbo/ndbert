import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def evaluate(model, data_loader, metrics, device):
    if model.training:
        model.eval()

    summary = {metric: 0 for metric in metrics}

    for step, mb in tqdm(enumerate(data_loader), desc='steps', total=len(data_loader)):
        x_mb, y_mb = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            y_hat_mb, _ = model(x_mb)

            for metric in metrics:
                summary[metric] += metrics[metric](y_hat_mb, y_mb).item() * y_mb.size()[0]
    else:
        for metric in metrics:
            summary[metric] /= len(data_loader.dataset)

    return summary


def acc(yhat, y):
    with torch.no_grad():
        yhat = yhat.max(dim=1)[1]
        acc = (yhat == y).float().mean()
    return acc


def entropy(probs):
    return torch.sum(probs * torch.log(probs), dim=-1)


class LSR(nn.Module):
    def __init__(self, epsilon=.1, num_classes=162):
        super(LSR, self).__init__()
        self._epsilon = epsilon
        self._num_classes = num_classes

    def forward(self, yhat, y):
        prior = torch.div(torch.ones_like(yhat), self._num_classes)
        loss = F.cross_entropy(yhat, y, reduction='none')
        reg = (-1 * F.log_softmax(yhat, dim=-1) * prior).sum(-1)
        total = (1 - self._epsilon) * loss + self._epsilon * reg
        lsr_loss = total.mean()
        return lsr_loss
