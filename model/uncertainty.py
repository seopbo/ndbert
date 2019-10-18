import torch
import numpy as np
from sklearn.covariance import ShrunkCovariance
from collections import defaultdict
from tqdm import tqdm


def get_penultimate_feature_params(model, num_classes, data_loader, device, cov=ShrunkCovariance()):
    if model.training:
        model.eval()

    penultimate_feature_each_class = defaultdict(list)
    penultimate_feature_mean_each_class = defaultdict(float)
    penultimate_feature_precision_each_class = defaultdict(float)

    for mb in tqdm(data_loader, total=len(data_loader)):
        x_mb, y_mb = map(lambda elm: elm.to(device), mb)
        with torch.no_grad():
            _, penultimate_features = model(x_mb, out_all_hidden_states=False)

            for idx in range(y_mb.size()[0]):
                penultimate_feature_each_class[y_mb[idx].item()].append(penultimate_features[idx].cpu().numpy())
    else:
        for label in range(num_classes):
            tmp = np.vstack(penultimate_feature_each_class.pop(label))
            penultimate_feature_mean_each_class[label] = tmp.mean(axis=0, keepdims=True)
            penultimate_feature_precision_each_class[label] = cov.fit(tmp).precision_.astype(np.float32)
            del tmp  # for memory usage

    return penultimate_feature_mean_each_class, penultimate_feature_precision_each_class


def get_feature_params(model, num_classes, data_loader, device, cov=ShrunkCovariance()):
    # 13개 feature에 대해서 각 feature의 mean과 precision을 계산
    if model.training:
        model.eval()

    ops_indices = list(range(13))
    layer_feature_per_class = {ops_idx: defaultdict(list) for ops_idx in ops_indices}
    layer_feature_mean_per_class = {ops_idx: defaultdict(float) for ops_idx in ops_indices}
    layer_feature_precision_per_class = {ops_idx: defaultdict(float) for ops_idx in ops_indices}

    for mb in tqdm(data_loader, total=len(data_loader)):
        x_mb, y_mb = map(lambda elm: elm.to(device), mb)
        with torch.no_grad():
            _, encoded_features = model(x_mb, out_all_hidden_states=True)

        for ops_idx in ops_indices:

            for idx in range(y_mb.size()[0]):
                layer_feature_per_class[ops_idx][y_mb[idx].item()].append(encoded_features[ops_idx][idx].cpu().numpy())
    else:
        for ops_idx in tqdm(ops_indices, total=len(ops_indices)):
            for label in range(num_classes):
                tmp = np.vstack(layer_feature_per_class[ops_idx].pop(label))
                layer_feature_mean_per_class[ops_idx][label] = tmp.mean(axis=0, keepdims=True)
                layer_feature_precision_per_class[ops_idx][label] = cov.fit(tmp).precision_.astype(np.float32)
                del tmp  # for memory usage

    return layer_feature_mean_per_class, layer_feature_precision_per_class


def get_mcb_score(features, feature_mean_each_class, feature_precision_each_class, topk=3):
    mean = feature_mean_each_class
    precision = feature_precision_each_class

    with torch.no_grad():
        residuals = features - mean
        residual_t = residuals.permute(0, 2, 1)

        score = -1 * torch.bmm(torch.bmm(residuals, precision), residual_t)[:, range(features.size(0)),
                     range(features.size(0))].t()
        # score, _ = score.max(dim=1)
        score_topk = torch.topk(score, k=topk).values
    return score_topk
