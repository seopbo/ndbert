import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_pretrained_bert.modeling import BertConfig
from pretrained.tokenization import BertTokenizer
from model.net import BertClassifier
from model.data import Corpus
from model.utils import Tokenizer, PadSequence
from model.uncertainty import get_mcb_score
from sklearn.metrics import classification_report
from tqdm import tqdm
from utils import Config, CheckpointManager, SummaryManager


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='active/train', help="Directory containing config.json of data")
parser.add_argument('--restore_dir', default='experiments/train', help="Directory containing config.json of model")
parser.add_argument('--topk', default=10, type=int)


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    restore_dir = Path(args.restore_dir)
    data_config = Config(json_path=data_dir / 'config.json')
    model_config = Config(json_path=restore_dir / 'config.json')

    # tokenizer
    ptr_tokenizer = BertTokenizer.from_pretrained('pretrained/vocab.korean.rawtext.list', do_lower_case=False)
    with open('pretrained/vocab.pkl', mode='rb') as io:
        vocab = pickle.load(io)
    pad_sequence = PadSequence(length=model_config.length, pad_val=vocab.to_indices(vocab.padding_token))
    tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer.tokenize, pad_fn=pad_sequence)

    # model (restore)
    checkpoint_manager = CheckpointManager(restore_dir)
    ckpt = checkpoint_manager.load_checkpoint('best.tar')
    config = BertConfig('pretrained/bert_config.json')
    model = BertClassifier(config, num_labels=model_config.num_classes, vocab=tokenizer.vocab)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    model.to(device)

    # evaluate detector
    ind_val_ds = Corpus(data_config.val_ind, tokenizer.preprocess)
    ind_val_dl = DataLoader(ind_val_ds, batch_size=128, num_workers=4)
    ood_val_ds = Corpus(data_config.val_ood, tokenizer.preprocess)
    ood_val_dl = DataLoader(ood_val_ds, batch_size=128, num_workers=4)

    with open(restore_dir / 'feature_params.pkl', mode='rb') as io:
        feature_params_all = pickle.load(io)
    ops_indices = list(range(len(feature_params_all['mean'].keys())))

    ind_features = []
    for ops_idx in tqdm(ops_indices, total=len(ops_indices)):
        layer_mean = torch.tensor(list(feature_params_all['mean'][ops_idx].values())).to(device)
        layer_precision = torch.tensor(list(feature_params_all['precision'][ops_idx].values())).to(device)
        mb_features = []
        for mb in tqdm(ind_val_dl, total=len(ind_val_dl)):

            x_mb, _ = map(lambda elm: elm .to(device), mb)

            with torch.no_grad():
                _, encoded_layers = model(x_mb, output_all_encoded_layers=True)
                mb_features.extend(get_mcb_score(encoded_layers[ops_idx], layer_mean,
                                                 layer_precision, topk=args.topk).cpu().numpy().tolist())
        else:
            ind_features.append(mb_features)
    else:
        ind_features = np.concatenate(ind_features, axis=1)
        ind_label = np.zeros(ind_features.shape[0])

    ood_features = []
    for ops_idx in tqdm(ops_indices, total=len(ops_indices)):
        layer_mean = torch.tensor(list(feature_params_all['mean'][ops_idx].values())).to(device)
        layer_precision = torch.tensor(list(feature_params_all['precision'][ops_idx].values())).to(device)
        mb_features = []

        for mb in tqdm(ood_val_dl, total=len(ood_val_dl)):

            x_mb, _ = map(lambda elm: elm.to(device), mb)

            with torch.no_grad():
                _, encoded_layers = model(x_mb, output_all_encoded_layers=True)
                mb_features.extend(get_mcb_score(encoded_layers[ops_idx], layer_mean,
                                                 layer_precision, topk=args.topk).cpu().numpy().tolist())
        else:
            ood_features.append(mb_features)
    else:
        ood_features = np.concatenate(ood_features, axis=1)
        ood_label = np.ones(ood_features.shape[0])

    X = np.concatenate([ind_features, ood_features])
    y = np.concatenate([ind_label, ood_label])

    with open(restore_dir / 'detector_{}.pkl'.format(args.topk), mode='rb') as io:
        detector = pickle.load(io)

    X = detector['sc'].transform(X)
    yhat = detector['lr'].predict(X)

    lr_summary = classification_report(y, yhat,
                                       target_names=['ind_val', 'ood_val'], output_dict=True)
    lr_summary = dict(**lr_summary)
    lr_summary = {'ood_validation_{}_{}'.format(args.topk, args.data_dir): lr_summary}
    summary_manger = SummaryManager(restore_dir)
    summary_manger.load('summary.json')
    summary_manger.update(lr_summary)
    summary_manger.save('summary.json')
