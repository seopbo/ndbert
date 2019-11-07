import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers.modeling_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer
from model.net import BertClassifier
from model.data import Corpus
from model.utils import PreProcessor, PadSequence
from model.uncertainty import get_mcb_score
from sklearn.metrics import classification_report
from tqdm import tqdm
from utils import Config, CheckpointManager, SummaryManager


parser = argparse.ArgumentParser()
parser.add_argument("--par_dir", default="ind_trec")
parser.add_argument("--sub_dir", default="ood_cr")
parser.add_argument("--tgt_dir", default='ood_cr')
parser.add_argument("--type", default="bert-base-uncased", help="pretrained weights of bert")
parser.add_argument('--topk', default=1, type=int)
parser.add_argument("--nh", default=12, type=int, help="using hidden states of model from the last hidden state")


if __name__ == '__main__':
    args = parser.parse_args()
    par_dir = Path(args.par_dir)
    sub_dir = par_dir / args.sub_dir
    tgt_dir = par_dir / args.tgt_dir
    backbone_dir = Path('experiments') / args.par_dir
    detector_dir = backbone_dir / args.sub_dir
    ptr_dir = Path("pretrained")
    par_config = Config(par_dir / "config.json")
    tgt_config = Config(tgt_dir / 'config.json')
    model_config = Config(backbone_dir / "config.json")

    # tokenizer
    ptr_tokenizer = BertTokenizer.from_pretrained(
        args.type, do_lower_case="uncased" in args.type
    )
    vocab_filepath = ptr_dir / "{}-vocab.pkl".format(args.type)
    with open(vocab_filepath, mode="rb") as io:
        vocab = pickle.load(io)
    pad_sequence = PadSequence(
        length=model_config.length, pad_val=vocab.to_indices(vocab.padding_token)
    )
    preprocessor = PreProcessor(
        vocab=vocab, split_fn=ptr_tokenizer.tokenize, pad_fn=pad_sequence
    )

    # model (restore)
    checkpoint_manager = CheckpointManager(backbone_dir)
    checkpoint = checkpoint_manager.load_checkpoint("best.tar")
    config_filepath = ptr_dir / "{}-config.json".format(args.type)

    if args.nh == 1:
        config = BertConfig.from_pretrained(config_filepath, output_hidden_states=False)
    else:
        config = BertConfig.from_pretrained(config_filepath, output_hidden_states=True)

    model = BertClassifier(
        config, num_classes=model_config.num_classes, vocab=preprocessor.vocab
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    model.eval()
    model.to(device)

    # evaluate detector
    test_ind_ds = Corpus(par_config.test_ind, preprocessor.preprocess)
    test_ind_dl = DataLoader(test_ind_ds, batch_size=128, num_workers=4)
    test_ood_ds = Corpus(tgt_config.test_ood, preprocessor.preprocess)
    test_ood_dl = DataLoader(test_ood_ds, batch_size=128, num_workers=4)

    with open(backbone_dir / 'feature_params_{}.pkl'.format(args.nh), mode='rb') as io:
        feature_params = pickle.load(io)
    ops_indices = list(range(len(feature_params['mean'].keys())))
    ind_features = []

    for ops_idx in tqdm(ops_indices, total=len(ops_indices)):
        if args.nh == 1:
            layer_mean = torch.tensor(list(feature_params['mean'].values())).to(device)
            layer_precision = torch.tensor(list(feature_params['precision'].values())).to(device)
        else:
            layer_mean = torch.tensor(list(feature_params['mean'][ops_idx].values())).to(device)
            layer_precision = torch.tensor(list(feature_params['precision'][ops_idx].values())).to(device)

        mb_features = []
        for mb in tqdm(test_ind_dl, total=len(test_ind_dl)):

            x_mb, _ = map(lambda elm: elm .to(device), mb)

            with torch.no_grad():
                _, encoded_layers = model(x_mb)

                if args.nh == 1:
                    mb_features.extend(get_mcb_score(encoded_layers, layer_mean,
                                                     layer_precision, topk=args.topk).cpu().numpy().tolist())
                else:
                    mb_features.extend(get_mcb_score(encoded_layers[ops_idx], layer_mean,
                                                     layer_precision, topk=args.topk).cpu().numpy().tolist())
        else:
            ind_features.append(mb_features)
    else:
        ind_features = np.concatenate(ind_features, axis=1)
        ind_label = np.zeros(ind_features.shape[0])

    ood_features = []
    for ops_idx in tqdm(ops_indices, total=len(ops_indices)):
        if args.nh == 1:
            layer_mean = torch.tensor(list(feature_params['mean'].values())).to(device)
            layer_precision = torch.tensor(list(feature_params['precision'].values())).to(device)
        else:
            layer_mean = torch.tensor(list(feature_params['mean'][ops_idx].values())).to(device)
            layer_precision = torch.tensor(list(feature_params['precision'][ops_idx].values())).to(device)

        mb_features = []

        for mb in tqdm(test_ood_dl, total=len(test_ood_dl)):

            x_mb, _ = map(lambda elm: elm.to(device), mb)

            with torch.no_grad():
                _, encoded_layers = model(x_mb)

                if args.nh == 1:
                    mb_features.extend(get_mcb_score(encoded_layers, layer_mean,
                                                     layer_precision, topk=args.topk).cpu().numpy().tolist())
                else:
                    mb_features.extend(get_mcb_score(encoded_layers[ops_idx], layer_mean,
                                                     layer_precision, topk=args.topk).cpu().numpy().tolist())
        else:
            ood_features.append(mb_features)
    else:
        ood_features = np.concatenate(ood_features, axis=1)
        ood_label = np.ones(ood_features.shape[0])

    with open(detector_dir / 'detector_topk_{}_nh_{}.pkl'.format(args.topk, args.nh), mode='rb') as io:
        detector = pickle.load(io)

    X = np.concatenate([ind_features, ood_features])
    y = np.concatenate([ind_label, ood_label])

    yhat = detector['lr'].predict(X)

    lr_summary = classification_report(y, yhat, target_names=['ind', 'ood'], output_dict=True)
    lr_summary = dict(**lr_summary)
    lr_summary = {'test_{}'.format(args.par_dir + '_' + args.tgt_dir): lr_summary}

    summary_manger = SummaryManager(backbone_dir)
    summary_manger.load('summary.json')
    summary_manger.summary['{}_{}_topk_{}_nh_{}'.format(args.par_dir,
                                                        args.tgt_dir, args.topk, args.nh)].update(lr_summary)
    summary_manger.save('summary.json')
