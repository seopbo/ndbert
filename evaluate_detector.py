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
from sklearn.metrics import precision_score, f1_score, recall_score
from tqdm import tqdm
from utils import Config, CheckpointManager, SummaryManager


parser = argparse.ArgumentParser()
parser.add_argument("--ind", default="trec",
                    help="directory of in distribution is not sub-directory")
parser.add_argument("--ood", default="cr",
                    help="directory of out of distribution is sub-directory from directory of in distribution")
parser.add_argument("--tgt", default='cr')
parser.add_argument("--type", default="bert-base-uncased", help="pretrained weights of bert")
parser.add_argument('--topk', default=1, type=int)
parser.add_argument('--nh', default=12, type=int, help="using hidden states of model from the last hidden state")
parser.add_argument('--data', default="test", help="predicting specific data")


if __name__ == '__main__':
    args = parser.parse_args()
    par_dir = Path(args.ind)
    sub_dir = par_dir / args.ood

    if args.tgt == args.ind:
        tgt_dir = par_dir
    else:
        tgt_dir = par_dir / args.tgt

    backbone_dir = Path('experiments') / args.ind
    detector_dir = backbone_dir / args.ood
    ptr_dir = Path("pretrained")
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
    filepath = getattr(tgt_config, args.data)
    ds = Corpus(filepath, preprocessor.preprocess)
    dl = DataLoader(ds, batch_size=128, num_workers=4)

    with open(backbone_dir / 'feature_params_{}.pkl'.format(args.nh), mode='rb') as io:
        feature_params = pickle.load(io)
    ops_indices = list(range(len(feature_params['mean'].keys())))
    features = []

    for ops_idx in tqdm(ops_indices, total=len(ops_indices)):
        if args.nh == 1:
            layer_mean = torch.tensor(list(feature_params['mean'].values())).to(device)
            layer_precision = torch.tensor(list(feature_params['precision'].values())).to(device)
        else:
            layer_mean = torch.tensor(list(feature_params['mean'][ops_idx].values())).to(device)
            layer_precision = torch.tensor(list(feature_params['precision'][ops_idx].values())).to(device)

        mb_features = []
        for mb in tqdm(dl, total=len(dl)):

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
            features.append(mb_features)
    else:
        features = np.concatenate(features, axis=1)
        label = np.zeros(features.shape[0]) if args.ind == args.tgt else np.ones(features.shape[0])

    with open(detector_dir / 'detector_topk={}_nh={}.pkl'.format(args.topk, args.nh), mode='rb') as io:
        detector = pickle.load(io)

    yhat = detector['lr'].predict(features)
    lr_summary = {'precision': precision_score(label, yhat, pos_label=0 if args.ind == args.tgt else 1),
                  'recall': recall_score(label, yhat, pos_label=0 if args.ind == args.tgt else 1),
                  'f1-score': f1_score(label, yhat, pos_label=0 if args.ind == args.tgt else 1),
                  'support': len(ds)}
    lr_summary = {'{}_{}'.format(args.data, args.tgt): lr_summary}

    summary_manger = SummaryManager(backbone_dir)
    summary_manger.load('summary.json')
    summary_manger.summary['{}&{}_topk={}_nh={}'.format(args.ind,
                                                        args.ood, args.topk, args.nh)].update(lr_summary)
    summary_manger.save('summary.json')
