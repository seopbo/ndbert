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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tqdm import tqdm
from utils import Config, CheckpointManager, SummaryManager

parser = argparse.ArgumentParser()
parser.add_argument("--ind", default="trec",
                    help="directory of in distribution is not sub-directory")
parser.add_argument("--ood", default="cr",
                    help="directory of out of distribution is sub-directory from directory of in distribution")
parser.add_argument("--type", default="bert-base-uncased", help="pretrained weights of bert")
parser.add_argument('--topk', default=1, type=int)
parser.add_argument("--nh", default=14, type=int, help="using hidden states of model from the pooled output")


if __name__ == '__main__':
    args = parser.parse_args()
    par_dir = Path(args.ind)
    sub_dir = par_dir / args.ood
    backbone_dir = Path('experiments') / args.ind
    detector_dir = backbone_dir / args.ood
    ptr_dir = Path("pretrained")
    par_config = Config(par_dir / "config.json")
    sub_config = Config(sub_dir / "config.json")
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

    # train detector
    dev_ind_ds = Corpus(par_config.dev, preprocessor.preprocess)
    dev_ind_dl = DataLoader(dev_ind_ds, batch_size=model_config.batch_size * 4, num_workers=4)
    dev_ood_ds = Corpus(sub_config.dev, preprocessor.preprocess)
    dev_ood_dl = DataLoader(dev_ood_ds, batch_size=model_config.batch_size * 4, num_workers=4)

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

        for mb in tqdm(dev_ind_dl, total=len(dev_ind_dl)):
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

        for mb in tqdm(dev_ood_dl, total=len(dev_ood_dl)):

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

    X = np.concatenate([ind_features, ood_features])
    y = np.concatenate([ind_label, ood_label])

    detector = LogisticRegression(random_state=777, max_iter=2000, solver='lbfgs').fit(X, y)
    y_hat = detector.predict(X)

    lr_summary = classification_report(y, y_hat, target_names=[args.ind, args.ood], output_dict=True)
    lr_summary = {'accuracy': lr_summary['accuracy'],
                  'dev_{}'.format(args.ind): lr_summary[args.ind],
                  'dev_{}'.format(args.ood): lr_summary[args.ood]}
    lr_summary = {'{}&{}_topk={}_nh={}'.format(args.ind, args.ood, args.topk, args.nh): lr_summary}

    summary_manger = SummaryManager(backbone_dir)
    summary_manger.load('summary.json')
    summary_manger.update(lr_summary)
    summary_manger.save('summary.json')

    with open(detector_dir / 'detector_topk={}_nh={}.pkl'.format(args.topk, args.nh), mode='wb') as io:
        pickle.dump({'lr': detector}, io)

