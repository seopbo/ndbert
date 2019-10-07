import argparse
import pickle
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from pytorch_pretrained_bert.modeling import BertConfig
from pretrained.tokenization import BertTokenizer
from model.net import BertClassifier
from model.data import Corpus
from model.utils import Tokenizer, PadSequence
from model.uncertainty import get_feature_params
from utils import Config, CheckpointManager

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='active/train', help="Directory containing config.json of data")
parser.add_argument('--restore_dir', default='experiments/train', help="Directory containing config.json of model")
parser.add_argument('--data_name', default='tr_ind', help='extract params from data denotes data_name')


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
    manager = CheckpointManager(restore_dir)
    ckpt = manager.load_checkpoint('best.tar')
    config = BertConfig('pretrained/bert_config.json')
    model = BertClassifier(config, num_labels=model_config.num_classes, vocab=tokenizer.vocab)
    model.load_state_dict(ckpt['model_state_dict'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # extract feature params
    tr_cps = Corpus('corpus/tr_cps.txt', tokenizer.preprocess)
    filepath = getattr(data_config, args.data_name)
    ds = Corpus(filepath, tokenizer.preprocess)
    ds_concat = ConcatDataset([ds, tr_cps])
    dl = DataLoader(ds_concat, batch_size=model_config.batch_size, num_workers=4)

    mean, precision = get_feature_params(model, model_config.num_classes, dl, device)
    with open(restore_dir / 'feature_params.pkl', mode='wb') as io:
        pickle.dump({'mean': mean, 'precision': precision}, io)
