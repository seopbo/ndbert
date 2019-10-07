import argparse
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_pretrained_bert.modeling import BertConfig
from pretrained.tokenization import BertTokenizer
from model.net import BertClassifier
from model.data import Corpus
from model.utils import Tokenizer, PadSequence
from model.metric import evaluate, acc
from utils import Config, CheckpointManager, SummaryManager

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='exp_1st_corpus', help="Directory containing config.json of data")
parser.add_argument('--restore_dir', default='experiments/exp_1st_corpus', help="Directory containing config.json of model")
parser.add_argument('--data_name', default='val', help="name of the data in_tagging --data_dir to be evaluate")


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

    # evaluation
    filepath = getattr(data_config, args.data_name)
    ds = Corpus(filepath, tokenizer.preprocess)
    dl = DataLoader(ds, batch_size=model_config.batch_size, num_workers=4)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    summ = evaluate(model, dl, {'loss': nn.CrossEntropyLoss(), 'acc': acc}, device)

    summary_manager = SummaryManager(restore_dir)
    summary_manager.load('summary.json')
    summary_manager.update({'{}'.format(args.data_name): summ})
    summary_manager.save('summary.json')

    print(args.data_name)
    print('loss: {:.3f}, acc: {:.2%}'.format(summ['loss'], summ['acc']))
