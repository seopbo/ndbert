import argparse
import pickle
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers.modeling_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer
from model.net import BertClassifier
from model.data import Corpus
from model.utils import PreProcessor, PadSequence
from model.metric import evaluate, acc, LSR
from utils import Config, CheckpointManager, SummaryManager
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='trec', help="directory containing config.json of data")
parser.add_argument('--model_dir', default='experiments/trec', help="directory containing config.json of model")
parser.add_argument('--pretrained', default='bert-base-uncased', help='pretrained weights of bert')


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = Path('dataset') / args.data_dir
    model_dir = Path(args.model_dir)
    pretrained_dir = Path('pretrained')
    data_config = Config(data_dir / 'config.json')
    model_config = Config(model_dir / 'config.json')

    # tokenizer
    ptr_tokenizer = BertTokenizer.from_pretrained(args.pretrained, do_lower_case=True)
    vocab_filepath = pretrained_dir / '{}-vocab.pkl'.format(args.pretrained)
    with open(vocab_filepath, mode='rb') as io:
        vocab = pickle.load(io)
    pad_sequence = PadSequence(length=model_config.length, pad_val=vocab.to_indices(vocab.padding_token))
    preprocessor = PreProcessor(vocab=vocab, split_fn=ptr_tokenizer.tokenize, pad_fn=pad_sequence)

    # model
    config_filepath = pretrained_dir / '{}-config.json'.format(args.pretrained)
    config = BertConfig.from_pretrained(config_filepath, output_hidden_states=True)
    model = BertClassifier(config, num_classes=model_config.num_classes, vocab=preprocessor.vocab)
    pretrained_weights = torch.load(pretrained_dir / '{}-pytorch_model.bin'.format(args.pretrained))
    model.load_state_dict(pretrained_weights, strict=False)

    # training
    tr_ds = Corpus(data_config.train, preprocessor.preprocess)
    tr_dl = DataLoader(tr_ds, batch_size=model_config.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_ds = Corpus(data_config.test, preprocessor.preprocess)
    val_dl = DataLoader(val_ds, batch_size=model_config.batch_size)

    loss_fn = LSR(epsilon=.1, num_classes=model_config.num_classes)


    opt = Adam(
        [
            {"params": model.bert.parameters(), "lr": model_config.learning_rate / 100},
            {"params": model.classifier.parameters(), "lr": model_config.learning_rate},

        ], weight_decay=5e-4)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    writer = SummaryWriter('{}/runs'.format(model_dir))
    checkpoint_manager = CheckpointManager(model_dir)
    summary_manager = SummaryManager(model_dir)
    best_val_loss = 1e+10

    x_mb, y_mb = map(lambda elm: elm.to(device), next(iter(tr_dl)))
    for epoch in tqdm(range(model_config.epochs), desc='epochs'):

        tr_loss = 0
        tr_acc = 0

        model.train()
        for step, mb in tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl)):
            x_mb, y_mb = map(lambda elm: elm.to(device), mb)
            opt.zero_grad()
            y_hat_mb, _ = model(x_mb, out_all_hidden_states=False)
            mb_loss = loss_fn(y_hat_mb, y_mb)
            mb_loss.backward()
            opt.step()

            with torch.no_grad():
                mb_acc = acc(y_hat_mb, y_mb)

            tr_loss += mb_loss.item()
            tr_acc += mb_acc.item()

            if (epoch * len(tr_dl) + step) % model_config.summary_step == 0:
                val_loss = evaluate(model, val_dl, {'loss': loss_fn}, device)['loss']
                writer.add_scalars('loss', {'train': tr_loss / (step + 1),
                                            'val': val_loss}, epoch * len(tr_dl) + step)
                tqdm.write('global_step: {:3}, tr_loss: {:.3f}, val_loss: {:.3f}'.format(epoch * len(tr_dl) + step,
                                                                                         tr_loss / (step + 1),
                                                                                         val_loss))
                model.train()
        else:
            tr_loss /= (step + 1)
            tr_acc /= (step + 1)

            tr_summ = {'loss': tr_loss, 'acc': tr_acc}
            val_summ = evaluate(model, val_dl, {'loss': loss_fn, 'acc': acc}, device)
            tqdm.write('epoch : {}, tr_loss: {:.3f}, val_loss: '
                       '{:.3f}, tr_acc: {:.2%}, val_acc: {:.2%}'.format(epoch + 1, tr_summ['loss'], val_summ['loss'],
                                                                        tr_summ['acc'], val_summ['acc']))

            val_loss = val_summ['loss']
            is_best = val_loss < best_val_loss

            if is_best:
                state = {'epoch': epoch + 1,
                         'model_state_dict': model.state_dict(),
                         'opt_state_dict': opt.state_dict()}
                summary = {'tr': tr_summ, 'val': val_summ}

                summary_manager.update(summary)
                summary_manager.save('summary.json')
                checkpoint_manager.save_checkpoint(state, 'best.tar')

                best_val_loss = val_loss
