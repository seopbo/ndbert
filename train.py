import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from transformers.modeling_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer
from model.net import BertClassifier
from model.data import Corpus
from model.utils import PreProcessor, PadSequence
from model.metric import evaluate, acc, LSR
from utils import Config, CheckpointManager, SummaryManager, replace_key
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

# for reproducibility
torch.manual_seed(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument("--ind", default="trec",
                    help="directory of in distribution is not sub-directory")
parser.add_argument("--type", default="bert-base-uncased", help="pretrained weights of bert")


if __name__ == "__main__":
    args = parser.parse_args()
    par_dir = Path(args.ind)
    backbone_dir = Path('experiments') / args.ind
    ptr_dir = Path("pretrained")
    data_config = Config(par_dir / "config.json")
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

    # model
    config_filepath = ptr_dir / "{}-config.json".format(args.type)
    config = BertConfig.from_pretrained(config_filepath, output_hidden_states=False)
    model = BertClassifier(
        config, num_classes=model_config.num_classes, vocab=preprocessor.vocab
    )
    bert_checkpoint = torch.load(ptr_dir / "{}-pytorch_model.bin".format(args.type))
    bert_checkpoint = OrderedDict(
        [(replace_key(k), bert_checkpoint.get(k)) for k in bert_checkpoint.keys()]
    )
    model.load_state_dict(bert_checkpoint, strict=False)

    # training
    tr_ds = Corpus(data_config.train, preprocessor.preprocess)
    tr_dl = DataLoader(
        tr_ds,
        batch_size=model_config.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    dev_ds = Corpus(data_config.dev, preprocessor.preprocess)
    dev_dl = DataLoader(dev_ds, batch_size=model_config.batch_size, num_workers=4)

    loss_fn = nn.CrossEntropyLoss()

    opt = optim.Adam(
        [
            {"params": model.bert.parameters(), "lr": model_config.learning_rate / 100},
            {"params": model.classifier.parameters(), "lr": model_config.learning_rate}
        ], weight_decay=5e-4)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    writer = SummaryWriter("{}/runs".format(backbone_dir))
    checkpoint_manager = CheckpointManager(backbone_dir)
    summary_manager = SummaryManager(backbone_dir)
    best_val_loss = 1e10

    for epoch in tqdm(range(model_config.epochs), desc="epochs"):

        tr_loss = 0
        tr_acc = 0

        model.train()
        for step, mb in tqdm(enumerate(tr_dl), desc="steps", total=len(tr_dl)):
            x_mb, y_mb = map(lambda elm: elm.to(device), mb)
            opt.zero_grad()
            y_hat_mb, _ = model(x_mb)
            mb_loss = loss_fn(y_hat_mb, y_mb)
            mb_loss.backward()
            opt.step()

            with torch.no_grad():
                mb_acc = acc(y_hat_mb, y_mb)

            tr_loss += mb_loss.item()
            tr_acc += mb_acc.item()

            if (epoch * len(tr_dl) + step) % model_config.summary_step == 0:
                dev_loss = evaluate(model, dev_dl, {"loss": loss_fn}, device)["loss"]
                writer.add_scalars(
                    "loss",
                    {"train": tr_loss / (step + 1), "dev": dev_loss},
                    epoch * len(tr_dl) + step,
                )
                tqdm.write(
                    "global_step: {:3}, tr_loss: {:.3f}, dev_loss: {:.3f}".format(
                        epoch * len(tr_dl) + step, tr_loss / (step + 1), dev_loss
                    )
                )
                model.train()
        else:
            tr_loss /= step + 1
            tr_acc /= step + 1

            tr_summ = {"loss": tr_loss, "acc": tr_acc}
            dev_summ = evaluate(model, dev_dl, {"loss": loss_fn, "acc": acc}, device)
            tqdm.write(
                "epoch : {}, tr_loss: {:.3f}, dev_loss: "
                "{:.3f}, tr_acc: {:.2%}, dev_acc: {:.2%}".format(
                    epoch + 1,
                    tr_summ["loss"],
                    dev_summ["loss"],
                    tr_summ["acc"],
                    dev_summ["acc"],
                )
            )

            dev_loss = dev_summ["loss"]
            is_best = dev_loss < best_val_loss

            if is_best:
                state = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "opt_state_dict": opt.state_dict(),
                }
                summary = {"train": tr_summ, "dev": dev_summ}

                summary_manager.update(summary)
                summary_manager.save("summary.json")
                checkpoint_manager.save_checkpoint(state, "best.tar")

                best_val_loss = dev_loss
