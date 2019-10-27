import argparse
import pickle
from pathlib import Path
from urllib.request import urlretrieve
from model.utils import Vocab
from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP


parser = argparse.ArgumentParser(description="download pretrained-bert")
parser.add_argument(
    "--type",
    type=str,
    choices=[
        "bert-base-uncased",
        "bert-large-uncased",
        "bert-base-cased",
        "bert-large-cased",
    ],
    default="bert-base-uncased",
)


if __name__ == "__main__":
    args = parser.parse_args()
    ptr_dir = Path("pretrained")
    if not ptr_dir.exists():
        ptr_dir.mkdir(parents=True)

    # saving config of pretrained model
    config_url = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.get(args.type)
    config_filename = config_url.split("/")[-1]
    config_filepath = ptr_dir / config_filename

    if not config_filepath.exists():
        urlretrieve(config_url, config_filepath)
    else:
        print("Already you have {}".format(config_filename))

    print("Saving the config of {} is done.".format(args.type))

    # saving vocab of pretrained model
    ptr_tokenizer = BertTokenizer.from_pretrained(
        args.type, do_lower_case="uncased" in args.type
    )

    idx_to_token = list(ptr_tokenizer.vocab.keys())
    token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
    vocab = Vocab(
        idx_to_token,
        padding_token="[PAD]",
        unknown_token="[UNK]",
        bos_token=None,
        eos_token=None,
        reserved_tokens=["[CLS]", "[SEP]", "[MASK]"],
        token_to_idx=token_to_idx,
    )
    vocab_filename = "{}-vocab.pkl".format(args.type)
    vocab_filepath = ptr_dir / vocab_filename

    if not vocab_filepath.exists():
        with open(vocab_filepath, mode="wb") as io:
            pickle.dump(vocab, io)
    else:
        print("Already you have {}".format(vocab_filename))

    print("Saving the vocab of {} is done.".format(args.type))

    # saving weights of pretrained model
    weights_url = BERT_PRETRAINED_MODEL_ARCHIVE_MAP.get(args.type)
    weights_filename = weights_url.split("/")[-1]
    weights_filepath = ptr_dir / weights_filename

    if not weights_filepath.exists():
        urlretrieve(weights_url, weights_filepath)
    else:
        print("Already you have {}".format(weights_filename))

    print("Saving weights of {} is done.".format(args.type))
