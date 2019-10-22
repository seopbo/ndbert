import argparse
import pandas as pd
# import re
from pathlib import Path
from utils import Config

#
# def clean_str(string, TREC=False):
#     """
#     Tokenization/string cleaning for all datasets except for SST.
#     Every dataset is lower cased except for TREC
#     """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip() if TREC else string.strip().lower()
#
#
# def clean_str_sst(string):
#     """
#     Tokenization/string cleaning for the SST dataset
#     """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip().lower()


parser = argparse.ArgumentParser(description="preprocessing specific dataset")
parser.add_argument(
    "--data", type=str, choices=["cr", "mpqa", "mr", "sst2", "subj", "trec"]
)


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_dir = Path('.') / "dataset"
    data_dir = dataset_dir / args.data
    list_of_filepath = list(data_dir.iterdir())
    # preprocess_fn = clean_str_sst if 'sst' in args.data else clean_str
    dict_of_filepath = {}

    for filepath in list_of_filepath:
        with open(filepath, mode="rb") as io:
            list_of_sentences = io.readlines()
            data = []

            for sentence in list_of_sentences:
                try:
                    decoded_sentence = sentence.strip().decode("utf-8")
                    label = int(decoded_sentence[0])
                    document = decoded_sentence[2:]
                    data.append({"label": label, "document": document})
                except UnicodeDecodeError:
                    continue
            else:
                data = pd.DataFrame(data)
                save_filepath = str(filepath) + '.txt'
                data.to_csv(save_filepath, sep='\t', index=False)

                if 'train' in str(filepath):
                    dict_of_filepath.update({'train': save_filepath})
                elif 'dev' in str(filepath):
                    dict_of_filepath.update({'dev': save_filepath})
                elif 'test' in str(filepath):
                    dict_of_filepath.update({'test': save_filepath})
                else:
                    dict_of_filepath.update({'all': save_filepath})
    else:
        config = Config(dict_of_filepath)
        config.save(data_dir / 'config.json')
