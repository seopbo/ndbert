import argparse
import pandas as pd
from pathlib import Path
from utils import Config

parser = argparse.ArgumentParser(description="preprocessing specific dataset")
parser.add_argument(
    "--dataset", type=str, choices=["cr", "mpqa", "mr", "sst2", "subj", "trec"]
)


if __name__ == "__main__":
    args = parser.parse_args()
    parent_dir = Path('dataset')
    data_dir = parent_dir / args.dataset
    list_of_filepath = list(data_dir.iterdir())
    dict_of_filepath = {}

    for filepath in list_of_filepath:
        print(filepath)
        with open(filepath, mode="rb") as io:
            list_of_sentences = io.readlines()
            data = []

            for sentence in list_of_sentences:
                try:
                    decoded_sentence = sentence.strip().decode("utf-8")
                    label = int(decoded_sentence[0])
                    document = decoded_sentence[2:]
                    data.append({"document": document, "label": label})
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
