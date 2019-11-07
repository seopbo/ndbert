import pandas as pd
import argparse
from pathlib import Path
from utils import SummaryManager
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--ind', default='trec',
                    help="directory of in distribution containing summary.json from experiments directory")


if __name__ == '__main__':
    args = parser.parse_args()
    backbone_dir = Path('experiments') / args.ind
    exp_summ = SummaryManager(backbone_dir)
    exp_summ.load('summary.json')

    list_of_detectors = [key for key in exp_summ.summary.keys() if args.ind in key]
    result_of_detectors = []

    for detector in tqdm(list_of_detectors):
        tmp = exp_summ.summary.get(detector)
        tmp_acc = tmp.pop('accuracy')
        tmp_df = pd.DataFrame(tmp.values())
        tmp_df = tmp_df.set_index([[detector] * len(tmp.keys()), [tmp_acc] * len(tmp.keys()), list(tmp.keys())])
        tmp_df = tmp_df.reset_index()
        tmp_df.columns = ['detector', 'dev_accuracy', 'data'] + list(tmp_df.columns[3:])
        result_of_detectors.append(tmp_df)
    else:
        results = pd.concat(result_of_detectors, sort=False, ignore_index=True)

    results.to_csv(backbone_dir / 'results_of_detectors.txt', sep='\t', index=False)