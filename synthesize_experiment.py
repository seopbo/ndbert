import pandas as pd
import argparse
from pathlib import Path
from utils import SummaryManager
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/ind_trec_ood_sst2',
                    help="directory containing summary.json of experiment")


if __name__ == '__main__':
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    exp_summ = SummaryManager(model_dir)
    exp_summ.load('summary.json')
    list_of_detectors = [key for key in exp_summ.summary.keys() if 'detector' in key]

    result_of_detectors = []

    for detector in tqdm(list_of_detectors):
        result = []

        for key, value in exp_summ.summary.get(detector).items():
            df = pd.DataFrame(value)[['ind', 'ood']].transpose()
            df.index = [[detector]*2, [key + '_acc_{}'.format(value.get('accuracy'))] * 2, df.index]
            result.append(df)
        else:
            result = pd.concat(result)
        result_of_detectors.append(result)
    else:
        result_of_detectors = pd.concat(result_of_detectors)
        result_of_detectors = result_of_detectors.reset_index()
        result_of_detectors.columns = ['model', 'case', 'type', 'precision', 'recall', 'f1_score', 'support']

    result_of_detectors.to_csv(model_dir / 'results_of_detectors.txt', sep='\t', index=False)