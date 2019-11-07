import argparse
import pandas as pd
from pathlib import Path
from utils import Config
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--ind', type=str, choices=["cr", "mpqa", "mr", "sst2", "subj", "trec"])
parser.add_argument('--ood', type=str, choices=["cr", "mpqa", "mr", "sst2", "subj", "trec"])
parser.add_argument('--dev_ind_size', type=int, default=500)
parser.add_argument('--val_ind_size', type=int, default=2000)


if __name__ == '__main__':
    args = parser.parse_args()
    raw_dataset_dir = Path('raw_dataset')
    par_dir = Path('{}'.format(args.ind))

    if not par_dir.exists():
        par_dir.mkdir(parents=True)

    ind_dir = raw_dataset_dir / args.ind
    ood_dir = raw_dataset_dir / args.ood

    ind_config = Config(ind_dir / 'config.json')
    ind_all = []

    for key in ind_config.dict:
        ind_all.append(pd.read_csv(ind_config.dict.get(key), sep='\t'))
    else:
        ind_all = pd.concat(ind_all, ignore_index=True, sort=False)
        ind_all = ind_all[~ind_all['document'].isna()]

    ood_config = Config(ood_dir / 'config.json')
    ood_all = []

    for key in ood_config.dict:
        ood_all.append(pd.read_csv(ood_config.dict.get(key), sep='\t'))
    else:
        ood_all = pd.concat(ood_all, ignore_index=True, sort=False)
        ood_all = ood_all[~ood_all['document'].isna()]

    tr_ind, val_ind = train_test_split(ind_all, test_size=args.val_ind_size, random_state=777)
    tr_ind, dev_ind = train_test_split(tr_ind, test_size=args.dev_ind_size, random_state=777)

    tr_ood, val_ood = train_test_split(ood_all, test_size=args.val_ind_size, random_state=777)
    tr_ood, dev_ood = train_test_split(tr_ood, test_size=args.dev_ind_size, random_state=777)

    sub_dir = par_dir / '{}'.format(args.ood)

    if not sub_dir.exists():
        sub_dir.mkdir(parents=True)

    tr_ind_path = str(par_dir / 'train_ind_{}.txt'.format(len(tr_ind)))
    dev_ind_path = str(par_dir / 'dev_ind_{}.txt'.format(len(dev_ind)))
    val_ind_path = str(par_dir / 'test_ind_{}.txt'.format(len(val_ind)))
    tr_ood_path = str(sub_dir / 'train_ood_{}.txt'.format(len(tr_ood)))
    dev_ood_path = str(sub_dir / 'dev_ood_{}.txt'.format(len(dev_ood)))
    val_ood_path = str(sub_dir / 'test_ood_{}.txt'.format(len(val_ood)))

    tr_ind.to_csv(tr_ind_path, sep='\t', index=False)
    dev_ind.to_csv(dev_ind_path, sep='\t', index=False)
    val_ind.to_csv(val_ind_path, sep='\t', index=False)

    tr_ood.to_csv(tr_ood_path, sep='\t', index=False)
    dev_ood.to_csv(dev_ood_path, sep='\t', index=False)
    val_ood.to_csv(val_ood_path, sep='\t', index=False)

    par_config = Config({'train': tr_ind_path,
                         'dev': dev_ind_path,
                         'test': val_ind_path})
    par_config.save(par_dir / 'config.json')

    sub_config = Config({'train': tr_ood_path,
                         'dev': dev_ood_path,
                         'test': val_ood_path})
    sub_config.save(sub_dir / 'config.json')

    experiment_dir = Path('experiments') / '{}'.format(args.ind) / '{}'.format(args.ood)

    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True)