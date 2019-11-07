import argparse
from subprocess import call
from itertools import product

parser = argparse.ArgumentParser()
parser.add_argument('--ind', type=str, choices=["cr", "mpqa", "mr", "sst2", "subj", "trec"])
parser.add_argument('--type', type=str, default='bert-base-uncased')
parser.add_argument('--nh', type=int, action='store', nargs='*')
parser.add_argument('--topk', type=int, action='store', nargs='*')


if __name__ == '__main__':
    args = parser.parse_args()
    list_of_data = ['cr', 'mpqa', 'mr', 'sst2', 'subj', 'trec']

    call('python train.py --par_dir ind_{} --type {}'.format(args.ind, args.type), shell=True)
    call('python evaluate.py --par_dir ind_{} --type {} --data train_ind'.format(
        args.ind, args.type), shell=True)
    call('python evaluate.py --par_dir ind_{} --type {} --data test_ind'.format(
        args.ind, args.type), shell=True)

    for nh in args.nhs:
        call('python extract_params.py --par_dir ind_{} --type {} --nh {}'.format(args.ind, args.type, nh), shell=True)

    count = 1
    for sub_dir in list_of_data:
        if sub_dir == args.ind:
            continue

        for comb in product(args.topk, args.nh):
            call('python train_detector.py --par_dir ind_{} --sub_dir ood_{} --type {} --topk {} --nh {}'.format(
                args.ind, sub_dir, args.type, comb[0], comb[1]), shell=True)

            for tgt_dir in list_of_data:
                if tgt_dir == args.ind:
                    continue

                call('python evaluate_detector.py --par_dir ind_{} --sub_dir ood_{} --tgt_dir ood_{} --type {} --topk {}'
                     ' --nh {}'.format(args.ind, sub_dir, tgt_dir, args.type, comb[0], comb[1]), shell=True)
                count += 1
                print(count)