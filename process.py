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

    call('python train.py --ind {} --type {}'.format(args.ind, args.type), shell=True)
    call('python evaluate.py --ind {} --type {} --data train'.format(args.ind, args.type), shell=True)
    call('python evaluate.py --ind {} --type {} --data test'.format(args.ind, args.type), shell=True)

    for nh in args.nh:
        call('python extract_params.py --ind {} --type {} --nh {}'.format(args.ind, args.type, nh), shell=True)

    count = 1
    for ood in list_of_data:
        if ood == args.ind:
            continue

        for comb in product(args.topk, args.nh):
            call('python train_detector.py --ind {} --ood {} --type {} --topk {} --nh {}'.format(
                args.ind, ood, args.type, comb[0], comb[1]), shell=True)

            for tgt in list_of_data:

                call('python evaluate_detector.py --ind {} --ood {} --tgt {} --type {} --topk {}'
                     ' --nh {}'.format(args.ind, ood, tgt, args.type, comb[0], comb[1]), shell=True)
                count += 1
                print(count)

    call('python synthesize_experiment.py --ind {}'.format(args.ind), shell=True)