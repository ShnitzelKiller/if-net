import argparse
import os
import functools
import numpy as np

parser = argparse.ArgumentParser(
    description='Create splits file'
)

parser.add_argument('--split', nargs=3, type=int)
parser.add_argument('--paths', type=str, nargs='+')
parser.add_argument('--splitpath', default='.', type=str)

def intersection(set1, set2):
    return set1.intersection(set2)

args = parser.parse_args()

id_to_path = {os.path.splitext(d.name)[0].split('_')[0]: d.path for d in os.scandir(args.paths[0])}
flists = [{os.path.splitext(d.name)[0].split('_')[0] for d in os.scandir(path)} for path in args.paths]
flist = np.array([id_to_path[ind] for ind in functools.reduce(intersection, flists)])

split = args.split
N = len(flist)
splittotal = sum(split)
train_i = int(split[0] / splittotal * N)
test_i = int((split[0] + split[1]) / splittotal * N)

split_dict = {'train': flist[:train_i],'test': flist[train_i:test_i], 'val': flist[test_i:]}

np.savez(os.path.join(args.splitpath, 'split.npz'), **split_dict)