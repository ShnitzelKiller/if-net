from voxels import VoxelGrid
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import glob
import os
import argparse
from data_processing.util import get_name_and_paths
import pathlib
from functools import reduce
import operator


def create_voxel_off(path):
    name, in_path, out_path = get_name_and_paths(path, args.voxpath)

    off_dir = out_path if args.outpath is None else args.outpath

    try:
        voxel_path = os.path.join(out_path, f'{name}.npy')
        off_path = os.path.join(off_dir, f'{name}.off')


        if unpackbits:
            occ = np.unpackbits(np.load(voxel_path))
            voxels = np.reshape(occ, (res,)*3)
        else:
            voxels = np.reshape(np.load(voxel_path)['occupancies'], (res,)*3)

        loc = ((min+max)/2, )*3
        scale = max - min

        VoxelGrid(voxels, loc, scale).to_mesh().export(off_path)
        print('Finished: {}'.format(path))
    except FileNotFoundError as e:
        print(e)







if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run voxalization to off'
    )

    parser.add_argument('-res', type=int)
    parser.add_argument('-root', type=str, default='shapenet/data')
    parser.add_argument('-depth',type=int, default=3)
    parser.add_argument('-count',type=int, default=-1)
    parser.add_argument('-voxpath',type=str, default=None)
    parser.add_argument('-outpath',type=str, default=None)
    parser.add_argument('-split', type=str, default=None)
    parser.add_argument('-mode',type=str, choices=['train','test','val'])

    args = parser.parse_args()

    if args.split is not None:
        split = np.load(args.split)
        paths = split[args.mode]
    else:
        ROOT = pathlib.Path(args.root)
        path = pathlib.Path('*')
        path = reduce(operator.truediv, (path for _ in range(args.depth)))
        paths = glob.glob( str(ROOT / path))

    unpackbits = True
    res = args.res
    min = -0.5
    max = 0.5

    paths = paths if args.count < 0 else paths[:args.count]
    p = Pool(mp.cpu_count())
    p.map(create_voxel_off, paths)