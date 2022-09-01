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
    name, in_path, out_path = get_name_and_paths(path, args.outpath)

    voxel_path = os.path.join(out_path, f'{name}_{res}.npy')
    off_path = os.path.join(out_path, f'{name}_{res}.off')


    if unpackbits:
        occ = np.unpackbits(np.load(voxel_path))
        voxels = np.reshape(occ, (res,)*3)
    else:
        voxels = np.reshape(np.load(voxel_path)['occupancies'], (res,)*3)

    loc = ((min+max)/2, )*3
    scale = max - min

    VoxelGrid(voxels, loc, scale).to_mesh().export(off_path)
    print('Finished: {}'.format(path))







if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run voxalization to off'
    )

    parser.add_argument('-res', type=int)
    parser.add_argument('-root', type=str, default='shapenet/data')
    parser.add_argument('-depth',type=int, default=3)
    parser.add_argument('-outpath',type=str, default=None)

    args = parser.parse_args()

    ROOT = pathlib.Path(args.root)
    path = pathlib.Path('*')
    path = reduce(operator.truediv, (path for _ in range(args.depth)))

    unpackbits = True
    res = args.res
    min = -0.5
    max = 0.5

    p = Pool(mp.cpu_count())
    p.map(create_voxel_off, glob.glob( str(ROOT / path)))