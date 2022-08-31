import trimesh
import numpy as np
import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial, reduce
import traceback
import voxels
import argparse
import pathlib
import operator


def voxelize(in_path, res, outpath=None):
    try:
        if os.path.isfile(in_path):
            infilename = os.path.splitext(os.path.split(in_path)[1])[0]
        else:
            infilename = 'voxelization'
        
        if outpath is None:
            outpath = in_path
        filename = os.path.join(outpath, f'{infilename}_{res}.npy')

        if os.path.exists(filename):
            return

        if os.path.isdir(in_path):
            in_path = os.path.join(in_path, '/isosurf_scaled.off')

        mesh = trimesh.load(in_path, process=False)
        occupancies = voxels.VoxelGrid.from_mesh(mesh, res, loc=[0, 0, 0], scale=1).data
        occupancies = np.reshape(occupancies, -1)

        if not occupancies.any():
            raise ValueError('No empty voxel grids allowed.')

        occupancies = np.packbits(occupancies)
        np.save(filename, occupancies)

    except Exception as err:
        path = os.path.normpath(in_path)
        print('Error with {}: {}'.format(path, traceback.format_exc()))
    print('finished {}'.format(in_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run voxalization'
    )
    parser.add_argument('--res', type=int)
    parser.add_argument('--root', type=str, default='shapenet/data')
    parser.add_argument('--depth',type=int, default=3)
    parser.add_argument('--outpath',type=str, default=None)

    args = parser.parse_args()

    ROOT = pathlib.Path(args.root)

    path = pathlib.Path('*')
    path = reduce(operator.truediv, (path for _ in range(args.depth)))

    p = Pool(mp.cpu_count())
    p.map(partial(voxelize, res=args.res, outpath=args.outpath), glob.glob( str(ROOT / path)))
