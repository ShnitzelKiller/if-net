import trimesh
import numpy as np
import implicit_waterproofing as iw
import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
import traceback
import pathlib
from functools import reduce
import operator
from data_processing.util import get_name_and_paths


def boundary_sampling(path):
    try:
        infilename, off_path, outpath = get_name_and_paths(path, args.outpath)

        out_file = os.path.join(outpath, f'{infilename}_{args.sigma}_samples.npz')
        if os.path.exists(out_file) and not args.replace:
            return
        

        mesh = trimesh.load(off_path)
        points = mesh.sample(sample_num)

        boundary_points = points + args.sigma * np.random.randn(sample_num, 3)

        occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

        if args.scale != 1:
            boundary_points /= args.scale
        
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

        grid_coords = 2 * grid_coords

        np.savez(out_file, points=boundary_points, occupancies = occupancies, grid_coords= grid_coords)
        print('Finished {}'.format(path))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--outpath',type=str, default=None)
    parser.add_argument('--root', type=str, default='shapenet/data')
    parser.add_argument('--depth',type=int, default=3)
    parser.add_argument('--scale', type=float, default=1, help='inverse scale (input object is assumed to be factor of `scale` w.r.t. normalized coordinate system')
    parser.add_argument('--replace', action='store_true')

    args = parser.parse_args()
    root = args.root
    path = pathlib.Path('*')
    path = reduce(operator.truediv, (path for _ in range(args.depth)))

    sample_num = 100000


    p = Pool(mp.cpu_count())
    p.map(boundary_sampling, glob.glob( str(root / path)))
