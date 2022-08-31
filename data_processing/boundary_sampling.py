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

def boundary_sampling(path):
    try:

        if os.path.isfile(path):
            infilename = os.path.splitext(os.path.split(path)[1])[0]
        else:
            infilename = 'boundary'
        
        outpath = args.outpath
        if outpath is None:
            outpath = path

        out_file = os.path.join(outpath, f'{infilename}_{args.sigma}_samples.npz')
        if os.path.exists(out_file):
            return
        

        if os.path.isdir(path):
            path = os.path.join(path, '/isosurf_scaled.off')
        
        off_path = path

        mesh = trimesh.load(off_path)
        points = mesh.sample(sample_num)

        boundary_points = points + args.sigma * np.random.randn(sample_num, 3)
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

        grid_coords = 2 * grid_coords

        occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

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

    args = parser.parse_args()
    root = args.root
    path = pathlib.Path('*')
    path = reduce(operator.truediv, (path for _ in range(args.depth)))

    sample_num = 100000


    p = Pool(mp.cpu_count())
    p.map(boundary_sampling, glob.glob( str(root / path)))
