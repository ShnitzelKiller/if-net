import os

def get_name_and_paths(path, out_path=None):
    if os.path.isfile(path):
        name = os.path.splitext(os.path.split(path)[1])[0]
    else:
        name = 'boundary'
    
    if out_path is None:
        out_path = path
    
    if os.path.isdir(path):
        path = os.path.join(path, '/isosurf_scaled.off')

    return name, path, out_path