import os
from omegaconf import OmegaConf


def update_paths(base_path, paths):
    """
    Recursively update all paths in the dictionary by prepending the base_path.
    """
    if isinstance(paths, dict):
        for key, value in paths.items():
            paths[key] = update_paths(base_path, value)
    elif isinstance(paths, str):
        paths = os.path.join(base_path, paths)
    return paths


def getconfig(configpath):
    args = OmegaConf.load(configpath)

    workspace = os.path.dirname(configpath)
    args.filepaths = update_paths(workspace, args.filepaths)

    return args
