import os
import numpy as np
from nilearn import surface
from nilearn import datasets
from .utils import ANNOTATION_HOME


def load_annotation(name):
    if name == "Glasser":
        return glasser()
    elif name == "NSD-Streams":
        return nsd_streams()
    elif name == "Destrieux":
        return destrieux()
    else:
        raise ValueError(f"Unknown annotation {name}")
    
def glasser():
    lh = surface.load_surf_data(os.path.join(ANNOTATION_HOME, 'lh.HCP-MMP1.annot'))
    rh = surface.load_surf_data(os.path.join(ANNOTATION_HOME, 'rh.HCP-MMP1.annot'))
    surf = np.concatenate([lh, rh])
    return surf

def nsd_streams():
    lh = np.load(os.path.join(ANNOTATION_HOME, 'lh.streams_fsaverage_space.npy'))
    rh = np.load(os.path.join(ANNOTATION_HOME, 'rh.streams_fsaverage_space.npy'))
    surf = np.concatenate([lh, rh])
    return surf

def destrieux():
    data = datasets.fetch_atlas_surf_destrieux()
    lh = data['map_left']
    rh = data['map_right']
    surf = np.concatenate([lh, rh])
    return surf