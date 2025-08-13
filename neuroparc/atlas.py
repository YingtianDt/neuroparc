import os
import json
import nibabel as nib
import numpy as np

from .utils import memorized
from .surfaces import Surface
from .annotations import load_annotation


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(MODULE_DIR, os.path.pardir, 'atlases/label/Human')
META_DIR = os.path.join(LABEL_DIR, 'Metadata-json')
LABEL_NAME_DIR = os.path.join(LABEL_DIR, 'Anatomical-labels-csv')
CACHE_DIR = os.path.join(MODULE_DIR, 'cache')


def get_label_name_map(atlas_name):
    csv_path = os.path.join(LABEL_NAME_DIR, atlas_name + '.csv')
    with open(csv_path) as f:
        lines = f.readlines()
    label_name_map = {}
    for line in lines:
        label_id, label_name = line.strip().split(',')
        label_name_map[int(label_id)] = label_name
    return label_name_map


class Atlas:
    def __init__(self, name, annotation=None):
        self.name = name
        annotation = annotation if annotation is not None else load_annotation(name)
        self.annotation = np.array(annotation)

    @property
    @memorized
    def original_surface(self):
        num_surf_node = len(self.annotation)
        if num_surf_node == 642 * 2:
            return "fsaverage3"
        elif num_surf_node == 2562 * 2:
            return "fsaverage4"
        elif num_surf_node == 10242 * 2:
            return "fsaverage5"
        elif num_surf_node == 40962 * 2:
            return "fsaverage6"
        elif num_surf_node == 163842 * 2:
            return "fsaverage7"
        else:
            raise ValueError("Unknown surface size {}".format(num_surf_node))

    @classmethod
    def get_atlas_names(cls):
        return [f.split('_')[0] for f in os.listdir(LABEL_DIR) if f.endswith('.nii.gz')]
    
    def label_surface(self, surface_name, knn=10):
        if surface_name == self.original_surface:
            return self.annotation

        other_surf = Surface(surface_name)
        this_surf = Surface(self.original_surface)
        this_xyz = this_surf.nodes
        other_xyz = other_surf.nodes

        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=knn)
        nn.fit(this_xyz)
        _, indices = nn.kneighbors(other_xyz)
        labels = self.annotation[indices]
        if labels.dtype == float:
            return np.array([np.median(labels[i]) for i in range(len(labels))])
        return np.array([np.argmax(np.bincount(labels[i])) for i in range(len(labels))])

    @property
    def rev_label_name_map(self):
        return {v: k for k, v in self.label_name_map.items()}
    
    @property
    @memorized
    def label_name_map(self):
        return get_label_name_map(self.name)
    
    def search_region(self, keyword):
        keyword = keyword.lower()
        for k, v in self.label_name_map.items():
            if keyword in v.lower():
                print(k, v)
