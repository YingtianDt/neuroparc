import os
import json
import nibabel as nib
import numpy as np

from .utils import memorized
from .surfaces import Surface


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(MODULE_DIR, os.path.pardir, 'atlases/label/Human')
META_DIR = os.path.join(LABEL_DIR, 'Metadata-json')
LABEL_NAME_DIR = os.path.join(LABEL_DIR, 'Anatomical-labels-csv')
CACHE_DIR = os.path.join(MODULE_DIR, 'cache')


def get_meta(volumn_name):
    json_path = os.path.join(META_DIR, volumn_name + '.json')
    with open(json_path) as f:
        meta = json.load(f)
    return meta

def get_label_volumn(volumn_name):
    nii_path = os.path.join(LABEL_DIR, volumn_name + '.nii.gz')
    return nib.load(nii_path)

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
    def __init__(self, name, space=None, resolution="1x1x1"):
        self.name = name

        if space is None:
            space = "space-MNI152NLin6"

        resolution = f"res-{resolution}"

        self.volumn_name = f"{name}_{space}_{resolution}"

    @classmethod
    def get_atlas_names(cls):
        return [f.split('_')[0] for f in os.listdir(LABEL_DIR) if f.endswith('.nii.gz')]

    @property
    @memorized
    def meta(self):
        return get_meta(self.volumn_name)
    
    @property
    @memorized
    def label_volumn(self):
        return get_label_volumn(self.volumn_name)
    
    def label_surface(self, surface_name, search_range=3):  # mm
        surface = Surface(surface_name)
        ijks = self.get_voxels(surface.nodes)

        # convert search range from mm to voxels
        if not self.voxel_sizes[0] == self.voxel_sizes[1] == self.voxel_sizes[2]:
            print('CAUTION: Voxel sizes are not equal in all dimensions, check ijk in the label_surface function again.')
        search_range = search_range / self.voxel_sizes
        search_range = np.ceil(search_range).astype(int)

        # search for valid labels around the nodes
        x_offset = range(-search_range[0], search_range[0])
        y_offset = range(-search_range[1], search_range[1])
        z_offset = range(-search_range[2], search_range[2])
        labels = []
        from itertools import product
        for i, j, k in product(x_offset, y_offset, z_offset):
            labels.append(self.label_volumn.get_fdata()[ijks[:, 0]+i, ijks[:, 1]+j, ijks[:, 2]+k].astype(int))

        labels = np.array(labels)
        # majority vote
        def majority_vote(x):
            if (x==0).all():
                return 0
            return np.bincount(x[x!=0]).argmax()
        labels = np.apply_along_axis(majority_vote, 0, labels)
        return labels

    @property
    def rev_label_name_map(self):
        return {v: k for k, v in self.label_name_map.items()}
    
    @property
    @memorized
    def label_name_map(self):
        return get_label_name_map(self.name)
    
    @staticmethod
    def is_standard_reference_space(nb_img):
        # scanner XYZ space: mm
        if nb_img.shape == (182, 218, 182):
            expected_affine = np.array([[-1., 0., 0., 90.],
                                        [0., 1., 0., -126.],
                                        [0., 0., 1., -72.],
                                        [0., 0., 0., 1.]])
            return np.allclose(nb_img.affine, expected_affine)
        
        elif nb_img.shape == (91, 109, 91):
            expected_affine = np.array([[-2., 0., 0., 90.],
                                        [0., 2., 0., -126.],
                                        [0., 0., 2., -72.],
                                        [0., 0., 0., 1.]])
            return np.allclose(nb_img.affine, expected_affine)
        
        elif nb_img.shape == (45, 54, 45):
            expected_affine = np.array([[-4., 0., 0., 88.],
                                        [0., 4., 0., -124.],
                                        [0., 0., 4., -70.],
                                        [0., 0., 0., 1.]])
            return np.allclose(nb_img.affine, expected_affine)
        
        else:
            return False
        
    def get_voxels(self, xyzs):
        xyzs_1 = np.concatenate([xyzs, np.ones((xyzs.shape[0], 1))], axis=1)
        ijks = np.rint(np.linalg.inv(self.label_volumn.affine).dot(xyzs_1.T)).astype(int).T
        ijks = ijks[:, :3]
        return ijks
    
    @property
    @memorized
    def voxel_sizes(self):
        return np.abs(np.diag(self.label_volumn.affine)[:3])
    
    def search_region(self, keyword):
        keyword = keyword.lower()
        for k, v in self.label_name_map.items():
            if keyword in v.lower():
                print(k, v)

if __name__ == '__main__':
    atlas = Atlas('Glasser', resolution='4x4x4')
    print(atlas.meta)
    print(atlas.label_name_map)
    print(atlas.label)
    print(atlas.label.get_fdata().shape)
    print(atlas.is_standard_reference_space(atlas.label))