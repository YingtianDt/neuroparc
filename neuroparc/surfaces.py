import numpy as np
from nilearn import datasets
from nilearn import surface

from .utils import memorized

class Surface:
    def __init__(self, name):
        self._surf = datasets.fetch_surf_fsaverage(name)
        self._left = surface.load_surf_data(self._surf['pial_left'])
        self._right = surface.load_surf_data(self._surf['pial_right'])
        self._data = []
        self._data.append(np.concatenate([self._left[0], self._right[0]]))
        self._data.append(np.concatenate([self._left[1], self._right[1] + len(self._left[0])]))

    @property
    def nodes(self):
        return self._data[0]
    
    @property
    @memorized
    def faces(self):
        faces = []
        for vertex_indices in self._data[1]:
            vertices = self.nodes[vertex_indices]
            faces.append(vertices)
        return np.array(faces)  # (n_faces, 3:vertices, 3:xyz)
    
    @property
    def face_centroids(self):
        return self.faces.mean(axis=1)
    

if __name__ == "__main__":
    surface = Surface("fsaverage5")
    print(surface.nodes.shape)
    print(surface.faces.shape)
    print(surface.face_centroids.shape)
    print(surface.face_centroids[:5])
