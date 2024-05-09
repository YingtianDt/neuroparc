import numpy as np
from neuroparc.atlas import Atlas
from neuroparc.surfaces import Surface

all_atlases = Atlas.get_atlas_names()
# print(all_atlases)

surface = Surface("fsaverage5")
altas = Atlas("Glasser")

search_range = 3  # mm

from matplotlib import pyplot as plt

# plot the nodes from the surface on the altas space in a 3D plot

ijks = altas.get_voxels(surface.nodes)

# convert search range from mm to voxels
search_range = search_range / altas.voxel_sizes
search_range = np.ceil(search_range).astype(int)

# search for valid labels around the nodes
x_offset = range(-search_range[0], search_range[0])
y_offset = range(-search_range[1], search_range[1])
z_offset = range(-search_range[2], search_range[2])

labels = []
from itertools import product
for i, j, k in product(x_offset, y_offset, z_offset):
    labels.append(altas.label_volumn.get_fdata()[ijks[:, 0]+i, ijks[:, 1]+j, ijks[:, 2]+k].astype(int))


labels = np.array(labels)
# majority vote
def majority_vote(x):
    if (x==0).all():
        return 0
    return np.bincount(x[x!=0]).argmax()
labels = np.apply_along_axis(majority_vote, 0, labels)
# breakpoint()
# assert (labels != 0).all()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(ijks[:, 0], ijks[:, 1], ijks[:, 2], c=labels, cmap='tab20')

# filter out the nodes with label > 0
mask = labels == 0
ax.scatter(ijks[mask, 0], ijks[mask, 1], ijks[mask, 2], c=labels[mask], cmap='tab20')
ax.scatter(ijks[~mask, 0], ijks[~mask, 1], ijks[~mask, 2], c=labels[~mask], cmap='tab20', alpha=0.01)

# scale the axes by the voxel size
voxel_sizes = altas.voxel_sizes
ax.set_xlim(ijks[:, 0].min()*voxel_sizes[0], ijks[:, 0].max()*voxel_sizes[0])
ax.set_ylim(ijks[:, 1].min()*voxel_sizes[1], ijks[:, 1].max()*voxel_sizes[1])
ax.set_zlim(ijks[:, 2].min()*voxel_sizes[2], ijks[:, 2].max()*voxel_sizes[2])

# rotate the axes and update
while True:
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)



# # plot the nodes from the surface and the original volumn on the altas space in a 3D plot

# ijks = altas.get_voxels(surface.nodes)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(ijks[:, 0], ijks[:, 1], ijks[:, 2], c='r', label='surface nodes', marker='o', s=3)
# volumn = altas.label_volumn.get_fdata()
# x, y, z = np.where(volumn)
# # plot x, y, z with label
# ax.scatter(x, y, z, c=volumn[x, y, z], cmap='tab20', label='volumn', marker='x', alpha=0.01, s=5)

# print(x.max(), x.min(), y.max(), y.min(), z.max(), z.min())
# print(ijks[:, 0].max(), ijks[:, 0].min(), ijks[:, 1].max(), ijks[:, 1].min(), ijks[:, 2].max(), ijks[:, 2].min())
# # rotate the axes and update
# for angle in range(-360, 0):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)