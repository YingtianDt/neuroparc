import numpy as np
from neuroparc.surfaces import Surface
from neuroparc.atlas import Atlas
from neuroparc.extra.glasser import cortical_divisions

areas = [
    "Primary_Visual",
    "Early_Visual",
    "Ventral_Stream_Visual",
    "Dorsal_Stream_Visual",
    "MT+_Complex",
]

surface = Surface("fsaverage5")
atlas = Atlas("Glasser")
labels = atlas.label_surface("fsaverage5")

xyzs = surface.nodes
    
from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

masks = []
for area in areas:
    regions = cortical_divisions[area]
    region_labels = [atlas.rev_label_name_map[r] for r in regions]
    mask = np.isin(labels, region_labels)
    masks.append(mask)
non_visual_mask = ~np.any(masks, axis=0)

# for i, mask in enumerate(masks):
#     ax.scatter(xyzs[mask, 0], xyzs[mask, 1], xyzs[mask, 2], label=areas[i], alpha=1, s=4)

# ax.scatter(xyzs[non_visual_mask, 0], xyzs[non_visual_mask, 1], xyzs[non_visual_mask, 2], label="Others", alpha=0.1, s=2)

# ax.legend()

# # rotate the axes and update
# while True:
#     for angle in range(0, 360):
#         ax.view_init(30, angle)
#         plt.draw()
#         plt.pause(.001)

# write to animation

import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    for i, mask in enumerate(masks):
        ax.scatter(xyzs[mask, 0], xyzs[mask, 1], xyzs[mask, 2], label=areas[i], alpha=1, s=4)

    ax.scatter(xyzs[non_visual_mask, 0], xyzs[non_visual_mask, 1], xyzs[non_visual_mask, 2], label="Others", alpha=0.05, s=2)

    ax.legend()
    ax.view_init(30, frame)
    return ax

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 3), interval=50)

ani.save("visual_areas.gif", writer="ffmpeg")

