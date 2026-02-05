import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cv2
import networkx as nx
from scipy import ndimage

# -----------------------------
# Parameters
# -----------------------------
MASK_PATH = "data/mask_0005.png"   # update this path
DOOR_KERNEL = np.ones((3, 3), np.uint8)

# -----------------------------
# Load mask
# -----------------------------
mask = imageio.imread(MASK_PATH).astype(np.uint8)
H, W = mask.shape

# -----------------------------
# Identify unique rooms
# -----------------------------
room_ids = np.unique(mask)
room_ids = room_ids[room_ids > 0]
print(f"Detected {len(room_ids)} rooms")

# -----------------------------
# Detect door regions
# -----------------------------
# Doors occur where neighboring pixels belong to different rooms.
grad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, DOOR_KERNEL)

# We'll scan gradient pixels and record which two room IDs meet there.
door_pairs = set()

for y in range(1, H - 1):
    for x in range(1, W - 1):
        if grad[y, x] > 0:
            neigh = mask[y - 1:y + 2, x - 1:x + 2].flatten()
            rooms = np.unique(neigh[neigh > 0])
            if len(rooms) == 2:
                a, b = sorted(rooms)
                door_pairs.add((int(a), int(b)))

# -----------------------------
# Build graph
# -----------------------------
G = nx.Graph()
for rid in room_ids:
    ys, xs = np.where(mask == rid)
    cy, cx = ys.mean(), xs.mean()
    G.add_node(int(rid), centroid=(cx, cy))

for a, b in door_pairs:
    G.add_edge(a, b)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# -----------------------------
# Visualization
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(mask, cmap='tab20')

for (a, b) in G.edges():
    ax.plot(
        [G.nodes[a]['centroid'][0], G.nodes[b]['centroid'][0]],
        [G.nodes[a]['centroid'][1], G.nodes[b]['centroid'][1]],
        'k-', linewidth=1.5
    )

for i, data in G.nodes(data=True):
    x, y = data['centroid']
    ax.plot(x, y, 'ro')
    ax.text(x + 3, y, f"R{i}", color='red', fontsize=8)

ax.set_title("Room Graph (Edges via Shared Doors Only)")
ax.axis('off')
plt.tight_layout()
plt.show()
