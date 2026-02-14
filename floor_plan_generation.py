"""
Synthetic Floor Plan and Navigation Graph Mask Generator

Generates paired floor plan images and navigation graph masks for UNet training.
Uses Binary Space Partitioning (BSP) for realistic room layouts with doors,
walls, and corridors.

Output:
    data/images/  - Grayscale floor plan images (black walls, white rooms, door openings)
    data/masks/   - Binary navigation graph masks (nodes at room/door centers, edges connecting them)
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Room:
    """A rectangular room produced by BSP partitioning."""

    def __init__(self, x, y, w, h, room_id=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.id = room_id

    @property
    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def is_corridor(self):
        aspect = max(self.w, self.h) / max(min(self.w, self.h), 1)
        return aspect > 3.0

    def interior(self, half_wall):
        """Return (x1, y1, x2, y2) of the room floor area after wall inset."""
        return (
            self.x + half_wall,
            self.y + half_wall,
            self.x + self.w - half_wall,
            self.y + self.h - half_wall,
        )


# ---------------------------------------------------------------------------
# BSP partitioning
# ---------------------------------------------------------------------------

def bsp_partition(x, y, w, h, min_room_size, max_depth, depth=0):
    """Recursively partition a rectangle into rooms via Binary Space Partitioning."""
    can_split_h = h >= min_room_size * 2
    can_split_v = w >= min_room_size * 2

    if depth >= max_depth or (not can_split_h and not can_split_v):
        return [Room(x, y, w, h)]

    # Pick split direction – prefer splitting the longer axis
    if can_split_h and can_split_v:
        if w > h * 1.3:
            split_h = False
        elif h > w * 1.3:
            split_h = True
        else:
            split_h = random.random() < 0.5
    elif can_split_h:
        split_h = True
    else:
        split_h = False

    # Split position: 35-65 % of the dimension for natural proportions
    ratio = random.uniform(0.35, 0.65)

    if split_h:
        split = int(h * ratio)
        split = max(min_room_size, min(split, h - min_room_size))
        upper = bsp_partition(x, y, w, split, min_room_size, max_depth, depth + 1)
        lower = bsp_partition(x, y + split, w, h - split, min_room_size, max_depth, depth + 1)
        return upper + lower
    else:
        split = int(w * ratio)
        split = max(min_room_size, min(split, w - min_room_size))
        left = bsp_partition(x, y, split, h, min_room_size, max_depth, depth + 1)
        right = bsp_partition(x + split, y, w - split, h, min_room_size, max_depth, depth + 1)
        return left + right


# ---------------------------------------------------------------------------
# Adjacency detection
# ---------------------------------------------------------------------------

def _shared_wall(r1, r2, min_overlap):
    """
    Return the shared wall segment between two rooms, or *None*.

    Returns
    -------
    tuple  (orientation, fixed_coord, seg_start, seg_end)
        orientation : 'h' (horizontal wall) or 'v' (vertical wall)
        fixed_coord : the y (for 'h') or x (for 'v') coordinate of the wall
        seg_start, seg_end : extent of the shared segment
    """
    tolerance = 2  # BSP tiles exactly, but allow tiny rounding slack

    # r1 bottom  <->  r2 top
    if abs((r1.y + r1.h) - r2.y) <= tolerance:
        xs = max(r1.x, r2.x)
        xe = min(r1.x + r1.w, r2.x + r2.w)
        if xe - xs >= min_overlap:
            return ('h', r1.y + r1.h, xs, xe)

    # r2 bottom  <->  r1 top
    if abs((r2.y + r2.h) - r1.y) <= tolerance:
        xs = max(r1.x, r2.x)
        xe = min(r1.x + r1.w, r2.x + r2.w)
        if xe - xs >= min_overlap:
            return ('h', r2.y + r2.h, xs, xe)

    # r1 right  <->  r2 left
    if abs((r1.x + r1.w) - r2.x) <= tolerance:
        ys = max(r1.y, r2.y)
        ye = min(r1.y + r1.h, r2.y + r2.h)
        if ye - ys >= min_overlap:
            return ('v', r1.x + r1.w, ys, ye)

    # r2 right  <->  r1 left
    if abs((r2.x + r2.w) - r1.x) <= tolerance:
        ys = max(r1.y, r2.y)
        ye = min(r1.y + r1.h, r2.y + r2.h)
        if ye - ys >= min_overlap:
            return ('v', r2.x + r2.w, ys, ye)

    return None


def find_adjacencies(rooms, door_width):
    """Return a list of (room_i, room_j, shared_wall_info) for adjacent room pairs."""
    min_overlap = door_width + 10
    adjacencies = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            shared = _shared_wall(rooms[i], rooms[j], min_overlap)
            if shared is not None:
                adjacencies.append((i, j, shared))
    return adjacencies


# ---------------------------------------------------------------------------
# Door placement (with guaranteed connectivity via spanning tree)
# ---------------------------------------------------------------------------

def place_doors(rooms, adjacencies, door_width, half_wall, extra_door_prob=0.3):
    """
    Place doors between adjacent rooms.

    A BFS spanning tree guarantees every room is reachable.  Additional doors
    are added randomly for variety (loops in the navigation graph).
    """
    n = len(rooms)
    if n <= 1:
        return []

    # Build adjacency look-up
    adj_by_room = {i: [] for i in range(n)}
    for idx, (i, j, _) in enumerate(adjacencies):
        adj_by_room[i].append((j, idx))
        adj_by_room[j].append((i, idx))

    # BFS spanning tree
    visited = {0}
    queue = [0]
    tree_edges = set()
    while queue:
        cur = queue.pop(0)
        for nbr, adj_idx in adj_by_room[cur]:
            if nbr not in visited:
                visited.add(nbr)
                tree_edges.add(adj_idx)
                queue.append(nbr)

    doors = []
    for idx, (i, j, shared) in enumerate(adjacencies):
        if idx not in tree_edges and random.random() >= extra_door_prob:
            continue  # skip non-tree edge

        orientation, fixed, seg_start, seg_end = shared
        margin = 8
        avail_start = seg_start + margin
        avail_end = seg_end - door_width - margin

        if avail_start >= avail_end:
            door_pos = (seg_start + seg_end - door_width) // 2
        else:
            door_pos = random.randint(avail_start, avail_end)

        if orientation == 'h':
            door = {
                'x': door_pos,
                'y': fixed - half_wall - 1,
                'w': door_width,
                'h': half_wall * 2 + 3,
                'center': (door_pos + door_width // 2, fixed),
                'room1': i,
                'room2': j,
            }
        else:
            door = {
                'x': fixed - half_wall - 1,
                'y': door_pos,
                'w': half_wall * 2 + 3,
                'h': door_width,
                'center': (fixed, door_pos + door_width // 2),
                'room1': i,
                'room2': j,
            }
        doors.append(door)

    return doors


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_floor_plan(width, height, rooms, doors, half_wall):
    """Draw the floor plan: black canvas, white room interiors, door openings."""
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)

    for room in rooms:
        x1, y1, x2, y2 = room.interior(half_wall)
        if x2 > x1 and y2 > y1:
            draw.rectangle([x1, y1, x2, y2], fill=255)

    for door in doors:
        draw.rectangle(
            [door['x'], door['y'],
             door['x'] + door['w'], door['y'] + door['h']],
            fill=255,
        )

    return img


def render_mask(width, height, rooms, doors, node_radius=8, edge_width=4):
    """Draw the navigation graph: nodes at room / door centres, edges between them."""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Edges: room-centre  →  door-centre  →  room-centre
    for door in doors:
        c1 = rooms[door['room1']].center
        c2 = rooms[door['room2']].center
        dc = door['center']
        draw.line([c1, dc], fill=255, width=edge_width)
        draw.line([dc, c2], fill=255, width=edge_width)

    # Door nodes (slightly smaller)
    dr = max(node_radius - 1, 1)
    for door in doors:
        cx, cy = door['center']
        draw.ellipse([cx - dr, cy - dr, cx + dr, cy + dr], fill=255)

    # Room-centre nodes
    nr = node_radius
    for room in rooms:
        cx, cy = room.center
        draw.ellipse([cx - nr, cy - nr, cx + nr, cy + nr], fill=255)

    return mask


def render_overlay(floor_plan, mask):
    """Create an RGB visualisation: floor plan in grey, nav-graph in green."""
    fp_arr = np.array(floor_plan)
    m_arr = np.array(mask)

    rgb = np.stack([fp_arr, fp_arr, fp_arr], axis=-1)  # greyscale → RGB

    graph_pixels = m_arr > 0
    rgb[graph_pixels, 0] = np.clip(rgb[graph_pixels, 0].astype(int) - 60, 0, 255)
    rgb[graph_pixels, 1] = 255
    rgb[graph_pixels, 2] = np.clip(rgb[graph_pixels, 2].astype(int) - 60, 0, 255)

    return Image.fromarray(rgb)


# ---------------------------------------------------------------------------
# Generator class (bundles parameters + randomisation)
# ---------------------------------------------------------------------------

class FloorPlanGenerator:
    """High-level generator: one call → (floor_plan, mask) pair."""

    def __init__(
        self,
        width=512,
        height=512,
        wall_thickness=4,
        outer_wall_thickness=6,
        min_room_size=60,
        max_depth=4,
        door_width=20,
        extra_door_prob=0.3,
        node_radius=4,
        edge_width=2,
    ):
        self.width = width
        self.height = height
        self.wall_thickness = wall_thickness
        self.outer_wall_thickness = outer_wall_thickness
        self.min_room_size = min_room_size
        self.max_depth = max_depth
        self.door_width = door_width
        self.extra_door_prob = extra_door_prob
        self.node_radius = node_radius
        self.edge_width = edge_width

    def generate(self, seed=None):
        """Return (floor_plan_image, mask_image, rooms, doors)."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**31))

        # Per-sample randomisation for variety
        wall_t = max(2, self.wall_thickness + random.choice([-1, 0, 0, 1]))
        half_wall = wall_t // 2
        door_w = max(12, self.door_width + random.randint(-4, 6))
        max_d = max(2, self.max_depth + random.choice([-1, 0, 0, 1]))
        min_rs = max(40, self.min_room_size + random.randint(-10, 20))

        # BSP area is inset from canvas edge so the outer wall is thicker
        margin = max(1, self.outer_wall_thickness - half_wall)
        bsp_x, bsp_y = margin, margin
        bsp_w = self.width - 2 * margin
        bsp_h = self.height - 2 * margin

        # Partition into rooms
        rooms = bsp_partition(bsp_x, bsp_y, bsp_w, bsp_h, min_rs, max_d)
        for idx, r in enumerate(rooms):
            r.id = idx

        # Adjacency + doors
        adjacencies = find_adjacencies(rooms, door_w)
        doors = place_doors(rooms, adjacencies, door_w, half_wall, self.extra_door_prob)

        # Render
        fp = render_floor_plan(self.width, self.height, rooms, doors, half_wall)
        msk = render_mask(self.width, self.height, rooms, doors,
                          self.node_radius, self.edge_width)

        return fp, msk, rooms, doors


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic floor plan / navigation-graph mask pairs',
    )
    parser.add_argument('-n', '--num-samples', type=int, default=100,
                        help='Number of pairs to generate (default: 100)')
    parser.add_argument('-o', '--output-dir', type=str, default='data',
                        help='Output root directory (default: data)')
    parser.add_argument('--width', type=int, default=512,
                        help='Image width in pixels (default: 512)')
    parser.add_argument('--height', type=int, default=512,
                        help='Image height in pixels (default: 512)')
    parser.add_argument('--wall-thickness', type=int, default=4,
                        help='Interior wall thickness (default: 4)')
    parser.add_argument('--min-room-size', type=int, default=60,
                        help='Minimum room dimension in pixels (default: 60)')
    parser.add_argument('--max-depth', type=int, default=4,
                        help='Max BSP recursion depth (default: 4)')
    parser.add_argument('--door-width', type=int, default=20,
                        help='Door opening width in pixels (default: 20)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Master random seed for reproducibility')
    parser.add_argument('--preview', type=int, default=5,
                        help='Number of overlay previews to save (default: 5)')
    args = parser.parse_args()

    out = Path(args.output_dir)
    img_dir = out / 'images'
    mask_dir = out / 'masks'
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    if args.preview > 0:
        preview_dir = out / 'previews'
        preview_dir.mkdir(parents=True, exist_ok=True)

    gen = FloorPlanGenerator(
        width=args.width,
        height=args.height,
        wall_thickness=args.wall_thickness,
        min_room_size=args.min_room_size,
        max_depth=args.max_depth,
        door_width=args.door_width,
    )

    # Deterministic per-sample seeds when a master seed is given
    if args.seed is not None:
        master = random.Random(args.seed)
        seeds = [master.randint(0, 2**31) for _ in range(args.num_samples)]
    else:
        seeds = [None] * args.num_samples

    print(f'Generating {args.num_samples} floor-plan / mask pairs ...')

    room_counts = []
    for i, seed_i in enumerate(seeds):
        fp, msk, rooms, doors = gen.generate(seed=seed_i)

        fname = f'floor_plan_{i:04d}.png'
        fp.save(img_dir / fname)
        msk.save(mask_dir / fname)
        room_counts.append(len(rooms))

        if i < args.preview:
            overlay = render_overlay(fp, msk)
            overlay.save(out / 'previews' / fname)

        if (i + 1) % max(1, args.num_samples // 10) == 0 or i == 0:
            print(f'  [{i + 1:>{len(str(args.num_samples))}}/{args.num_samples}]  '
                  f'rooms={len(rooms)}, doors={len(doors)}')

    print(f'\nDone!  rooms/plan: min={min(room_counts)}, '
          f'max={max(room_counts)}, avg={np.mean(room_counts):.1f}')
    print(f'  Images   -> {img_dir.resolve()}')
    print(f'  Masks    -> {mask_dir.resolve()}')
    if args.preview > 0:
        print(f'  Previews -> {(out / "previews").resolve()}')


if __name__ == '__main__':
    main()
