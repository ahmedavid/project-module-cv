#!/usr/bin/env python3
"""
generate_floorplans.py

Generate simple synthetic floor plan images (black & white) using BSP-style subdivision.

Features:
 - Rectangular rooms only.
 - Random number of rooms per image (configurable min/max).
 - Doors drawn explicitly as small gaps along shared walls.
 - Every split creates at least one door -> ensures full connectivity.
 - Labels placed in the center of each room: R1, R2, ...
 - Save PNG images to an output directory.

Usage:
    python generate_floorplans.py
    # To customize via CLI:
    python generate_floorplans.py --n_images 100 --min_rooms 6 --max_rooms 14 --outdir out/

Author: ChatGPT (GPT-5 Thinking mini)
"""

import os
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Room:
    x: float  # left
    y: float  # bottom
    w: float  # width
    h: float  # height
    id: int = -1

    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def right(self) -> float:
        return self.x + self.w

    def top(self) -> float:
        return self.y + self.h

# door: ((x1,y1), (x2,y2)) coordinates in plan space
Door = Tuple[Tuple[float, float], Tuple[float, float], int, int]  # (p1, p2, room_id_a, room_id_b)

# ----------------------------
# BSP partitioning
# ----------------------------
def split_room(room: Room, min_size: float, rand=0.2) -> Tuple[Room, Room, bool]:
    """
    Split room either vertically or horizontally.
    Return (room_a, room_b, split_was_possible)
    rand controls how close to center the split tends to be.
    """
    horizontal = random.choice([True, False])
    # ensure we choose a direction that is splittable
    can_h = room.h >= 2 * min_size
    can_v = room.w >= 2 * min_size
    if not can_h and not can_v:
        return (room, None, False)
    if can_h and not can_v:
        horizontal = True
    elif can_v and not can_h:
        horizontal = False

    if horizontal:
        # split along y into bottom and top
        min_split = room.y + min_size
        max_split = room.top() - min_size
        if min_split >= max_split:
            return (room, None, False)
        split_y = random.uniform(min_split, max_split)
        a = Room(room.x, room.y, room.w, split_y - room.y)
        b = Room(room.x, split_y, room.w, room.top() - split_y)
        return (a, b, True)
    else:
        # vertical split into left and right
        min_split = room.x + min_size
        max_split = room.right() - min_size
        if min_split >= max_split:
            return (room, None, False)
        split_x = random.uniform(min_split, max_split)
        a = Room(room.x, room.y, split_x - room.x, room.h)
        b = Room(split_x, room.y, room.right() - split_x, room.h)
        return (a, b, True)

# ----------------------------
# Adjacency / door creation
# ----------------------------
EPS = 1e-6

def shared_edge(room_a: Room, room_b: Room):
    """
    If rooms share an edge (non-zero overlap along the edge), return:
     - orientation: 'v' or 'h'
     - coordinate info: (fixed_coord, interval_start, interval_end)
       for vertical edge fixed_coord is x value, interval is [y0,y1]
       for horizontal edge fixed_coord is y value, interval is [x0,x1]
     - side: which side of a is shared ('right' or 'left' or 'top' or 'bottom')
    Otherwise return None.
    """
    # check vertical adjacency (a right == b left or a left == b right)
    if abs(room_a.right() - room_b.x) < 1e-6 or abs(room_b.right() - room_a.x) < 1e-6:
        # overlapping y-interval?
        y0 = max(room_a.y, room_b.y)
        y1 = min(room_a.top(), room_b.top())
        if y1 - y0 > 0:
            # shared vertical edge at x = ...
            if abs(room_a.right() - room_b.x) < 1e-6:
                fixed_x = room_a.right()
                side = 'a_right_b_left'
            else:
                fixed_x = room_b.right()
                side = 'b_right_a_left'
            return ('v', fixed_x, y0, y1, side)
    # check horizontal adjacency
    if abs(room_a.top() - room_b.y) < 1e-6 or abs(room_b.top() - room_a.y) < 1e-6:
        x0 = max(room_a.x, room_b.x)
        x1 = min(room_a.right(), room_b.right())
        if x1 - x0 > 0:
            if abs(room_a.top() - room_b.y) < 1e-6:
                fixed_y = room_a.top()
                side = 'a_top_b_bottom'
            else:
                fixed_y = room_b.top()
                side = 'b_top_a_bottom'
            return ('h', fixed_y, x0, x1, side)
    return None

def make_door_on_shared_edge(shared_info, door_length_frac=0.25):
    """
    Given shared edge info from shared_edge, choose a random door segment (two points).
    door_length_frac is fraction of shared edge length for door length (max).
    Returns ((x1,y1),(x2,y2))
    """
    orient = shared_info[0]
    if orient == 'v':
        _, x, y0, y1, _ = shared_info
        length = y1 - y0
        door_len = min(length * door_length_frac, length * 0.8)
        # choose center along [y0+door_len/2, y1-door_len/2]
        if length <= 0:
            return None
        c = random.uniform(y0 + door_len/2.0, y1 - door_len/2.0)
        return ((x, c - door_len/2.0), (x, c + door_len/2.0))
    else:
        _, y, x0, x1, _ = shared_info
        length = x1 - x0
        door_len = min(length * door_length_frac, length * 0.8)
        if length <= 0:
            return None
        c = random.uniform(x0 + door_len/2.0, x1 - door_len/2.0)
        return ((c - door_len/2.0, y), (c + door_len/2.0, y))

# ----------------------------
# Renderer
# ----------------------------
def render_floorplan(rooms: List[Room], doors: List[Door], filename: str, image_size=512, linewidth=3.0):
    """
    Render black & white floorplan image with room labels centered.
    Rooms: list of Room with room.id set
    Doors: list of doors ((x1,y1),(x2,y2),id_a,id_b)
    """
    fig, ax = plt.subplots(figsize=(6,6), dpi=96)
    ax.set_aspect('equal')
    ax.axis('off')

    # Determine bounding box
    minx = min(r.x for r in rooms)
    miny = min(r.y for r in rooms)
    maxx = max(r.right() for r in rooms)
    maxy = max(r.top() for r in rooms)

    # Scale to fit a square drawing area with padding
    pad = 0.02 * max(maxx-minx, maxy-miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)

    # Draw walls: draw full rectangle edges for each room BUT skip door segments
    # We'll collect line segments and subtract door spans when drawing.
    # For simplicity, draw each room's 4 edges, but for each edge, cut out any doors that lie on that edge.
    for r in rooms:
        edges = []
        # left edge: (x, y) -> (x, y+h)
        edges.append(((r.x, r.y), (r.x, r.top())))
        # right
        edges.append(((r.right(), r.y), (r.right(), r.top())))
        # bottom
        edges.append(((r.x, r.y), (r.right(), r.y)))
        # top
        edges.append(((r.x, r.top()), (r.right(), r.top())))

        for e in edges:
            ex0, ey0 = e[0]
            ex1, ey1 = e[1]
            # gather door segments that sit on this edge (with tolerance)
            edge_segs = [((ex0,ey0),(ex1,ey1))]
            new_segs = []
            for seg in edge_segs:
                segs_to_process = [seg]
                for d in doors:
                    (dx0,dy0),(dx1,dy1),da,db = d
                    # Check collinear & overlapping with a small tolerance
                    if abs(ex0 - ex1) < 1e-9:
                        # vertical edge at x = ex0; check door x's approx equals that
                        if abs(dx0 - ex0) < 1e-6 and abs(dx1 - ex0) < 1e-6:
                            # door interval along y: [dy0,dy1]
                            door_y0, door_y1 = sorted([dy0,dy1])
                            # current segment
                            s0y = min(seg[0][1], seg[1][1])
                            s1y = max(seg[0][1], seg[1][1])
                            # subtract door interval from segment (cut out)
                            pieces = subtract_interval((s0y,s1y),(door_y0,door_y1))
                            # convert back to segments at x coordinate
                            segs_to_process = []
                            for (a,b) in pieces:
                                segs_to_process.append(((ex0,a),(ex0,b)))
                        else:
                            # door not on this edge
                            pass
                    else:
                        # horizontal edge
                        if abs(dy0 - ey0) < 1e-6 and abs(dy1 - ey0) < 1e-6:
                            door_x0, door_x1 = sorted([dx0,dx1])
                            s0x = min(seg[0][0], seg[1][0])
                            s1x = max(seg[0][0], seg[1][0])
                            pieces = subtract_interval((s0x,s1x),(door_x0,door_x1))
                            segs_to_process = []
                            for (a,b) in pieces:
                                segs_to_process.append(((a,ey0),(b,ey0)))
                        else:
                            pass
                new_segs.extend(segs_to_process)
            # draw new_segs as wall parts (black lines)
            for s in new_segs:
                (sx0,sy0),(sx1,sy1) = s
                ax.plot([sx0,sx1],[sy0,sy1], color='black', linewidth=linewidth, solid_capstyle='butt')

    # Draw doors as small ticks (optional) - but since walls have gaps, doors appear as gaps.
    # Add room labels
    for r in rooms:
        cx, cy = r.center()
        ax.text(cx, cy, f"R{r.id}", ha='center', va='center', fontsize=8, color='black')

    # Save to file
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
    fig.set_size_inches(6,6)
    plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def subtract_interval(seg, hole):
    """
    Subtract hole interval from seg interval on a line.
    seg = (s0,s1), hole = (h0,h1)
    Returns list of remaining intervals (possibly 0,1 or 2 segments).
    """
    s0, s1 = seg
    h0, h1 = hole
    # ensure order
    if h1 <= s0 + EPS or h0 >= s1 - EPS:
        # no overlap
        return [(s0,s1)]
    pieces = []
    # left piece
    if h0 > s0 + EPS:
        pieces.append((s0, max(s0, min(h0, s1))))
    # right piece
    if h1 < s1 - EPS:
        pieces.append((max(s0, min(h1, s1)), s1))
    return pieces

# ----------------------------
# Main generator per plan
# ----------------------------
def generate_one_floorplan(bbox=(0.0,0.0,1.0,1.0),
                           min_rooms=6, max_rooms=14,
                           min_room_size_frac=0.08,
                           extra_door_prob=0.1,
                           door_length_frac=0.25,
                           seed=None):
    """
    Generate a single floorplan.
    bbox: (x,y,w,h) bounding box; default unit square.
    Returns: rooms (List[Room]), doors (List[Door])
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    bx, by, bw, bh = bbox
    # minimum room size absolute based on bbox smaller dimension
    min_dimension = min(bw, bh)
    min_room_size = min_room_size_frac * min_dimension

    # choose random target number of rooms
    target_rooms = random.randint(min_rooms, max_rooms)

    # start with single room occupying bbox
    initial = Room(bx, by, bw, bh, id=1)
    rooms: List[Room] = [initial]
    doors: List[Door] = []

    next_id = 1

    # iterative splitting: always split the largest room until we reach target_rooms or cannot split
    while len(rooms) < target_rooms:
        # pick the room with largest area that is splittable
        splittable = []
        for r in rooms:
            if r.w >= 2 * min_room_size - EPS or r.h >= 2 * min_room_size - EPS:
                splittable.append(r)
        if not splittable:
            break
        # choose the largest area splittable room
        r = max(splittable, key=lambda rr: rr.w * rr.h)
        # attempt split
        a, b, ok = split_room(r, min_room_size)
        if not ok or b is None:
            # cannot split this one; to avoid infinite loop remove it from consideration
            # mark it as unsplittable by shrinking its min dimension (skip it)
            # but simpler: break if no split possible
            break
        # replace r with a and b
        rooms.remove(r)
        next_id += 1
        a.id = next_id - 1
        next_id += 1
        b.id = next_id - 1
        rooms.append(a)
        rooms.append(b)

        # create a door along the split edge between a and b
        shared = shared_edge(a, b)
        if shared is None:
            # should not happen, they were produced by split
            pass
        else:
            seg = make_door_on_shared_edge(shared, door_length_frac=door_length_frac)
            if seg:
                # door references room IDs
                doors.append((seg[0], seg[1], a.id, b.id))

    # Now rooms list has final rooms. Ensure ids are compact and monotonically assigned from 1..N
    # Reassign ids in order
    rooms = sorted(rooms, key=lambda rr: (rr.x, rr.y))
    for i, r in enumerate(rooms, start=1):
        r.id = i
    # update doors' room ids mapping (they referenced a.id/b.id earlier but ids may have changed).
    # Build mapping from bounding boxes to ids (approx)
    def room_key(r: Room):
        return (round(r.x,6), round(r.y,6), round(r.w,6), round(r.h,6))
    room_map = {room_key(r): r.id for r in rooms}

    # Rebuild doors by finding the rooms that contain the door endpoints (or that are adjacent)
    cleaned_doors: List[Door] = []
    for d in doors:
        (p0, p1, old_a, old_b) = d
        # find room A and B by adjacency: among rooms, find two that share the door line
        found_a = None
        found_b = None
        for r in rooms:
            # a door segment lies on edge of a room if both endpoints are on room boundary
            if point_on_room_edge(p0, r) and point_on_room_edge(p1, r):
                if found_a is None:
                    found_a = r.id
                elif found_b is None and r.id != found_a:
                    found_b = r.id
        if found_a is None or found_b is None:
            # fallback: find rooms that are nearest to endpoints
            fa = find_room_containing_point(p0, rooms)
            fb = find_room_containing_point(p1, rooms)
            if fa is not None and fb is not None and fa != fb:
                cleaned_doors.append((p0,p1,fa,fb))
        else:
            cleaned_doors.append((p0,p1,found_a,found_b))

    # Optionally add extra random doors between adjacent rooms
    # Find all adjacent pairs and with some probability add a door if not exists
    adj_pairs = []
    n = len(rooms)
    for i in range(n):
        for j in range(i+1, n):
            info = shared_edge(rooms[i], rooms[j])
            if info:
                adj_pairs.append((i,j,info))
    # existing adjacency set
    exists = set()
    for d in cleaned_doors:
        a,b = d[2], d[3]
        key = tuple(sorted((a,b)))
        exists.add(key)

    for i,j,info in adj_pairs:
        a_id = rooms[i].id
        b_id = rooms[j].id
        key = tuple(sorted((a_id,b_id)))
        if key in exists:
            continue
        if random.random() < extra_door_prob:
            seg = make_door_on_shared_edge(info, door_length_frac=door_length_frac)
            if seg:
                cleaned_doors.append((seg[0], seg[1], a_id, b_id))
                exists.add(key)

    # final rooms and doors
    return rooms, cleaned_doors

# ----------------------------
# Helper geometry utilities
# ----------------------------
def point_on_room_edge(p, r: Room, tol=1e-6):
    x,y = p
    on_left = abs(x - r.x) < tol and (r.y - tol <= y <= r.top() + tol)
    on_right = abs(x - r.right()) < tol and (r.y - tol <= y <= r.top() + tol)
    on_bottom = abs(y - r.y) < tol and (r.x - tol <= x <= r.right() + tol)
    on_top = abs(y - r.top()) < tol and (r.x - tol <= x <= r.right() + tol)
    return on_left or on_right or on_bottom or on_top

def find_room_containing_point(p, rooms: List[Room], tol=1e-9):
    x,y = p
    for r in rooms:
        if (r.x - tol <= x <= r.right() + tol) and (r.y - tol <= y <= r.top() + tol):
            return r.id
    return None

# ----------------------------
# CLI / Batch generation
# ----------------------------
def generate_dataset(outdir="out",
                     n_images=100,
                     bbox=(0.0,0.0,500.0,500.0),
                     min_rooms=6,
                     max_rooms=14,
                     min_room_size_frac=0.06,
                     extra_door_prob=0.12,
                     door_length_frac=0.25,
                     image_size=512,
                     seed_base=None):
    os.makedirs(outdir, exist_ok=True)
    for i in range(1, n_images + 1):
        seed = None
        if seed_base is not None:
            seed = seed_base + i
        rooms, doors = generate_one_floorplan(bbox=bbox,
                                              min_rooms=min_rooms,
                                              max_rooms=max_rooms,
                                              min_room_size_frac=min_room_size_frac,
                                              extra_door_prob=extra_door_prob,
                                              door_length_frac=door_length_frac,
                                              seed=seed)
        # fname = os.path.join(outdir, f"floorplan_{i:04d}.png")
        # render_floorplan(rooms, doors, fname, image_size=image_size, linewidth=3.0)
        base = f"{i:04d}"
        img_path = os.path.join(outdir, f"floorplan_{base}.png")
        mask_path = os.path.join(outdir, f"mask_{base}.png")

        render_floorplan(rooms, doors, img_path, image_size=image_size, linewidth=3.0)
        render_mask(rooms, mask_path, image_size=image_size)
        
        # if i % 10 == 0 or i == 1:
        #     print(f"Saved {fname} with {len(rooms)} rooms and {len(doors)} doors.")
    print("Done. Generated", n_images, "floorplans in", outdir)

def parse_args():
    p = argparse.ArgumentParser(description="Generate simple rectangular floorplans (B/W PNGs).")
    p.add_argument("--n_images", type=int, default=100, help="Number of images to generate (default 100).")
    p.add_argument("--outdir", type=str, default="out", help="Output directory.")
    p.add_argument("--min_rooms", type=int, default=6, help="Minimum rooms per image.")
    p.add_argument("--max_rooms", type=int, default=14, help="Maximum rooms per image.")
    p.add_argument("--bbox_w", type=float, default=500.0, help="Bounding box width.")
    p.add_argument("--bbox_h", type=float, default=500.0, help="Bounding box height.")
    p.add_argument("--seed", type=int, default=None, help="Base seed (optional).")
    return p.parse_args()

def render_mask(rooms, filename, image_size=512):
    """
    Render grayscale segmentation mask for the given rooms.
    Each room gets a unique integer ID.
    Background = 0
    """
    minx = min(r.x for r in rooms)
    miny = min(r.y for r in rooms)
    maxx = max(r.right() for r in rooms)
    maxy = max(r.top() for r in rooms)

    W, H = image_size, image_size
    mask = np.zeros((H, W), dtype=np.uint8)

    for r in rooms:
        # map room coordinates to pixel grid
        x0 = int((r.x - minx) / (maxx - minx) * (W - 1))
        x1 = int((r.right() - minx) / (maxx - minx) * (W - 1))
        y0 = int((r.y - miny) / (maxy - miny) * (H - 1))
        y1 = int((r.top() - miny) / (maxy - miny) * (H - 1))
        mask[y0:y1, x0:x1] = r.id  # fill rectangle region with room ID

    import imageio.v2 as imageio
    imageio.imwrite(filename, mask)


if __name__ == "__main__":
    args = parse_args()
    bbox = (0.0, 0.0, args.bbox_w, args.bbox_h)
    print("Generating floorplans...", "n_images=", args.n_images)
    generate_dataset(outdir=args.outdir,
                     n_images=args.n_images,
                     bbox=bbox,
                     min_rooms=args.min_rooms,
                     max_rooms=args.max_rooms,
                     seed_base=args.seed)
