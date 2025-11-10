# dsl_primitives.py
# Minimal, typed DSL for ARC-style grid programs — now with `dsl_` prefixes.
from typing import List, Tuple, Dict, Iterable, Optional
Grid = List[List[int]]

# ------------------------- utils -------------------------

def dsl_shape(G: Grid) -> Tuple[int, int]:
    return (len(G), len(G[0]) if G and G[0] is not None else 0)

def dsl_clone(G: Grid) -> Grid:
    return [row[:] for row in G]

def dsl_clamp(x: int) -> int:
    # ARC colors are 0..9
    if x < 0: return 0
    if x > 9: return 9
    return x

def dsl_zeros_like(G: Grid, val: int = 0) -> Grid:
    H, W = dsl_shape(G)
    return [[dsl_clamp(val) for _ in range(W)] for _ in range(H)]

def dsl_full(h: int, w: int, val: int = 0) -> Grid:
    return [[dsl_clamp(val) for _ in range(w)] for _ in range(h)]

def dsl_in_bounds(G: Grid, r: int, c: int) -> bool:
    H, W = dsl_shape(G)
    return 0 <= r < H and 0 <= c < W

def dsl_iter_coords(G: Grid) -> Iterable[Tuple[int, int]]:
    H, W = dsl_shape(G)
    for r in range(H):
        for c in range(W):
            yield r, c

# ------------------------- basic read/write -------------------------

def dsl_get_cell(G: Grid, r: int, c: int) -> int:
    return G[r][c] if dsl_in_bounds(G, r, c) else 0

def dsl_set_cell(G: Grid, r: int, c: int, val: int) -> Grid:
    if not dsl_in_bounds(G, r, c): return G
    G2 = dsl_clone(G)
    G2[r][c] = dsl_clamp(val)
    return G2

def dsl_paint_cell(G: Grid, r: int, c: int, color: int) -> Grid:
    return dsl_set_cell(G, r, c, color)

def dsl_replace_color(G: Grid, old: int, new: int) -> Grid:
    old, new = dsl_clamp(old), dsl_clamp(new)
    H, W = dsl_shape(G)
    G2 = dsl_clone(G)
    for r in range(H):
        for c in range(W):
            if G2[r][c] == old:
                G2[r][c] = new
    return G2

def dsl_remap_colors(G: Grid, mapping: Dict[int, int]) -> Grid:
    # Non-present keys ignored; values clamped.
    H, W = dsl_shape(G)
    G2 = dsl_clone(G)
    for r in range(H):
        for c in range(W):
            if G2[r][c] in mapping:
                G2[r][c] = dsl_clamp(mapping[G2[r][c]])
    return G2

# ------------------------- row/col ops -------------------------

def dsl_paint_row(G: Grid, r: int, color: int) -> Grid:
    H, W = dsl_shape(G)
    if not (0 <= r < H): return G
    color = dsl_clamp(color)
    G2 = dsl_clone(G)
    for c in range(W):
        G2[r][c] = color
    return G2

def dsl_paint_col(G: Grid, c: int, color: int) -> Grid:
    H, W = dsl_shape(G)
    if not (0 <= c < W): return G
    color = dsl_clamp(color)
    G2 = dsl_clone(G)
    for r in range(H):
        G2[r][c] = color
    return G2

def dsl_copy_row(G: Grid, r_src: int, r_dst: int) -> Grid:
    H, _ = dsl_shape(G)
    if not (0 <= r_src < H and 0 <= r_dst < H): return G
    G2 = dsl_clone(G)
    G2[r_dst] = G2[r_src][:]
    return G2

def dsl_copy_col(G: Grid, c_src: int, c_dst: int) -> Grid:
    _, W = dsl_shape(G)
    if not (0 <= c_src < W and 0 <= c_dst < W): return G
    G2 = dsl_clone(G)
    for r in range(len(G2)):
        G2[r][c_dst] = G2[r][c_src]
    return G2

# ------------------------- geometric ops -------------------------

def dsl_flip_h(G: Grid) -> Grid:
    return [list(reversed(row)) for row in G]

def dsl_flip_v(G: Grid) -> Grid:
    return list(reversed([row[:] for row in G]))

def dsl_transpose(G: Grid) -> Grid:
    H, W = dsl_shape(G)
    if H == 0 or W == 0: return []
    return [[G[r][c] for r in range(H)] for c in range(W)]

def dsl_rot90(G: Grid, k: int = 1) -> Grid:
    k = k % 4
    R = dsl_clone(G)
    for _ in range(k):
        R = dsl_transpose(dsl_flip_h(R))
    return R

# ------------------------- region ops -------------------------

def dsl_fill_rect(G: Grid, r0: int, c0: int, h: int, w: int, color: int) -> Grid:
    color = dsl_clamp(color)
    H, W = dsl_shape(G)
    if h <= 0 or w <= 0: return G
    G2 = dsl_clone(G)
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            if dsl_in_bounds(G2, r, c):
                G2[r][c] = color
    return G2

def dsl_paste(G: Grid, src: Grid, r0: int, c0: int) -> Grid:
    # Paste src onto G at (r0, c0), clipped.
    Hs, Ws = dsl_shape(src)
    G2 = dsl_clone(G)
    for r in range(Hs):
        for c in range(Ws):
            rr, cc = r0 + r, c0 + c
            if dsl_in_bounds(G2, rr, cc):
                G2[rr][cc] = dsl_clamp(src[r][c])
    return G2

def dsl_paste_masked(G: Grid, src: Grid, r0: int, c0: int, mask: Grid) -> Grid:
    # mask: 1 => copy, 0 => keep
    Hs, Ws = dsl_shape(src)
    G2 = dsl_clone(G)
    for r in range(Hs):
        for c in range(Ws):
            if r < len(mask) and c < len(mask[0]) and mask[r][c]:
                rr, cc = r0 + r, c0 + c
                if dsl_in_bounds(G2, rr, cc):
                    G2[rr][cc] = dsl_clamp(src[r][c])
    return G2

# ------------------------- masks & morphology -------------------------

def dsl_mask_eq(G: Grid, color: int) -> Grid:
    color = dsl_clamp(color)
    H, W = dsl_shape(G)
    return [[1 if G[r][c] == color else 0 for c in range(W)] for r in range(H)]

def dsl_apply_mask_color(G: Grid, mask: Grid, color: int) -> Grid:
    color = dsl_clamp(color)
    H, W = dsl_shape(G)
    G2 = dsl_clone(G)
    for r in range(min(H, len(mask))):
        rowm = mask[r]
        for c in range(min(W, len(rowm))):
            if rowm[c]:
                G2[r][c] = color
    return G2

def dsl_dilate(mask: Grid, radius: int = 1) -> Grid:
    H, W = dsl_shape(mask)
    out = [[0]*W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if mask[r][c]:
                for dr in range(-radius, radius+1):
                    for dc in range(-radius, radius+1):
                        rr, cc = r+dr, c+dc
                        if 0 <= rr < H and 0 <= cc < W:
                            out[rr][cc] = 1
    return out

def dsl_erode(mask: Grid, radius: int = 1) -> Grid:
    H, W = dsl_shape(mask)
    out = [[0]*W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            ok = True
            for dr in range(-radius, radius+1):
                for dc in range(-radius, radius+1):
                    rr, cc = r+dr, c+dc
                    if not (0 <= rr < H and 0 <= cc < W and mask[rr][cc]):
                        ok = False
                        break
                if not ok: break
            out[r][c] = 1 if ok else 0
    return out

# ------------------------- connected components -------------------------

def dsl_neighbors4(r: int, c: int) -> Iterable[Tuple[int,int]]:
    yield r-1, c
    yield r+1, c
    yield r, c-1
    yield r, c+1

def dsl_flood_fill(G: Grid, r: int, c: int, new_color: int) -> Grid:
    if not dsl_in_bounds(G, r, c): return G
    target = G[r][c]
    new_color = dsl_clamp(new_color)
    if target == new_color: return G
    H, W = dsl_shape(G)
    G2 = dsl_clone(G)
    stack = [(r, c)]
    while stack:
        rr, cc = stack.pop()
        if not dsl_in_bounds(G2, rr, cc): continue
        if G2[rr][cc] != target: continue
        G2[rr][cc] = new_color
        for nr, nc in dsl_neighbors4(rr, cc):
            if dsl_in_bounds(G2, nr, nc) and G2[nr][nc] == target:
                stack.append((nr, nc))
    return G2

def dsl_component_mask(G: Grid, r: int, c: int) -> Grid:
    # returns 0/1 mask of the 4-connected component at (r,c)
    if not dsl_in_bounds(G, r, c): return dsl_zeros_like(G, 0)
    target = G[r][c]
    H, W = dsl_shape(G)
    seen = [[0]*W for _ in range(H)]
    stack = [(r,c)]
    seen[r][c] = 1
    while stack:
        rr, cc = stack.pop()
        for nr, nc in dsl_neighbors4(rr, cc):
            if dsl_in_bounds(G, nr, nc) and not seen[nr][nc] and G[nr][nc] == target:
                seen[nr][nc] = 1
                stack.append((nr, nc))
    return seen

def dsl_bbox_of_mask(mask: Grid) -> Optional[Tuple[int,int,int,int]]:
    H, W = dsl_shape(mask)
    rmin, rmax, cmin, cmax = H, -1, W, -1
    for r in range(H):
        for c in range(W):
            if mask[r][c]:
                rmin = min(rmin, r); rmax = max(rmax, r)
                cmin = min(cmin, c); cmax = max(cmax, c)
    if rmax < rmin: return None
    return (rmin, cmin, rmax - rmin + 1, cmax - cmin + 1)

def dsl_crop(G: Grid, r0: int, c0: int, h: int, w: int) -> Grid:
    H, W = dsl_shape(G)
    out = []
    for r in range(r0, r0+h):
        row = []
        for c in range(c0, c0+w):
            if dsl_in_bounds(G, r, c):
                row.append(G[r][c])
        if row:
            out.append(row)
    return out

# ------------------------- compositional helpers -------------------------

def dsl_copy_cell_from(G: Grid, SRC: Grid, r: int, c: int) -> Grid:
    if not dsl_in_bounds(G, r, c): return G
    if not dsl_in_bounds(SRC, r, c): return G
    return dsl_set_cell(G, r, c, SRC[r][c])

def dsl_write_component(G: Grid, mask: Grid, SRC: Grid, r0: int, c0: int) -> Grid:
    # Write SRC (clipped) only where mask==1 (mask is in G's coords).
    Hs, Ws = dsl_shape(SRC)
    Hm, Wm = dsl_shape(mask)
    G2 = dsl_clone(G)
    for r in range(Hs):
        for c in range(Ws):
            rr, cc = r0 + r, c0 + c
            if dsl_in_bounds(G2, rr, cc) and rr < Hm and cc < Wm and mask[rr][cc]:
                G2[rr][cc] = dsl_clamp(SRC[r][c])
    return G2

# ------------------------- Backward-compatible aliases (optional) -------------------------
# If other parts of your code still call the old names, keep these.
shape = dsl_shape
clone = dsl_clone
zeros_like = dsl_zeros_like
full = dsl_full
clamp = dsl_clamp
in_bounds = dsl_in_bounds
iter_coords = dsl_iter_coords
get_cell = dsl_get_cell
set_cell = dsl_set_cell
paint_cell = dsl_paint_cell
replace_color = dsl_replace_color
remap_colors = dsl_remap_colors
paint_row = dsl_paint_row
paint_col = dsl_paint_col
copy_row = dsl_copy_row
copy_col = dsl_copy_col
flip_h = dsl_flip_h
flip_v = dsl_flip_v
transpose = dsl_transpose
rot90 = dsl_rot90
fill_rect = dsl_fill_rect
paste = dsl_paste
paste_masked = dsl_paste_masked
mask_eq = dsl_mask_eq
apply_mask_color = dsl_apply_mask_color
dilate = dsl_dilate
erode = dsl_erode
neighbors4 = dsl_neighbors4
flood_fill = dsl_flood_fill
component_mask = dsl_component_mask
bbox_of_mask = dsl_bbox_of_mask
crop = dsl_crop
copy_cell_from = dsl_copy_cell_from
write_component = dsl_write_component