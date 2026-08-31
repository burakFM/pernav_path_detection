import math

import numpy as np


def filter_fov(
    xy: np.ndarray,
    x_min: float = 0.0,
    x_max: float = 20.0,
    y_min: float = -10.0,
    y_max: float = 10.0,
) -> np.ndarray:
    """Crop XY points to a rectangular FOV."""
    if xy.size == 0:
        return xy

    mask = (
        (xy[:, 0] >= x_min) & (xy[:, 0] <= x_max) &
        (xy[:, 1] >= y_min) & (xy[:, 1] <= y_max)
    )
    return xy[mask]


def remove_xy_box(
    xy: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Remove points inside an XY rectangle, including its boundaries."""
    if xy.size == 0:
        return xy

    inside_box = (
        (xy[:, 0] >= x_min) & (xy[:, 0] <= x_max) &
        (xy[:, 1] >= y_min) & (xy[:, 1] <= y_max)
    )
    return xy[~inside_box]


def ransac_line_2d_continuous(
    points: np.ndarray,
    distance_threshold: float,
    max_gap: float,
    max_iterations: int,
    min_segment_inliers: int,
) -> tuple[tuple[float, float, float] | None, np.ndarray | None]:
    """Detect one dominant continuous 2D line segment using random sampling."""
    n_points = len(points)
    if n_points < 2:
        return None, None

    rng = np.random.default_rng(0)
    best_line = None
    best_segment_mask = None
    best_score = -1.0

    for _ in range(max_iterations):
        idx1, idx2 = rng.choice(n_points, size=2, replace=False)
        p1, p2 = points[idx1], points[idx2]

        direction = p2 - p1
        direction_length = float(np.linalg.norm(direction))
        if direction_length < 1e-8:
            continue

        unit_direction = direction / direction_length

        a = float(direction[1])
        b = float(-direction[0])
        c = float(-(a * p1[0] + b * p1[1]))
        normal_length = math.sqrt(a * a + b * b)

        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c) / normal_length
        initial_inlier_indices = np.where(distances < distance_threshold)[0]
        if len(initial_inlier_indices) < min_segment_inliers:
            continue

        inlier_points = points[initial_inlier_indices]
        projections = inlier_points @ unit_direction
        order = np.argsort(projections)
        sorted_indices = initial_inlier_indices[order]
        sorted_projections = projections[order]

        gaps = np.diff(sorted_projections)
        split_positions = np.where(gaps > max_gap)[0] + 1
        segments = np.split(sorted_indices, split_positions)
        largest_segment = max(segments, key=len)
        if len(largest_segment) < min_segment_inliers:
            continue

        segment_points = points[largest_segment]
        segment_projections = segment_points @ unit_direction
        segment_length = float(segment_projections.max() - segment_projections.min())

        score = float(len(largest_segment)) + segment_length
        if score > best_score:
            best_score = score
            best_line = (a, b, c)
            best_segment_mask = np.zeros(n_points, dtype=bool)
            best_segment_mask[largest_segment] = True

    return best_line, best_segment_mask


def detect_rows_count(
    xy: np.ndarray,
    distance_threshold: float,
    max_gap: float,
    max_iterations: int,
    min_segment_inliers: int,
    max_rows: int,
    remove_radius: float,
    min_points_left: int,
) -> int:
    """Iteratively detect and remove row-like line segments; return detected row count."""
    row_records = detect_rows_from_xy(
        xy,
        distance_threshold=distance_threshold,
        max_gap=max_gap,
        max_iterations=max_iterations,
        min_segment_inliers=min_segment_inliers,
        max_rows=max_rows,
        remove_radius=remove_radius,
        start_ref_point=(3.0, 0.0),
        min_points_left=min_points_left,
    )
    return len(row_records)


def detect_rows_from_xy(
    xy: np.ndarray,
    distance_threshold: float,
    max_gap: float,
    max_iterations: int,
    min_segment_inliers: int,
    max_rows: int,
    remove_radius: float,
    start_ref_point: tuple[float, float],
    min_points_left: int,
) -> list[dict]:
    """Detect row segments and return notebook-style row metadata records."""
    if xy.size == 0:
        return []

    working_points = xy.copy()
    row_records: list[dict] = []
    ref_point = np.array(start_ref_point, dtype=float)

    for _ in range(max_rows):
        line_k, inlier_mask_k = ransac_line_2d_continuous(
            working_points,
            distance_threshold=distance_threshold,
            max_gap=max_gap,
            max_iterations=max_iterations,
            min_segment_inliers=min_segment_inliers,
        )

        if line_k is None or inlier_mask_k is None:
            break

        a_k, b_k, c_k = line_k
        seg_points = working_points[inlier_mask_k]
        if seg_points.shape[0] < min_segment_inliers:
            break

        line_dir = np.array([-b_k, a_k], dtype=float)
        line_dir /= np.linalg.norm(line_dir)
        line_org = -c_k * np.array([a_k, b_k], dtype=float) / (a_k**2 + b_k**2)

        projections = (seg_points - line_org) @ line_dir
        t_min, t_max = np.percentile(projections, [2, 98])
        p_start = line_org + t_min * line_dir
        p_end = line_org + t_max * line_dir

        if np.linalg.norm(p_start - ref_point) > np.linalg.norm(p_end - ref_point):
            p_start, p_end = p_end, p_start

        row_records.append(
            {
                'row_id': len(row_records) + 1,
                'A': float(a_k),
                'B': float(b_k),
                'C': float(c_k),
                'inlier_count': int(np.sum(inlier_mask_k)),
                'start_x': float(p_start[0]),
                'start_y': float(p_start[1]),
                'end_x': float(p_end[0]),
                'end_y': float(p_end[1]),
            }
        )

        line_norm = math.sqrt(a_k * a_k + b_k * b_k)
        distances = np.abs(a_k * working_points[:, 0] + b_k * working_points[:, 1] + c_k) / line_norm
        keep_mask = distances >= remove_radius
        working_points = working_points[keep_mask]

        if working_points.shape[0] < min_points_left:
            break

    return row_records


def build_paths_from_rows(
    row_records: list[dict],
    path_width: float,
    min_path_length: float = 0.0,
) -> list[dict]:
    """Build path metadata from row pairs whose starts are closer than path_width."""
    if not row_records:
        return []

    path_records: list[dict] = []

    for i in range(len(row_records)):
        for j in range(i + 1, len(row_records)):
            row_i = row_records[i]
            row_j = row_records[j]

            start_i = np.array([row_i['start_x'], row_i['start_y']], dtype=float)
            start_j = np.array([row_j['start_x'], row_j['start_y']], dtype=float)
            row_spacing = float(np.linalg.norm(start_i - start_j))

            if row_spacing >= path_width:
                continue

            end_i = np.array([row_i['end_x'], row_i['end_y']], dtype=float)
            end_j = np.array([row_j['end_x'], row_j['end_y']], dtype=float)

            p_start = 0.5 * (start_i + start_j)
            p_end = 0.5 * (end_i + end_j)
            path_len = float(np.linalg.norm(p_end - p_start))

            if path_len < min_path_length:
                continue

            path_records.append(
                {
                    'path_id': len(path_records) + 1,
                    'row_a_id': int(row_i['row_id']),
                    'row_b_id': int(row_j['row_id']),
                    'row_a_start_x': float(start_i[0]),
                    'row_a_start_y': float(start_i[1]),
                    'row_b_start_x': float(start_j[0]),
                    'row_b_start_y': float(start_j[1]),
                    'path_start_x': float(p_start[0]),
                    'path_start_y': float(p_start[1]),
                    'path_end_x': float(p_end[0]),
                    'path_end_y': float(p_end[1]),
                    'path_length': path_len,
                    'row_spacing': row_spacing,
                }
            )

    return path_records


def line_angle_deg(sx: float, sy: float, ex: float, ey: float) -> float:
    return math.degrees(math.atan2(ey - sy, ex - sx))


def wrap_angle_diff_deg(a_deg: float, b_deg: float) -> float:
    d = abs(a_deg - b_deg) % 180.0
    return min(d, 180.0 - d)


def path_features(path_row: dict) -> dict | None:
    sx, sy = float(path_row['path_start_x']), float(path_row['path_start_y'])
    ex, ey = float(path_row['path_end_x']), float(path_row['path_end_y'])
    mx, my = 0.5 * (sx + ex), 0.5 * (sy + ey)
    dx, dy = ex - sx, ey - sy
    length = float(np.hypot(dx, dy))
    if length < 1e-9:
        return None

    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    ang = line_angle_deg(sx, sy, ex, ey)
    return {
        'sx': sx,
        'sy': sy,
        'ex': ex,
        'ey': ey,
        'mx': mx,
        'my': my,
        'ux': ux,
        'uy': uy,
        'nx': nx,
        'ny': ny,
        'angle_deg': ang,
        'length': length,
    }


def make_parallel_with_dir(path_row: dict, ux: float, uy: float) -> tuple[float, float, float, float]:
    mid_x = 0.5 * (float(path_row['path_start_x']) + float(path_row['path_end_x']))
    mid_y = 0.5 * (float(path_row['path_start_y']) + float(path_row['path_end_y']))
    half_len = 0.5 * float(path_row['path_length'])
    return (
        mid_x - ux * half_len,
        mid_y - uy * half_len,
        mid_x + ux * half_len,
        mid_y + uy * half_len,
    )


def build_path_groups(
    path_records: list[dict],
    angle_thresh_deg: float,
    midpoint_thresh: float,
    lateral_thresh: float,
) -> list[list[int]]:
    """Build connected groups of paths based on orientation and spatial proximity."""
    if not path_records:
        return []

    features = [path_features(path_records[i]) for i in range(len(path_records))]
    n = len(features)
    adj = {i: set() for i in range(n)}

    for i in range(n):
        fi = features[i]
        if fi is None:
            continue
        for j in range(i + 1, n):
            fj = features[j]
            if fj is None:
                continue

            angle_diff = wrap_angle_diff_deg(fi['angle_deg'], fj['angle_deg'])
            if angle_diff > angle_thresh_deg:
                continue

            dmx, dmy = fj['mx'] - fi['mx'], fj['my'] - fi['my']
            midpoint_dist = float(np.hypot(dmx, dmy))
            if midpoint_dist > midpoint_thresh:
                continue

            navg_x = fi['nx'] + fj['nx']
            navg_y = fi['ny'] + fj['ny']
            navg_len = float(np.hypot(navg_x, navg_y))
            if navg_len < 1e-9:
                navg_x, navg_y, navg_len = fi['nx'], fi['ny'], 1.0
            navg_x, navg_y = navg_x / navg_len, navg_y / navg_len

            lateral_gap = abs(dmx * navg_x + dmy * navg_y)
            if lateral_gap > lateral_thresh:
                continue

            adj[i].add(j)
            adj[j].add(i)

    visited = set()
    groups = []
    for i in range(n):
        if i in visited:
            continue
        stack = [i]
        comp = []
        visited.add(i)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        groups.append(sorted(comp))

    return groups


def apply_parallel_correction(
    path_records: list[dict],
    groups: list[list[int]],
) -> list[dict]:
    """For each group, keep longest path as reference and align others parallel to it."""
    if not path_records:
        return []

    corrected_records: list[dict] = []

    for gidx, comp in enumerate(groups, start=1):
        group_paths = [path_records[k] for k in comp]
        ref_path = max(group_paths, key=lambda p: float(p.get('path_length', 0.0)))
        ref_pid = int(ref_path['path_id'])

        dx_ref = float(ref_path['path_end_x']) - float(ref_path['path_start_x'])
        dy_ref = float(ref_path['path_end_y']) - float(ref_path['path_start_y'])
        len_ref = float(np.hypot(dx_ref, dy_ref))
        if len_ref < 1e-9:
            continue
        ux_ref, uy_ref = dx_ref / len_ref, dy_ref / len_ref

        for row in group_paths:
            pid = int(row['path_id'])
            if pid == ref_pid:
                sx = float(row['path_start_x'])
                sy = float(row['path_start_y'])
                ex = float(row['path_end_x'])
                ey = float(row['path_end_y'])
                is_reference = True
            else:
                sx, sy, ex, ey = make_parallel_with_dir(row, ux_ref, uy_ref)
                is_reference = False

            corrected_records.append(
                {
                    'group_id': gidx,
                    'ref_path_id': ref_pid,
                    'path_id': pid,
                    'row_a_id': int(row['row_a_id']),
                    'row_b_id': int(row['row_b_id']),
                    'path_start_x': float(sx),
                    'path_start_y': float(sy),
                    'path_end_x': float(ex),
                    'path_end_y': float(ey),
                    'path_length': float(np.hypot(ex - sx, ey - sy)),
                    'row_spacing': float(row['row_spacing']),
                    'is_reference': is_reference,
                }
            )

    return corrected_records


def build_group_start_lines(
    corrected_path_records: list[dict],
    extension_m: float = 1.0,
) -> list[dict]:
    """Build best-fit start lines per group from corrected path start points."""
    if not corrected_path_records:
        return []

    grouped: dict[int, list[tuple[float, float]]] = {}
    for rec in corrected_path_records:
        gid = int(rec.get('group_id', -1))
        if gid < 0:
            continue
        grouped.setdefault(gid, []).append((float(rec['path_start_x']), float(rec['path_start_y'])))

    line_records: list[dict] = []
    for gid in sorted(grouped.keys()):
        start_points = np.asarray(grouped[gid], dtype=float)
        if start_points.ndim != 2 or start_points.shape[0] < 2:
            continue

        centroid = start_points.mean(axis=0)
        centered = start_points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        direction_len = float(np.hypot(direction[0], direction[1]))
        if direction_len < 1e-9:
            continue
        direction = direction / direction_len

        projections = centered @ direction
        p0 = centroid + (projections.min() - extension_m) * direction
        p1 = centroid + (projections.max() + extension_m) * direction

        line_records.append(
            {
                'group_id': gid,
                'start_x': float(p0[0]),
                'start_y': float(p0[1]),
                'end_x': float(p1[0]),
                'end_y': float(p1[1]),
            }
        )

    return line_records
