import math

import numpy as np


def trimmed_pca_line_2d(
    points: np.ndarray,
    keep_fraction: float,
    min_segment_inliers: int,
) -> tuple[tuple[float, float, float] | None, np.ndarray | None, float, float]:
    """Fit a deterministic line after retaining points closest to an initial PCA line."""
    n_points = len(points)
    if n_points < min_segment_inliers:
        return None, None, -1.0, 0.0

    initial_origin = np.mean(points, axis=0)
    _, _, right_singular_vectors = np.linalg.svd(
        points - initial_origin,
        full_matrices=False,
    )
    initial_direction = right_singular_vectors[0]
    direction_norm = float(np.linalg.norm(initial_direction))
    if direction_norm <= 1e-9:
        return None, None, -1.0, 0.0
    initial_direction /= direction_norm
    initial_normal = np.array([-initial_direction[1], initial_direction[0]], dtype=float)
    residuals = np.abs((points - initial_origin) @ initial_normal)

    keep_count = max(min_segment_inliers, int(math.ceil(n_points * keep_fraction)))
    keep_count = min(keep_count, n_points)
    selected_indices = np.argpartition(residuals, keep_count - 1)[:keep_count]
    selected_points = points[selected_indices]

    selected_origin = np.mean(selected_points, axis=0)
    _, _, right_singular_vectors = np.linalg.svd(
        selected_points - selected_origin,
        full_matrices=False,
    )
    selected_direction = right_singular_vectors[0]
    direction_norm = float(np.linalg.norm(selected_direction))
    if direction_norm <= 1e-9:
        return None, None, -1.0, 0.0
    selected_direction /= direction_norm
    selected_normal = np.array([-selected_direction[1], selected_direction[0]], dtype=float)
    line = (
        float(selected_normal[0]),
        float(selected_normal[1]),
        float(-(selected_normal @ selected_origin)),
    )
    projections = (selected_points - selected_origin) @ selected_direction
    segment_length = float(projections.max() - projections.min())
    selected_mask = np.zeros(n_points, dtype=bool)
    selected_mask[selected_indices] = True
    return line, selected_mask, float(keep_count) + segment_length, segment_length


def detect_rows_count(
    xy: np.ndarray,
    min_segment_inliers: int,
    max_rows: int,
    remove_radius: float,
    min_points_left: int,
    eor_line_reference: tuple[float, float],
    eor_line_direction: tuple[float, float],
    trimmed_pca_keep_fraction: float = 0.9,
) -> int:
    """Iteratively detect and remove row-like line segments; return detected row count."""
    row_records = detect_rows_from_xy(
        xy,
        min_segment_inliers=min_segment_inliers,
        max_rows=max_rows,
        remove_radius=remove_radius,
        min_points_left=min_points_left,
        eor_line_reference=eor_line_reference,
        eor_line_direction=eor_line_direction,
        trimmed_pca_keep_fraction=trimmed_pca_keep_fraction,
    )
    return len(row_records)


def detect_rows_from_xy(
    xy: np.ndarray,
    min_segment_inliers: int,
    max_rows: int,
    remove_radius: float,
    min_points_left: int,
    eor_line_reference: tuple[float, float],
    eor_line_direction: tuple[float, float],
    trimmed_pca_keep_fraction: float = 0.9,
) -> list[dict]:
    """Detect row segments oriented with their starts nearest the EOR line."""
    if xy.size == 0:
        return []

    working_points = xy.copy()
    row_records: list[dict] = []
    eor_reference = np.asarray(eor_line_reference, dtype=float)
    eor_direction = np.asarray(eor_line_direction, dtype=float)
    direction_length = float(np.linalg.norm(eor_direction))
    if (
        eor_reference.shape != (2,)
        or eor_direction.shape != (2,)
        or not np.isfinite(eor_reference).all()
        or not np.isfinite(eor_direction).all()
        or direction_length <= 1e-9
    ):
        return []
    eor_direction /= direction_length
    eor_normal = np.array([-eor_direction[1], eor_direction[0]], dtype=float)
    if not 0.0 < trimmed_pca_keep_fraction <= 1.0:
        raise ValueError('trimmed_pca_keep_fraction must be in (0, 1].')

    for _ in range(max_rows):
        line_k, inlier_mask_k, fit_score, fit_segment_length = trimmed_pca_line_2d(
            working_points,
            keep_fraction=trimmed_pca_keep_fraction,
            min_segment_inliers=min_segment_inliers,
        )

        if line_k is None or inlier_mask_k is None:
            break

        a_k, b_k, _ = line_k
        seg_points = working_points[inlier_mask_k]
        if seg_points.shape[0] < min_segment_inliers:
            break

        # Fit the selected points with total least squares (PCA) so the
        # published row uses all retained points.
        line_org = np.mean(seg_points, axis=0)
        _, _, right_singular_vectors = np.linalg.svd(
            seg_points - line_org,
            full_matrices=False,
        )
        line_dir = right_singular_vectors[0]
        line_dir_norm = float(np.linalg.norm(line_dir))
        if line_dir_norm <= 1e-9:
            break
        line_dir /= line_dir_norm

        # Keep a stable orientation relative to the initial fit proposal.
        proposal_dir = np.array([-b_k, a_k], dtype=float)
        proposal_dir /= np.linalg.norm(proposal_dir)
        if float(line_dir @ proposal_dir) < 0.0:
            line_dir = -line_dir

        line_normal = np.array([-line_dir[1], line_dir[0]], dtype=float)
        a_k, b_k = float(line_normal[0]), float(line_normal[1])
        c_k = float(-(line_normal @ line_org))

        projections = (seg_points - line_org) @ line_dir
        t_min, t_max = np.percentile(projections, [2, 98])
        p_start = line_org + t_min * line_dir
        p_end = line_org + t_max * line_dir

        start_distance = abs(float((p_start - eor_reference) @ eor_normal))
        end_distance = abs(float((p_end - eor_reference) @ eor_normal))
        if start_distance > end_distance:
            p_start, p_end = p_end, p_start

        row_records.append(
            {
                'row_id': len(row_records) + 1,
                'A': float(a_k),
                'B': float(b_k),
                'C': float(c_k),
                'inlier_count': int(np.sum(inlier_mask_k)),
                'fit_input_point_count': int(working_points.shape[0]),
                'fit_inlier_ratio': float(np.mean(inlier_mask_k)),
                'fit_score': float(fit_score),
                'fit_segment_length': float(fit_segment_length),
                'start_x': float(p_start[0]),
                'start_y': float(p_start[1]),
                'end_x': float(p_end[0]),
                'end_y': float(p_end[1]),
            }
        )

        distances = np.abs((working_points - line_org) @ line_normal)
        keep_mask = distances >= remove_radius
        working_points = working_points[keep_mask]

        if working_points.shape[0] < min_points_left:
            break

    return row_records


def align_rows_to_reference(row_records: list[dict]) -> list[dict]:
    """Align row endpoints to the highest-inlier row without moving their starts."""
    if len(row_records) < 2:
        return [dict(row) for row in row_records]

    reference_row = max(row_records, key=lambda row: int(row.get('inlier_count', 0)))
    reference_start = np.array(
        [reference_row['start_x'], reference_row['start_y']],
        dtype=float,
    )
    reference_end = np.array(
        [reference_row['end_x'], reference_row['end_y']],
        dtype=float,
    )
    reference_direction = reference_end - reference_start
    reference_length = float(np.linalg.norm(reference_direction))
    if reference_length <= 1e-9:
        return [dict(row) for row in row_records]
    reference_direction /= reference_length
    reference_normal = np.array(
        [-reference_direction[1], reference_direction[0]],
        dtype=float,
    )

    aligned_rows: list[dict] = []
    for row in row_records:
        aligned_row = dict(row)
        if row is not reference_row:
            start = np.array([row['start_x'], row['start_y']], dtype=float)
            end = np.array([row['end_x'], row['end_y']], dtype=float)
            length = float(np.linalg.norm(end - start))
            if length > 1e-9:
                aligned_end = start + length * reference_direction
                aligned_row.update(
                    {
                        'A': float(reference_normal[0]),
                        'B': float(reference_normal[1]),
                        'C': float(-(reference_normal @ start)),
                        'end_x': float(aligned_end[0]),
                        'end_y': float(aligned_end[1]),
                    }
                )
        aligned_rows.append(aligned_row)
    return aligned_rows


def build_paths_from_rows(
    row_records: list[dict],
    min_path_length: float = 0.0,
) -> list[dict]:
    """Build paths only between rows from consecutive ROI strips.

    The row detector keeps at most one row per strip. The ordered strip IDs
    therefore define physical adjacency: a row in strip ``i`` may only pair
    with a row in strip ``i + 1``. Missing strips intentionally leave a gap
    rather than creating a path across it.
    """
    if not row_records:
        return []

    path_records: list[dict] = []

    rows_by_strip: dict[int, dict] = {}
    for row in row_records:
        strip_id = row.get('roi_strip_id')
        if isinstance(strip_id, (int, np.integer)) and strip_id > 0:
            rows_by_strip[int(strip_id)] = row

    for strip_id in sorted(rows_by_strip):
        row_i = rows_by_strip[strip_id]
        row_j = rows_by_strip.get(strip_id + 1)
        if row_j is None:
            continue

        start_i = np.array([row_i['start_x'], row_i['start_y']], dtype=float)
        start_j = np.array([row_j['start_x'], row_j['start_y']], dtype=float)
        row_spacing = float(np.linalg.norm(start_i - start_j))

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
                # A path occupies the corridor between this strip and the
                # next one.  The pipeline uses this stable key to retain a
                # path when another strip temporarily has no row-fit result.
                'roi_strip_id': int(strip_id),
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
