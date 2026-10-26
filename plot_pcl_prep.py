# %% Imports and setup
"""Computes a Start  Row algorithm on a ROS 2 MCAP bag without installing ROS.
By: Edward Kikken
Date: 2024-010-26

In VS Code, run these cells in order using Run Cell (Shift+Enter).
Edit the bag settings and read/filter cells to select a timestamp and plot options.

Install dependencies:
    python -m pip install rosbags numpy matplotlib scipy scikit-image ipympl

Relative timestamps are seconds from the start of the entire bag. Point clouds
within a window ending at the requested timestamp are concatenated, then placed
in the world frame using the last GNSS pose strictly before each cloud timestamp.
The cloud is split by lines following each end-of-row pose's local X-axis.
End-of-row poses are extrapolated at each edge using the nearest pair's spacing
until the cloud is covered or the configured cross-row extension limit is reached.
Segmentation excludes points more than the configured distance after the
end-of-row line, measured in the positive local X direction.
Projected starts farther than the configured XY distance from every end-of-row
point (including extrapolated points) are excluded before start-line fitting.
Line fitting caps dense XY cells using fitting_voxel_size and
fitting_max_points_per_voxel. Sparse cells, density searches, and plotting
retain original points. Set downsample_fitting=False for unsampled fitting.
"""

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("module://ipympl.backend_nbagg")
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import skimage as ski


# %%########################################################################################
################################### HELPER FUNCTIONS ######################################
###########################################################################################

# Timestamp parsing
def timestamp_nanoseconds(value):
    """Convert finite decimal seconds to an integer number of nanoseconds.

    Decimal strings avoid the rounding introduced by large Unix-time floats.
    Fractions smaller than one nanosecond are truncated toward zero. Invalid
    values raise argparse.ArgumentTypeError for use in command-line parsing.
    """
    try:
        seconds = Decimal(value)
        if not seconds.is_finite():
            raise ValueError("Timestamp must be finite.")
        return int(seconds * NANOSECONDS)
    except (InvalidOperation, ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError("Timestamp must be finite decimal seconds.") from error


# Load and concatenate point clouds in a window
def load_cloud_window(bag_path, end_timestamp_ns, window_duration_ns, time_base):
    """Read point clouds from bag_topic in a window ending at the requested time.

    Times are integer nanoseconds; time_base='relative' measures the requested
    end from the bag start, while other values treat it as an absolute time.
    Convert o_lidar/os_lidar clouds to rearAxle using lidar_pose_topic; other
    frames pass through for downstream validation. The topic's 'lidar' child
    name is treated as an alias for the lidar cloud frame.

    Return (XYZ arrays, frame, record times, bag start, window start, window end).
    Each XYZ array has shape (N, 3); all timestamps use bag recording time.
    """
    bag_path = Path(bag_path)
    if not bag_path.is_dir():
        raise ValueError(f"Bag directory does not exist: {bag_path}")
    if window_duration_ns <= 0:
        raise ValueError("Cloud window duration must be positive.")
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [connection for connection in reader.connections if connection.topic == bag_topic]
        if not connections:
            raise ValueError(f"Topic {bag_topic} was not found in {bag_path}.")
        if any(connection.msgtype != "sensor_msgs/msg/PointCloud2" for connection in connections):
            raise ValueError(f"Topic {bag_topic} is not a sensor_msgs/msg/PointCloud2 topic.")
        window_end_ns = end_timestamp_ns + reader.start_time if time_base == "relative" else end_timestamp_ns
        if not reader.start_time <= window_end_ns < reader.end_time:
            duration = (reader.end_time - reader.start_time) / NANOSECONDS
            raise ValueError(f"Window end is outside the bag (duration: {duration:.3f} s).")

        window_start_ns = max(reader.start_time, window_end_ns - window_duration_ns)
        # Reader stop times are exclusive; include records exactly at the window end.
        window_stop_ns = min(reader.end_time, window_end_ns + 1)
        xyz_parts = []
        record_times = []
        cloud_frame = None
        for connection, record_ns, raw_data in reader.messages(
            connections=connections, start=window_start_ns, stop=window_stop_ns
        ):
            cloud = reader.deserialize(raw_data, connection.msgtype)
            if cloud_frame is None:
                cloud_frame = cloud.header.frame_id
            elif cloud.header.frame_id != cloud_frame:
                raise ValueError("Point clouds in the selected window use different frames.")
            xyz_parts.append(cloud_xyz(cloud))
            record_times.append(record_ns)

        if not xyz_parts:
            raise ValueError(
                f"Topic {bag_topic} contains no point clouds in the selected window."
            )
        if cloud_frame in ("o_lidar", "os_lidar"):
            lidar_connections = [
                connection for connection in reader.connections
                if connection.topic == lidar_pose_topic
            ]
            if not lidar_connections:
                raise ValueError(f"Topic {lidar_pose_topic} is required for {cloud_frame} clouds.")
            if any(connection.msgtype != "geometry_msgs/msg/TransformStamped" for connection in lidar_connections):
                raise ValueError(f"Topic {lidar_pose_topic} must contain TransformStamped messages.")
            lidar_poses = []
            lidar_times = []
            # Include earlier calibration records, even outside the cloud window.
            for connection, record_ns, raw_data in reader.messages(
                connections=lidar_connections, stop=max(record_times) + 1
            ):
                pose = reader.deserialize(raw_data, connection.msgtype)
                if pose.header.frame_id != "rearAxle" or pose.child_frame_id not in (cloud_frame, "lidar"):
                    raise ValueError(
                        f"Expected lidar transform from {cloud_frame} (or lidar) to rearAxle, got "
                        f"{pose.child_frame_id!r} to {pose.header.frame_id!r}."
                    )
                rotation = pose.transform.rotation
                # Pre-multiply XYZW by a 180-degree rearAxle-Z rotation; keep translation.
                rotation.x, rotation.y, rotation.z, rotation.w = (
                    -rotation.y, rotation.x, rotation.w, -rotation.z
                )
                lidar_poses.append(pose)
                lidar_times.append(record_ns)
            for index, record_ns in enumerate(record_times):
                # Unlike GNSS selection below, an extrinsic at the same time is valid.
                pose_index = int(np.searchsorted(lidar_times, record_ns, side="right")) - 1
                if pose_index < 0:
                    raise ValueError(f"No lidar-to-rearAxle pose is available at cloud timestamp {record_ns}.")
                xyz_parts[index] = transform_points(xyz_parts[index], lidar_poses[pose_index].transform)
            cloud_frame = "rearAxle"
        return xyz_parts, cloud_frame, record_times, reader.start_time, window_start_ns, window_end_ns


# Load all GNSS poses in a bounded window
def load_gnss_window(bag_path, window_start_ns, window_end_ns):
    """Read rear-axle-to-world transforms from the configured gnss_topic.

    Bounds are inclusive absolute bag recording times in nanoseconds. Return
    a pose list and matching sorted int64 timestamps. The selected topic may
    contain rearAxleGNSS or rearAxleKF transforms; this function does not fuse
    them or interpolate poses. No records outside the window are loaded.
    """
    bag_path = Path(bag_path)
    if not bag_path.is_dir():
        raise ValueError(f"Bag directory does not exist: {bag_path}")
    with AnyReader([bag_path], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections = [connection for connection in reader.connections if connection.topic == gnss_topic]
        if not connections:
            raise ValueError(f"Topic {gnss_topic} was not found in {bag_path}.")
        if any(connection.msgtype != "geometry_msgs/msg/TransformStamped" for connection in connections):
            raise ValueError(f"Topic {gnss_topic} must contain TransformStamped messages.")
        if not reader.start_time <= window_start_ns <= window_end_ns < reader.end_time:
            raise ValueError("GNSS search window is outside the bag or has invalid bounds.")
        poses = []
        record_times = []
        for connection, record_time, raw_data in reader.messages(
            connections=connections, start=window_start_ns, stop=window_end_ns + 1
        ):
            pose = reader.deserialize(raw_data, connection.msgtype)
            if pose.header.frame_id != "world" or (pose.child_frame_id != "rearAxleGNSS" and pose.child_frame_id != "rearAxleKF"):
                print(f"Unexpected GNSS transform: {pose.header.frame_id} -> {pose.child_frame_id}")
                raise ValueError("Expected GNSS transforms from rearAxleGNSS or rearAxleKF to world.")
            translation = pose.transform.translation
            if not np.isfinite([translation.x, translation.y, translation.z]).all():
                raise ValueError("GNSS pose position is not finite.")
            poses.append(pose)
            record_times.append(record_time)
        if not poses:
            raise ValueError(f"Topic {gnss_topic} contains no messages in the point-cloud window.")
        order = np.argsort(record_times)
        return [poses[index] for index in order], np.asarray(record_times, dtype=np.int64)[order]


# Load only the nearest end-of-row locations
def load_nearest_end_of_row(bag_path, target_ns):
    """Read a nearby end-of-row PoseArray without changing its world coordinates.

    Search around the absolute bag recording time target_ns, starting at
    +/-0.25 seconds and doubling until a window contains a message. Select
    the nearest record within that first nonempty window; future records
    are allowed. Return (positions (N, 3), record time, XYZW quaternions (N, 4)).
    These are the recorded GNSS-based poses, with no KF correction.
    """
    bag_path = Path(bag_path)
    if not bag_path.is_dir():
        raise ValueError(f"Bag directory does not exist: {bag_path}")
    with AnyReader([bag_path], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections = [connection for connection in reader.connections if connection.topic == end_of_row_topic]
        if not connections:
            raise ValueError(f"Topic {end_of_row_topic} was not found in {bag_path}.")
        if any(connection.msgtype != "geometry_msgs/msg/PoseArray" for connection in connections):
            raise ValueError(f"Topic {end_of_row_topic} must contain PoseArray messages.")
        if not reader.start_time <= target_ns < reader.end_time:
            raise ValueError("Requested end-of-row time is outside the bag.")
        window_ns = NANOSECONDS // 4
        while True:
            start_ns = max(reader.start_time, target_ns - window_ns)
            stop_ns = min(reader.end_time, target_ns + window_ns + 1)
            nearest = None
            for connection, record_time, raw_data in reader.messages(
                connections=connections, start=start_ns, stop=stop_ns
            ):
                if nearest is None or abs(record_time - target_ns) < abs(nearest[1] - target_ns):
                    nearest = (connection, record_time, raw_data)
            if nearest is not None:
                connection, record_time, raw_data = nearest
                message = reader.deserialize(raw_data, connection.msgtype)
                positions, orientations = end_of_row_arrays(message)
                return positions, record_time, orientations
            if start_ns == reader.start_time and stop_ns == reader.end_time:
                raise ValueError(f"Topic {end_of_row_topic} contains no messages.")
            window_ns *= 2


def end_of_row_arrays(message):
    """Decode a world-frame PoseArray into positions and XYZW quaternions.

    Return float arrays of shape (N, 3) and (N, 4). Drop poses with nonfinite
    positions from both arrays to preserve alignment. Quaternion validity
    is checked later when segmentation needs the poses' local X-axes.
    """
    if message.header.frame_id != "world":
        raise ValueError(f"Expected end-of-row frame world, got {message.header.frame_id!r}.")
    positions = np.asarray([
        [pose.position.x, pose.position.y, pose.position.z] for pose in message.poses
    ], dtype=float).reshape(-1, 3)
    orientations = np.asarray([
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        for pose in message.poses
    ], dtype=float).reshape(-1, 4)
    finite = np.isfinite(positions).all(axis=1)
    return positions[finite], orientations[finite]


def cloud_in_world(xyz, cloud_frame, gnss_pose):
    """Place rearAxle XYZ points in world using the supplied rear-axle pose.

    xyz is an (N, 3) array in metres. Lidar-frame input must already have been
    converted by load_cloud_window. Return a new array; do not modify xyz.
    Pose timestamp selection is the responsibility of cloud_window_in_world.
    """
    if cloud_frame != "rearAxle":
        raise ValueError(f"Expected cloud frame rearAxle, got {cloud_frame!r}.")
    return transform_points(xyz, gnss_pose.transform)


def transform_points(xyz, transform):
    """Apply a ROS rotation and translation to an (N, 3) XYZ array.

    The transform maps source coordinates into destination coordinates.
    Normalize its XYZW quaternion and compute p_destination = R p_source + t.
    Coordinates are stored as row vectors, so the array operation uses R.T.
    Reject zero/nonfinite quaternions and nonfinite translations.
    """
    rotation = transform.rotation
    quaternion = np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=float)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Transform has an invalid rotation quaternion.")
    vector = quaternion[:3] / norm
    scalar = quaternion[3] / norm
    # The skew-symmetric matrix implements the quaternion vector's cross product.
    cross_matrix = np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0],
    ])
    rotation_matrix = np.eye(3) + 2 * scalar * cross_matrix + 2 * cross_matrix @ cross_matrix
    translation = transform.translation
    if not np.isfinite([translation.x, translation.y, translation.z]).all():
        raise ValueError("Transform position is not finite.")
    return xyz @ rotation_matrix.T + [translation.x, translation.y, translation.z]


def cloud_window_in_world(pointclouds, cloud_record_times, cloud_frame, gnss_poses, gnss_record_times):
    """Transform and concatenate clouds using preceding rear-axle world poses.

    Cloud arrays and recording times must be aligned; pose times must be
    sorted. Select the last supplied pose strictly before each cloud, with
    no interpolation or future-pose fallback. An exact timestamp match is
    deliberately not eligible. Raise ValueError if no preceding pose exists.
    Return stacked world XYZ and one selected pose timestamp per cloud.
    """
    if len(pointclouds) != len(cloud_record_times):
        raise ValueError("Each point cloud must have one recording timestamp.")
    if not gnss_poses or len(gnss_poses) != len(gnss_record_times):
        raise ValueError("GNSS poses and recording timestamps must be nonempty and aligned.")
    gnss_record_times = np.asarray(gnss_record_times, dtype=np.int64)
    if np.any(gnss_record_times[1:] < gnss_record_times[:-1]):
        raise ValueError("GNSS recording timestamps must be sorted in ascending order.")
    world_clouds = []
    selected_gnss_times = []
    for points, cloud_time in zip(pointclouds, cloud_record_times):
        pose_index = int(np.searchsorted(gnss_record_times, cloud_time, side="left")) - 1
        if pose_index < 0:
            raise ValueError(f"No GNSS pose is available strictly before cloud timestamp {cloud_time}.")
        world_clouds.append(cloud_in_world(points, cloud_frame, gnss_poses[pose_index]))
        selected_gnss_times.append(gnss_record_times[pose_index])
    return np.vstack(world_clouds), np.asarray(selected_gnss_times, dtype=np.int64)


def midpoints_between_points(points):
    """Return XYZ midpoints between consecutive input points, in input order.

    The points are not sorted here: neighboring entries must already describe
    neighboring row ends. Fewer than two points produce an empty (0, 3) array.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 2:
        return np.empty((0, 3), dtype=float)
    return (points[:-1] + points[1:]) / 2


# Decode PointCloud2 coordinates
def cloud_xyz(cloud):
    """Decode finite XYZ coordinates from a ROS PointCloud2 without ROS installed.

    Use the message's scalar field types, byte order, field offsets and row/
    point strides rather than assuming tightly packed float32 coordinates.
    Flatten organized clouds into an (N, 3) array in the original cloud frame
    and discard points with NaN or infinite coordinates. Ignore other fields.
    """
    field_types = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    fields = {field.name: field for field in cloud.fields}
    formats = []
    offsets = []
    for name in ("x", "y", "z"):
        field = fields.get(name)
        if field is None or field.count != 1 or field.datatype not in field_types:
            raise ValueError(f"Point cloud needs a supported scalar '{name}' field.")
        byte_order = ">" if cloud.is_bigendian else "<"
        formats.append(np.dtype(byte_order + field_types[field.datatype]))
        offsets.append(field.offset)
    if cloud.width == 0 or cloud.height == 0:
        raise ValueError("Selected point cloud is empty.")
    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": formats,
        "offsets": offsets,
        "itemsize": cloud.point_step,
    })
    # Explicit strides preserve padding between points and organized-cloud rows.
    points = np.ndarray(
        shape=(cloud.height, cloud.width), dtype=dtype, buffer=cloud.data,
        strides=(cloud.row_step, cloud.point_step),
    )
    xyz = np.column_stack([points[name].reshape(-1) for name in ("x", "y", "z")])
    return xyz[np.isfinite(xyz).all(axis=1)]



# Segment a cloud along the end-of-row poses' local X-axes
def segment_cloud_by_end_of_row_x(xyz_world, end_of_row_xyz, orientations, max_distance_after_end_of_row=2.0):
    """Split XY into bands, clipping after the interpolated end-of-row line.

    The limit is in metres along positive local X. Excluded points get ID -1
    and are omitted from segments. Without end-of-row poses, no limit is applied.

    Inputs use world XYZ and aligned XYZW orientations. Project local X-axes
    into XY and require parallel row directions. Return (segment arrays,
    per-point IDs, sorted cross-row boundaries, XY origin, XY row direction).
    Z is preserved but does not affect band assignment or the headland cutoff.
    """
    if not np.isfinite(max_distance_after_end_of_row) or max_distance_after_end_of_row < 0:
        raise ValueError("Maximum distance after end-of-row must be finite and nonnegative.")
    if not len(end_of_row_xyz):
        return [xyz_world.copy()], np.zeros(len(xyz_world), dtype=int), np.empty(0), np.zeros(2), np.array([1., 0.])
    norms = np.linalg.norm(orientations, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("End-of-row poses have invalid rotation quaternions.")
    quaternions = orientations / norms[:, None]
    quat_x, quat_y, quat_z, quat_w = quaternions.T
    # Only the first rotation-matrix column is needed: the local X-axis in world.
    directions = np.column_stack((
        1 - 2 * (quat_y**2 + quat_z**2),
        2 * (quat_x * quat_y + quat_w * quat_z),
    ))
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError("End-of-row local X-axis cannot be projected onto the XY plot.")
    directions = directions / lengths[:, None]
    direction = directions[0]
    if not np.allclose(np.abs(directions @ direction), 1., atol=1e-5, rtol=0):
        raise ValueError("End-of-row local X-axes must be parallel to define segmentation bands.")
    origin = end_of_row_xyz[0, :2]
    normal = np.array([-direction[1], direction[0]])
    # Row ends define boundaries across rows; their X-axes run along the rows.
    boundaries, boundary_indices = np.unique(
        (end_of_row_xyz[:, :2] - origin) @ normal, return_index=True
    )
    across_row = (xyz_world[:, :2] - origin) @ normal
    segment_ids = np.searchsorted(boundaries, across_row, side="right")
    along_row = (xyz_world[:, :2] - origin) @ direction
    end_of_row_along = (end_of_row_xyz[boundary_indices, :2] - origin) @ direction
    # Interpolation follows staggered row ends rather than imposing a flat cutoff.
    cutoff = np.interp(across_row, boundaries, end_of_row_along) + max_distance_after_end_of_row
    segment_ids[along_row > cutoff] = -1
    segments = [xyz_world[segment_ids == index] for index in range(len(boundaries) + 1)]
    return segments, segment_ids, boundaries, origin, direction


def extend_end_of_row_points(
    xyz_world, end_of_row_xyz, orientations, max_extension_distance=10.0
):
    """Extend toward the cloud, at most max_extension_distance metres across each edge.

    Keep the edge pair's spacing; omit any new pose that would exceed the limit.
    Sort existing world poses across the rows, extrapolate XYZ using the first
    and last pair, and copy each edge's quaternion to its new poses. Return
    aligned position/orientation arrays. With fewer than two row ends or an
    empty cloud, return copies of the inputs without extrapolating.
    """
    if not np.isfinite(max_extension_distance) or max_extension_distance < 0:
        raise ValueError("Maximum end-of-row extension distance must be finite and nonnegative.")
    if len(end_of_row_xyz) < 2 or not len(xyz_world):
        return end_of_row_xyz.copy(), orientations.copy()
    _, _, _, origin, direction = segment_cloud_by_end_of_row_x(
        np.empty((0, 3)), end_of_row_xyz, orientations
    )
    normal = np.array([-direction[1], direction[0]])
    offsets = (end_of_row_xyz[:, :2] - origin) @ normal
    order = np.argsort(offsets, kind="stable")
    points = end_of_row_xyz[order]
    rotations = orientations[order]
    offsets = offsets[order]
    spacing = np.diff(offsets)
    if np.any(spacing <= 1e-8):
        raise ValueError("End-of-row points need distinct cross-row positions for extrapolation.")
    cloud_offsets = (xyz_world[:, :2] - origin) @ normal
    # Ceil covers the cloud extent; floor enforces the hard extension-distance cap.
    before_count = max(0, int(np.ceil((offsets[0] - cloud_offsets.min()) / spacing[0])))
    after_count = max(0, int(np.ceil((cloud_offsets.max() - offsets[-1]) / spacing[-1])))
    before_count = min(before_count, int(np.floor(max_extension_distance / spacing[0])))
    after_count = min(after_count, int(np.floor(max_extension_distance / spacing[-1])))
    before = points[0] - np.arange(before_count, 0, -1)[:, None] * (points[1] - points[0])
    after = points[-1] + np.arange(1, after_count + 1)[:, None] * (points[-1] - points[-2])
    return (
        np.vstack((before, points, after)),
        np.vstack((
            np.repeat(rotations[:1], before_count, axis=0),
            rotations,
            np.repeat(rotations[-1:], after_count, axis=0),
        )),
    )


def find_segment_starts_nearest_to_gnss(
    segments, common_heading, gnss_position, min_neighbor_points, neighbor_radius
):
    """Find a supported row start by scanning inward from the GNSS-facing end.

    For each nonempty segment, choose the heading-extreme endpoint nearest
    GNSS in XY, then scan inward until a point has min_neighbor_points within
    neighbor_radius metres. The point itself is not counted as a neighbor.
    If none qualify, use the densest point, breaking ties by inward order.

    Return (segment index, XYZ point, neighbor count, density passed) tuples.
    Heading and neighbor distances use XY; preserve the selected point's Z.
    """
    heading = np.asarray(common_heading, dtype=float)
    heading_norm = np.linalg.norm(heading)
    if heading.shape != (2,) or not np.isfinite(heading_norm) or heading_norm == 0:
        raise ValueError("Common heading must be a finite, nonzero XY direction.")
    heading = heading / heading_norm
    if not isinstance(min_neighbor_points, (int, np.integer)) or min_neighbor_points < 0:
        raise ValueError("Minimum neighbor count must be a nonnegative integer.")
    if not np.isfinite(neighbor_radius) or neighbor_radius <= 0:
        raise ValueError("Neighbor radius must be finite and positive.")
    gnss_xy = np.asarray(gnss_position, dtype=float)[:2]
    if gnss_xy.shape != (2,) or not np.isfinite(gnss_xy).all():
        raise ValueError("GNSS position must contain finite XY coordinates.")

    starts = []
    for segment_index, segment in enumerate(segments):
        if not len(segment):
            continue
        points = np.asarray(segment, dtype=float)
        along_heading = (points[:, :2] - gnss_xy) @ heading
        endpoint_indices = (np.argmin(along_heading), np.argmax(along_heading))
        point_tree = cKDTree(points[:, :2])
        # Choose between along-row endpoints, not the globally nearest cloud point.
        endpoint_index = min(
            endpoint_indices,
            key=lambda index: np.linalg.norm(points[index, :2] - gnss_xy),
        )
        neighbor_counts = np.asarray(
            point_tree.query_ball_point(points[:, :2], neighbor_radius, return_length=True)
        ) - 1
        heading_step = 1 if endpoint_index == endpoint_indices[0] else -1
        inward_distance = (along_heading - along_heading[endpoint_index]) * heading_step
        candidate_order = np.argsort(inward_distance, kind="stable")
        supported_candidates = candidate_order[
            neighbor_counts[candidate_order] >= min_neighbor_points
        ]
        if len(supported_candidates):
            start_index = supported_candidates[0]
            meets_density = True
        else:
            start_index = candidate_order[np.argmax(neighbor_counts[candidate_order])]
            meets_density = False
        neighbor_count = int(neighbor_counts[start_index])
        start = points[start_index]
        starts.append((segment_index, start.copy(), neighbor_count, meets_density))

    return starts


def project_segment_starts_onto_fit_lines(segment_starts, fits):
    """Project start XY onto the finite fitted line segment for the same row.

    Clamp projections to the displayed endpoints, rather than extending the
    line indefinitely. Preserve Z, neighbor counts and density flags. fits
    is indexed by segment ID; omit starts whose corresponding fit is None.
    """
    projected_starts = []
    for segment_index, start, neighbor_count, meets_density in segment_starts:
        fit = fits[segment_index]
        if fit is None:
            continue
        line_start, line_end = fit["line_xy"]
        line_vector = line_end - line_start
        length_squared = np.dot(line_vector, line_vector)
        fraction = (
            np.clip(np.dot(start[:2] - line_start, line_vector) / length_squared, 0., 1.)
            if length_squared > 0 else 0.
        )
        projected_start = start.copy()
        projected_start[:2] = line_start + fraction * line_vector
        projected_starts.append(
            (segment_index, projected_start, neighbor_count, meets_density)
        )
    return projected_starts


def lidar_fov_geometry(gnss_position, gnss_heading, fov_angle_deg=120.0):
    """Construct a simplified horizontal LiDAR field of view in world XY.

    gnss_position supplies x/y in metres and gnss_heading is in radians.
    Return the origin 1 m forward of GNSS and two unit boundary directions
    at +/- half fov_angle_deg. This fixed plotting/filtering geometry does
    not derive its offset or orientation from the recorded lidar extrinsic.
    """
    if not np.isfinite(fov_angle_deg) or not 0 < fov_angle_deg < 360:
        raise ValueError("LiDAR FOV angle must be finite and strictly between 0 and 360 degrees.")
    if not np.isfinite([gnss_position.x, gnss_position.y, gnss_heading]).all():
        raise ValueError("LiDAR geometry requires finite GNSS XY coordinates and heading.")
    origin = np.array([gnss_position.x, gnss_position.y]) + np.array([
        np.cos(gnss_heading), np.sin(gnss_heading)
    ])
    angles = gnss_heading + np.deg2rad([-fov_angle_deg / 2, fov_angle_deg / 2])
    return origin, np.column_stack((np.cos(angles), np.sin(angles)))


def distances_to_fov_rays(points_xy, origin, rays):
    """Return each XY point's distance to the nearer forward FOV boundary ray.

    points_xy has shape (N, 2), origin is (2,), and rays contains two unit
    directions. Clamp negative along-ray projections to the origin, so points
    behind a ray are measured to its endpoint, not its backward extension.
    """
    relative = points_xy - origin
    along = np.maximum(relative @ rays.T, 0.0)
    residuals = relative[:, None, :] - along[:, :, None] * rays[None, :, :]
    return np.linalg.norm(residuals, axis=2).min(axis=1)


def exclude_segments_near_lidar_fov(
    segments, segment_ids, gnss_position, gnss_heading, distance=0.5, fov_angle_deg=120.0
):
    """Empty entire segments with any point within distance metres of either FOV ray.

    Test original segment points, including outside-FOV points. Preserve segment
    indices; mark all excluded points with ID -1 for plotting.
    Return (filtered segment list, copied point IDs, excluded segment IDs).
    Empty arrays retain their original segment slots so fits and starts can
    continue to use the same segment indices. Input arrays are not modified.
    """
    if not np.isfinite(distance) or distance < 0:
        raise ValueError("Segment FOV exclusion distance must be finite and nonnegative.")
    origin, rays = lidar_fov_geometry(gnss_position, gnss_heading, fov_angle_deg)
    filtered = list(segments)
    ids = segment_ids.copy()
    excluded = []
    for index, segment in enumerate(segments):
        if not len(segment):
            continue
        if not np.isfinite(segment[:, :2]).all():
            raise ValueError("Segment points must contain finite XY coordinates.")
        if np.any(distances_to_fov_rays(segment[:, :2], origin, rays) <= distance):
            filtered[index] = segment[:0].copy()
            ids[segment_ids == index] = -1
            excluded.append(index)
    return filtered, ids, tuple(excluded)


def filter_starts_away_from_lidar_fov(
    segment_starts, gnss_position, gnss_heading, margin_fov_distance=0.5, fov_angle_deg=120.0
):
    """Keep starts inside the configured FOV with clearance from both boundary rays.

    Distances use half-lines, not their backward extensions or plot-clipped
    segments. Starts exactly at the minimum distance are retained.
    Applies equally to density-passing starts and densest-point fallbacks.
    Use world XY positions and a GNSS heading in radians; Z does not participate.
    Return the surviving start tuples in their original order, without changing
    their point coordinates or density metadata.
    """
    if not np.isfinite(margin_fov_distance) or margin_fov_distance < 0:
        raise ValueError("Minimum FOV boundary distance must be finite and nonnegative.")
    origin, rays = lidar_fov_geometry(gnss_position, gnss_heading, fov_angle_deg)
    if not segment_starts:
        return []
    points = np.asarray([entry[1][:2] for entry in segment_starts], dtype=float)
    if not np.isfinite(points).all():
        raise ValueError("Segment starts must contain finite XY coordinates.")
    relative = points - origin
    forward = np.array([np.cos(gnss_heading), np.sin(gnss_heading)])
    left = np.array([-forward[1], forward[0]])
    relative_angles = np.arctan2(relative @ left, relative @ forward)
    inside_fov = np.abs(relative_angles) <= np.deg2rad(fov_angle_deg / 2)
    distances = distances_to_fov_rays(points, origin, rays)
    keep = inside_fov & (distances >= margin_fov_distance)
    return [entry for entry, retain in zip(segment_starts, keep) if retain]


def filter_starts_near_end_of_row(segment_starts, end_of_row_xyz, max_distance=2.0):
    """Keep starts within max_distance metres in XY of any end-of-row point.

    Query nearest discrete row-end locations, including any extrapolated
    points supplied by the caller; do not measure distance to connecting
    lines. Keep starts exactly on the distance limit and preserve tuple order.
    """
    if not np.isfinite(max_distance) or max_distance < 0:
        raise ValueError("Maximum start-to-end-of-row distance must be finite and nonnegative.")
    if not segment_starts:
        return []
    end_of_row_xy = np.asarray(end_of_row_xyz, dtype=float).reshape(-1, 3)[:, :2]
    if not len(end_of_row_xy) or not np.isfinite(end_of_row_xy).all():
        raise ValueError("Start distance filtering requires finite, nonempty end-of-row points.")
    start_xy = np.asarray([entry[1][:2] for entry in segment_starts], dtype=float)
    if not np.isfinite(start_xy).all():
        raise ValueError("Segment starts must contain finite XY coordinates.")
    distances, _ = cKDTree(end_of_row_xy).query(start_xy)
    return [entry for entry, distance in zip(segment_starts, distances) if distance <= max_distance]


def fit_line_through_largest_heading_starts(segment_starts, common_heading):
    """Connect the two starts farthest along the supplied common XY heading.

    This selects the two largest projections, not the minimum and maximum.
    Return a dictionary with XY origin, unit direction, endpoints and source
    segment IDs, or None for fewer than two starts or coincident endpoints.
    """
    if len(segment_starts) < 2:
        return None
    heading = np.asarray(common_heading, dtype=float)
    heading_norm = np.linalg.norm(heading)
    if heading.shape != (2,) or not np.isfinite(heading_norm) or heading_norm == 0:
        raise ValueError("Common heading must be a finite, nonzero XY direction.")
    heading = heading / heading_norm
    start_points = np.asarray([entry[1][:2] for entry in segment_starts])
    projections = start_points @ heading
    selected_indices = np.argsort(projections, kind="stable")[-2:]
    line_xy = start_points[selected_indices]
    line_vector = line_xy[1] - line_xy[0]
    line_length = np.linalg.norm(line_vector)
    if line_length == 0:
        return None
    return {
        "origin": line_xy[0],
        "direction": line_vector / line_length,
        "line_xy": line_xy,
        "segment_indices": tuple(segment_starts[index][0] for index in selected_indices),
    }


def fit_line_through_all_starts(segment_starts, common_heading):
    """Fit an orthogonal least-squares XY line robust to one outlying start.

    With at least four starts, retain the leave-one-out subset with the lowest
    squared perpendicular error, considering only omitted starts whose
    perpendicular residual has negative projection onto common_heading.
    Otherwise use all starts. All density statuses are eligible; endpoints
    span the retained starts' projections.
    Return XY origin/direction/endpoints, retained and excluded segment IDs,
    and an inlier mask aligned with segment_starts. Return None when fewer
    than two distinct XY positions are available.
    """
    if len(segment_starts) < 2:
        return None
    heading = np.asarray(common_heading, dtype=float)
    heading_norm = np.linalg.norm(heading)
    if heading.shape != (2,) or not np.isfinite(heading_norm) or heading_norm == 0:
        raise ValueError("Common heading must be a finite, nonzero XY direction.")
    heading = heading / heading_norm
    start_points = np.asarray([entry[1][:2] for entry in segment_starts], dtype=float)
    if not np.isfinite(start_points).all():
        raise ValueError("Segment starts must contain finite XY coordinates.")
    if len(np.unique(start_points, axis=0)) < 2:
        return None
    inlier_mask = np.ones(len(start_points), dtype=bool)
    if len(start_points) >= 4:
        # Test each single omission; only the specified side of the line is eligible.
        best_error = np.inf
        for excluded_index in range(len(start_points)):
            candidate_mask = np.arange(len(start_points)) != excluded_index
            candidate_points = start_points[candidate_mask]
            centered = candidate_points - candidate_points.mean(axis=0)
            _, singular_values, candidate_axes = np.linalg.svd(centered, full_matrices=False)
            if singular_values[0] == 0:
                continue
            candidate_direction = candidate_axes[0]
            displacement = start_points[excluded_index] - candidate_points.mean(axis=0)
            residual = displacement - np.dot(displacement, candidate_direction) * candidate_direction
            if np.dot(residual, heading) >= -1e-8:  # only consider outlier if away from headland space 
                continue
            squared_error = singular_values[1] ** 2
            if squared_error < best_error:
                best_error = squared_error
                inlier_mask = candidate_mask
    retained_points = start_points[inlier_mask]
    origin = retained_points.mean(axis=0)
    centered = retained_points - origin
    # The leading right-singular vector minimizes orthogonal, not vertical, errors.
    _, _, right_transpose = np.linalg.svd(centered, full_matrices=False)
    direction = right_transpose[0]
    projections = centered @ direction
    line_xy = origin + np.array([projections.min(), projections.max()])[:, None] * direction
    return {
        "origin": origin,
        "direction": direction,
        "line_xy": line_xy,
        "segment_indices": tuple(
            entry[0] for entry, keep in zip(segment_starts, inlier_mask) if keep
        ),
        "excluded_segment_indices": tuple(
            entry[0] for entry, keep in zip(segment_starts, inlier_mask) if not keep
        ),
        "inlier_mask": inlier_mask,
    }


def estimate_start_to_midpoint_transform(start_line_fit, midpoint_points):
    """Estimate the XY rigid transform matching line endpoints to distinct nearest midpoints.

    Heading error is the signed counterclockwise start-to-midpoint rotation.
    X/Y errors are the transform's world-frame translation in metres, applied
    after rotation about the world origin: midpoint = rotation @ start + error.
    Match the two fitted endpoints to different midpoint locations using a
    minimum-total-distance assignment, then solve a proper rotation (not a
    reflection) and translation. Return a homogeneous 3x3 matrix, heading
    errors, translation components, matched points and residual RMSE.
    Return None if a line or two distinct matched positions are unavailable.
    """
    if start_line_fit is None or len(midpoint_points) < 2:
        return None
    start_points = np.asarray(start_line_fit["line_xy"], dtype=float)
    midpoint_xy = np.asarray(midpoint_points, dtype=float)[:, :2]
    distances = np.linalg.norm(start_points[:, None, :] - midpoint_xy[None, :, :], axis=2)
    # One-to-one assignment prevents both endpoints from selecting the same midpoint.
    start_indices, midpoint_indices = linear_sum_assignment(distances)
    matched_starts = start_points[start_indices]
    matched_midpoints = midpoint_xy[midpoint_indices]
    if np.linalg.norm(np.diff(matched_starts, axis=0)) == 0 or np.linalg.norm(
        np.diff(matched_midpoints, axis=0)
    ) == 0:
        return None

    start_center = matched_starts.mean(axis=0)
    midpoint_center = matched_midpoints.mean(axis=0)
    covariance = (matched_starts - start_center).T @ (matched_midpoints - midpoint_center)
    left, _, right_transpose = np.linalg.svd(covariance)
    correction = np.eye(2)
    # The determinant correction prevents the SVD solution from reflecting the rows.
    correction[-1, -1] = np.linalg.det(left @ right_transpose)
    rotation = (left @ correction @ right_transpose).T
    translation = midpoint_center - rotation @ start_center
    predicted_midpoints = (rotation @ matched_starts.T).T + translation
    residuals = np.linalg.norm(predicted_midpoints - matched_midpoints, axis=1)
    transform = np.eye(3)
    transform[:2, :2] = rotation
    transform[:2, 2] = translation
    heading_error_rad = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return {
        "matrix": transform,
        "heading_error_rad": heading_error_rad,
        "heading_error_deg": float(np.degrees(heading_error_rad)),
        "x_error_m": float(translation[0]),
        "y_error_m": float(translation[1]),
        "start_points": matched_starts,
        "midpoint_points": matched_midpoints,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
    }


# Fit XY lines with RANSAC and at most 5 percent rejected points
def dense_region_sample_indices(segment, voxel_size=0.05, max_points_per_voxel=5):
    """Keep sparse XY cells intact and spatially sample dense cells deterministically.

    Return original point indices. Cell size is in metres; Z is not voxelized.
    Preserve every point in sparse cells. In dense cells, sort by coordinates
    and retain evenly spaced ranks up to the cap, avoiding random resampling.
    Return indices in original input order so callers can sample only fitting
    data while retaining the complete cloud for density checks and plotting.
    """
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("Fitting voxel size must be finite and positive.")
    if not isinstance(max_points_per_voxel, (int, np.integer)) or max_points_per_voxel < 1:
        raise ValueError("Maximum fitting points per voxel must be a positive integer.")
    if not len(segment):
        return np.empty(0, dtype=int)
    if not np.isfinite(segment).all():
        raise ValueError("Fitting points must be finite.")
    cells = np.floor(segment[:, :2] / voxel_size)
    _, groups, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    order = np.lexsort((segment[:, 2], segment[:, 1], segment[:, 0], groups))
    group_counts = counts[groups[order]]
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    ranks = np.arange(len(segment)) - starts
    retained_counts = np.minimum(group_counts, max_points_per_voxel)
    # Integer bucket transitions select evenly distributed ranks within each cell.
    keep = ((ranks + 1) * retained_counts // group_counts
            > ranks * retained_counts // group_counts)
    return np.sort(order[keep])


def fit_segment_ransac(
    segment, max_trials=10, seed=42, common_heading=None, retained_percentile=0.95,
    sample_indices=None,
):
    """Fit a robust XY row line, optionally using a sampled subset of XYZ points.

    retained_percentile is a fraction, e.g. 0.95; retain the lowest-residual
    points after RANSAC. Without common_heading, refit their orthogonal line.
    With a heading, fix the direction and use their median perpendicular offset.
    sample_indices restrict fitting only; returned masks and residuals always
    align with the original segment. Fall back to the full cloud if the sample
    is degenerate, and return None when no two distinct fitting points exist.

    Return origin, unit direction, finite XY endpoints, inlier mask, residuals,
    inlier RMSE and the number of points used for fitting. Z is not fitted.
    """
    full_points = segment[:, :2]
    points = full_points if sample_indices is None else full_points[sample_indices]
    if sample_indices is not None and len(np.unique(points, axis=0)) < 2:
        points = full_points
        sample_indices = None
    if len(points) < 2 or len(np.unique(points, axis=0)) < 2:
        return None
    rejected_count = int(len(points) * (1 - retained_percentile))
    retained_count = len(points) - rejected_count
    initial_model = ski.measure.LineModelND.from_estimate(points)
    initial_residuals = initial_model.residuals(points)
    # Adapt the RANSAC distance threshold to the requested retained fraction.
    threshold = max(float(np.partition(initial_residuals, retained_count - 1)[retained_count - 1]), 1e-9)
    model, _ = ski.measure.ransac(
        points,
        ski.measure.LineModelND,
        min_samples=min(100, len(points)),
        residual_threshold=threshold,
        is_data_valid=lambda sample: np.any(np.ptp(sample, axis=0) > 0),
        max_trials=max_trials,
        stop_probability=0.99,
        rng=seed,
    )
    if model is None:
        raise ValueError("RANSAC could not find a valid line for a nondegenerate segment.")
    retained_indices = np.argsort(model.residuals(points), kind="stable")[:retained_count]
    inlier_mask = np.zeros(len(points), dtype=bool)
    inlier_mask[retained_indices] = True
    if common_heading is None:
        model = ski.measure.LineModelND.from_estimate(points[inlier_mask])
        origin, direction = model.origin, model.direction
    else:
        direction = np.asarray(common_heading, dtype=float)
        heading_norm = np.linalg.norm(direction)
        if not np.isfinite(heading_norm) or heading_norm == 0:
            raise ValueError("Common RANSAC heading must be finite and nonzero.")
        direction = direction / heading_norm
        normal = np.array([-direction[1], direction[0]])
        origin = normal * np.median(points[inlier_mask] @ normal)
    # Classify original points so masks, heading weights and plotted rejects
    # retain their original-cloud meaning after fitting on sampled points.
    full_residuals = (
        model.residuals(full_points) if common_heading is None
        else np.abs((full_points - origin) @ normal)
    )
    full_retained_count = len(full_points) - int(len(full_points) * (1 - retained_percentile))
    if sample_indices is not None:
        retained_indices = np.argsort(full_residuals, kind="stable")[:full_retained_count]
        inlier_mask = np.zeros(len(full_points), dtype=bool)
        inlier_mask[retained_indices] = True
    along_line = (full_points[inlier_mask] - origin) @ direction
    line_xy = origin + np.array([along_line.min(), along_line.max()])[:, None] * direction
    if common_heading is None:
        residuals = full_residuals
    else:
        residuals = full_residuals
    return {
        "origin": origin,
        "direction": direction,
        "inlier_mask": inlier_mask,
        "line_xy": line_xy,
        "residuals": residuals,
        "inlier_rmse": float(np.sqrt(np.mean(residuals[inlier_mask] ** 2))),
        "fit_point_count": len(points),
    }


def process_world_cloud(xyz_world, gnss_position, recorded_end_of_row_xyz, orientations, gnss_heading):
    """Run row/start detection, printing and returning processing time in milliseconds.

    Inputs are world-frame XYZ arrays, aligned end-of-row XYZW quaternions,
    a GNSS translation with x/y fields, and a heading in radians. Detection
    settings are read from the module's parameter cell.

    Extend row ends, segment the cloud, fit individual and common-heading
    lines, detect/filter starts, and estimate a start-to-midpoint XY transform.
    Return a dictionary containing intermediate geometry, fits, starts,
    rejected points and timing for inspection and plotting. The plot-only
    height cutoff is not applied here. Timing excludes bag loading and plotting.
    """
    computation_start = perf_counter()
    # Build a shared row coordinate system before fitting individual segments.
    end_of_row_xyz, orientations = extend_end_of_row_points(
        xyz_world, recorded_end_of_row_xyz, orientations, max_end_of_row_extension_distance
    )
    midpoints = midpoints_between_points(end_of_row_xyz)
    segments, ids, boundaries, origin, direction = segment_cloud_by_end_of_row_x(
        xyz_world, end_of_row_xyz, orientations, max_distance_after_end_of_row
    )
    segments, ids, fov_excluded_segments = exclude_segments_near_lidar_fov(
        segments, ids, gnss_position, gnss_heading, distance=0.5,
        fov_angle_deg=lidar_fov_angle_deg,
    )

    # Sampling reduces fitting cost; density searches still use the original points.
    fitting_samples = [
        dense_region_sample_indices(segment, fitting_voxel_size, fitting_max_points_per_voxel)
        if downsample_fitting else None for segment in segments  # 
    ]
    individual_fits = [
        fit_segment_ransac(segment, max_trials=max_RANSAC_trials, retained_percentile=retained_percentile, sample_indices=sample)
        for segment, sample in zip(segments, fitting_samples)
    ]
    valid_fits = [fit for fit in individual_fits if fit is not None][:-1]
    if valid_fits:
        reference = direction if len(end_of_row_xyz) else valid_fits[0]["direction"]
        # A line has two equivalent directions; align signs before averaging headings.
        aligned = [
            fit["direction"] if np.dot(fit["direction"], reference) >= 0
            else -fit["direction"] for fit in valid_fits
        ]
        heading = np.average(
            aligned, axis=0, weights=[fit["inlier_mask"].sum() for fit in valid_fits]
        )
        heading_norm = np.linalg.norm(heading)
        heading = heading / heading_norm if heading_norm > 1e-8 else reference
    else:
        heading = direction if len(end_of_row_xyz) else None
    common_fits = [
        fit_segment_ransac(
            segment, common_heading=heading, max_trials=max_RANSAC_trials, retained_percentile=retained_percentile,
            sample_indices=sample,
        )
        for segment, sample in zip(segments, fitting_samples)
    ]
    fit_count = len(segments) - int(bool(len(end_of_row_xyz)))
    if heading is None:
        first_fit = next((fit for fit in common_fits[:fit_count] if fit is not None), None)
        if first_fit is not None:
            heading = first_fit["direction"]
    starts = []
    if heading is not None and fit_count > 0:
        # Detect on raw points, project onto fits, then apply row-end and FOV gates.
        starts = find_segment_starts_nearest_to_gnss(
            segments[:fit_count], heading, [gnss_position.x, gnss_position.y],
            start_point_min_neighbors, start_point_neighbor_radius,
        )
        starts = project_segment_starts_onto_fit_lines(starts, common_fits)
        starts = filter_starts_near_end_of_row(
            starts, end_of_row_xyz, start_point_max_end_of_row_distance
        )
        starts = filter_starts_away_from_lidar_fov(
            starts, gnss_position, gnss_heading, margin_fov_distance, lidar_fov_angle_deg
        )
    start_line = fit_line_through_all_starts(starts, heading)
    transform = estimate_start_to_midpoint_transform(start_line, midpoints)
    transformed_midpoints = None
    transformed_end_of_row_xyz = None
    if transform is not None:
        midpoint_to_start = np.linalg.inv(transform["matrix"])
        transformed_midpoints = midpoints.copy()
        transformed_midpoints[:, :2] = (
            midpoint_to_start[:2, :2] @ midpoints[:, :2].T
        ).T + midpoint_to_start[:2, 2]
        transformed_end_of_row_xyz = end_of_row_xyz.copy()
        transformed_end_of_row_xyz[:, :2] = (
            midpoint_to_start[:2, :2] @ end_of_row_xyz[:, :2].T
        ).T + midpoint_to_start[:2, 2]
    rejected = [
        segment[~fit["inlier_mask"]] for segment, fit in zip(segments, individual_fits)
        if fit is not None
    ]
    result = {
        "xyz_world": xyz_world, "gnss_position": gnss_position, "gnss_heading": gnss_heading,
        "recorded_end_of_row_xyz": recorded_end_of_row_xyz, "end_of_row_xyz": end_of_row_xyz,
        "midpoints": midpoints, "segments": segments, "segment_ids": ids,
        "boundaries": boundaries, "origin": origin, "direction": direction,
        "individual_fits": individual_fits, "common_fits": common_fits,
        "fit_count": fit_count, "segment_starts": starts, "start_line_fit": start_line,
        "start_midpoint_transform": transform,
        "transformed_midpoints": transformed_midpoints,
        "transformed_end_of_row_xyz": transformed_end_of_row_xyz,
        "fov_excluded_segment_indices": fov_excluded_segments,
        "ransac_outliers": np.vstack(rejected) if rejected else np.empty((0, 3)),
    }
    result["computation_time_ms"] = (perf_counter() - computation_start) * 1000.0
    print(f"process_world_cloud computation time: {result['computation_time_ms']:.3f} ms")
    return result


 


# %%#######################################################################################
################################### PARAMETERS ############################################
###########################################################################################

# Load bag folder and topic
bag_folder = "C:\\Users\\edward.kikken\\OneDrive - Flanders Make vzw\\Documenten\\66_PERNAV\\Jetson_logs\\2026-10-13_17-07-26"
bag_topic = "/pcl_prep"
lidar_pose_topic = "/pose_lidar2rearAxle"
gnss_topic = "/pose_rearAxle2worldGNSS" 
end_of_row_topic = "/end_of_row"
NANOSECONDS = 1_000_000_000

# time and stitching window settings
for timestamp_seconds in [44.0]:# np.arange(38, 47, 0.1):#  pick a time to compute and plot
    timestamp_seconds = str(timestamp_seconds)
    window_duration_seconds = "0.5"  #  [s]
    time_base = "relative"
    save_path = "results/"+ (timestamp_seconds.replace(".", "_") + ".png")
    show_plot = True
    print(f"Processing timestamp: {timestamp_seconds}")

    # Parameters for start point detection
    start_point_min_neighbors = 5   # adapt this when changing the window size
    start_point_neighbor_radius = 0.25  # radius to search for neighboring points when detecting start points
    margin_end_of_row = 2.0  # margin around original EoR to look for SoR
    max_end_of_row_extension_distance = 10.0  # how far do we extrapolate the end of row points?
    margin_fov_distance = 0.5   # how close to FoV edge can we detect a start of vine?   
    lidar_fov_angle_deg = 120.0  # total horizontal field of view, centered on GNSS heading
    retained_percentile = 0.99  # percentile of points to retain after RANSAC fitting
    downsample_fitting = True
    fitting_voxel_size = 0.1  # XY cell width [m], only for line fitting
    fitting_max_points_per_voxel = 5
    max_RANSAC_trials = 2

    start_point_max_end_of_row_distance = margin_end_of_row  # how far away a start point can be from the nearest end-of-row point
    max_distance_after_end_of_row = margin_end_of_row # maximum allowed distance that the pointcloud can go onto the headland

    ##########################################################################################
    ################################### START OF ALGORITHM #####################################
    ############################################################################################
    """Run the configured single-timestamp workflow."""
    window_ns = timestamp_nanoseconds(window_duration_seconds)
    clouds, frame, times, bag_start_ns, lower, upper = load_cloud_window(
        bag_folder, timestamp_nanoseconds(timestamp_seconds), window_ns, time_base
    )
    poses, pose_times = load_gnss_window(bag_folder, lower, upper)
    gnss_path_poses, gnss_path_times = load_gnss_window(
        bag_folder,
        bag_start_ns + timestamp_nanoseconds("40"),
        bag_start_ns + timestamp_nanoseconds("50"),
    )
    gnss_path = np.asarray([
        [pose.transform.translation.x, pose.transform.translation.y]
        for pose in gnss_path_poses
    ])
    xyz, selected_times = cloud_window_in_world(clouds, times, frame, poses, pose_times)
    gnss_pose = poses[int(np.argmin(np.abs(pose_times - upper)))]
    gnss = gnss_pose.transform.translation
    ends, _, orientations = load_nearest_end_of_row(bag_folder, upper)
    forward_points = cloud_in_world(
        np.array([[0., 0., 0.], [1., 0., 0.]]), "rearAxle", gnss_pose
    )
    forward = forward_points[1, :2] - forward_points[0, :2]
    if np.linalg.norm(forward) < 1e-8:
        raise ValueError("GNSS forward axis cannot be projected onto the XY plot.")
    gnss_heading = float(np.arctan2(forward[1], forward[0]))
    result = process_world_cloud(xyz, gnss, ends, orientations, gnss_heading)
    print(f"Point clouds: {len(clouds)}; segments: {len(result['segments'])}; "
        f"starts: {len(result['segment_starts'])}")
    print(f"Maximum cloud-to-GNSS offset: "
        f"{np.max(np.asarray(times) - selected_times) / NANOSECONDS:.3f} s")
    for index, point, neighbors, density in result["segment_starts"]:
        print(f"Segment {index} start: {point}; neighbors: {neighbors}; density passed: {density}")
    transform = result["start_midpoint_transform"]
    if transform is not None:
        print("Start-to-midpoint XY transform:")
        print(transform["matrix"])
        print(f"Heading: {transform['heading_error_deg']:+.3f} deg; "
            f"X: {transform['x_error_m']:+.3f} m; Y: {transform['y_error_m']:+.3f} m; "
            f"RMSE: {transform['rmse']:.3f} m")
    transformed_ends = result["transformed_end_of_row_xyz"]
    minimum_error_m = None
    nearest_path_xy = None
    nearest_transformed_end_xy = None
    if transformed_ends is not None and len(transformed_ends) and len(gnss_path):
        distances, nearest_end_indices = cKDTree(transformed_ends[:, :2]).query(gnss_path[:, :2])
        nearest_path_index = int(np.argmin(distances))
        nearest_end_index = int(nearest_end_indices[nearest_path_index])
        minimum_error_m = float(distances[nearest_path_index])
        nearest_path_xy = gnss_path[nearest_path_index, :2]
        nearest_transformed_end_xy = transformed_ends[nearest_end_index, :2]
        nearest_path_time_s = (
            gnss_path_times[nearest_path_index] - bag_start_ns
        ) / NANOSECONDS
        print(f"Minimum GNSS-path-to-transformed-end-of-row distance: "
            f"{minimum_error_m:.3f} m at t={nearest_path_time_s:.3f} s")
        print(f"Nearest XY pair (GNSS -> transformed end-of-row): "
            f"{nearest_path_xy} -> {nearest_transformed_end_xy}")
    else:
        print("Minimum GNSS-path-to-transformed-end-of-row distance unavailable.")


    ##########################################################################################
    ################################### START OF PLOTTING #####################################
    ############################################################################################  

    figure = plt.figure(figsize=(12, 12), constrained_layout=True)
    figure.set_layout_engine(None)
    axes = figure.add_axes([0.08, 0.09, 0.78, 0.84])
    colorbar_axes = figure.add_axes([0.89, 0.09, 0.02, 0.84])

    # pcl segmentation plot
    xyz = result["xyz_world"]
    ids = result["segment_ids"]
    segmented = ids >= 0
    segment_count = len(result["segments"])
    scatter = axes.scatter(
        xyz[segmented, 0], xyz[segmented, 1], c=ids[segmented], s=3,
        cmap=plt.get_cmap("tab20", segment_count), vmin=-0.5, vmax=segment_count - 0.5,
        linewidths=0, rasterized=True,
    )
    if np.any(~segmented):
        axes.scatter(
            xyz[~segmented, 0], xyz[~segmented, 1], s=1, color="dimgray",
            linewidths=0, label="Excluded by end-of-row or segment FOV limit",
        )
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("World X [m]")
    axes.set_ylabel("World Y [m]")
    axes.set_title(
        f"{bag_topic} + {end_of_row_topic} | {timestamp_seconds} s end, "
        f"{window_duration_seconds} s window | frame: world"
    )
    axes.grid(True, alpha=0.25)
    figure.colorbar(scatter, cax=colorbar_axes, ticks=range(segment_count), label="End-of-row segment")
    axes.legend(loc="upper right", fontsize=8)
    # axes.set_xlim(gnss.x - 15.0 - 0.0, gnss.x + 15.0 - 0.0)
    # axes.set_ylim(gnss.y - 15.0 - 0.0, gnss.y + 15.0 - 0.0)
    axes.set_xlim(gnss.x - 15.0 - 0.0, gnss.x + 15.0 - 0.0)
    axes.set_ylim(gnss.y - 15.0 - 10.0, gnss.y + 15.0 - 10.0)

    axes.plot(
        gnss_path[:, 0], gnss_path[:, 1], color="white", linewidth=1.5,
        zorder=7, label="GNSS path (40-50 s)",
    )
    if minimum_error_m is not None:
        axes.plot(
            [nearest_path_xy[0], nearest_transformed_end_xy[0]],
            [nearest_path_xy[1], nearest_transformed_end_xy[1]],
            color="gold", linestyle=":", linewidth=2, zorder=9,
            label=f"Minimum separation: {minimum_error_m:.2f} m",
        )
    x_limits, y_limits = axes.get_xlim(), axes.get_ylim()
    axes.set_xlim(
        min(x_limits[0], gnss_path[:, 0].min() - 1.0),
        max(x_limits[1], gnss_path[:, 0].max() + 1.0),
    )
    axes.set_ylim(
        min(y_limits[0], gnss_path[:, 1].min() - 1.0),
        max(y_limits[1], gnss_path[:, 1].max() + 1.0),
    )

    # GNSS and LiDAR origin plot
    gnss = result["gnss_position"]
    axes.scatter([gnss.x], [gnss.y], marker="*", s=500, color="white",
                edgecolors="black", zorder=4, label="rearAxleGNSS at cloud time")
    heading = result["gnss_heading"]

    # EoR segment starts and end-of-row points
    ends = result["end_of_row_xyz"]
    for points, marker, color, label in (
        (ends, "+", "darkorange", "Extended end-of-row locations"),
        (result["recorded_end_of_row_xyz"], "X", "darkorange", end_of_row_topic),
        (result["midpoints"], "D", "cyan", "Midpoints between end-of-row points"),
    ):
        if len(points):
            axes.scatter(points[:, 0], points[:, 1], marker=marker, s=65,
                        color=color, zorder=5, label=label)
    transformed_midpoints = result["transformed_midpoints"]
    if transformed_midpoints is not None and len(transformed_midpoints):
        axes.scatter(
            transformed_midpoints[:, 0], transformed_midpoints[:, 1], marker="o", s=65,
            color="magenta", zorder=6, label="Transformed midpoints",
        )
    transformed_ends = result["transformed_end_of_row_xyz"]
    if transformed_ends is not None and len(transformed_ends):
        axes.scatter(
            transformed_ends[:, 0], transformed_ends[:, 1], marker="^", s=65,
            color="lime", zorder=6, label="Transformed end-of-row locations",
        )
    axes.legend(loc="upper right", fontsize=8)

    plt.show()

    #%%
    # LiDAR field of view geometry
    lidar_origin, fov_rays = lidar_fov_geometry(gnss, heading, lidar_fov_angle_deg)
    lidar_offset = lidar_origin - np.array([gnss.x, gnss.y])
    axes.scatter(
        [lidar_origin[0]], [lidar_origin[1]], marker="o", s=40, color="cyan",
        zorder=6, label="LiDAR origin (1 m forward of GNSS)",
    )
    for index, ray in enumerate(fov_rays):
        length = min(
            (15.0 - np.sign(component) * displacement) / abs(component)
            for component, displacement in zip(ray, lidar_offset) if abs(component) > 1e-12
        )
        endpoint = lidar_origin + length * ray
        axes.plot(
            [lidar_origin[0], endpoint[0]], [lidar_origin[1], endpoint[1]], color="cyan",
            linestyle="-", linewidth=2, zorder=5,
            label=f"LiDAR {lidar_fov_angle_deg:g} deg field of view" if index == 0 else None,
        )
    axes.legend(loc="upper right", fontsize=8)

    # End-of-row limits
    origin, direction = result["origin"], result["direction"]
    normal = np.array([-direction[1], direction[0]])
    if len(ends) >= 2:
        order = np.argsort((ends[:, :2] - origin) @ normal, kind="stable")
        end_line = ends[order, :2]
        for offset, color, label in (
            (-start_point_max_end_of_row_distance, "lime",
             f"Before end-of-row: {start_point_max_end_of_row_distance:g} m (start-distance guide)"),
            (max_distance_after_end_of_row, "orangered",
             f"After end-of-row: {max_distance_after_end_of_row:g} m (cloud cutoff)"),
        ):
            limit_line = end_line + offset * direction
            axes.plot(
                limit_line[:, 0], limit_line[:, 1], color=color, linestyle="--",
                linewidth=1, zorder=5, label=label,
            )
    axes.legend(loc="upper right", fontsize=8)

    # end of row x axes
    all_xy = np.vstack((xyz[:, :2], ends[:, :2]))
    if len(all_xy):
        along = (all_xy - origin) @ direction
        extent = np.array([along.min() - 1, along.max() + 1])
        for index, boundary in enumerate(result["boundaries"]):
            line = origin + boundary * normal + extent[:, None] * direction
            axes.plot(line[:, 0], line[:, 1], color="gray", linestyle="--",
                    linewidth=1, alpha=1.0,
                    label="End-of-row local X-axis" if index == 0 else None)
    axes.legend(loc="upper right", fontsize=8)


    # RANSAC individual line fits and rejected points
    for key, color, style, width, label in (
        ("individual_fits", "deepskyblue", "--", 1.5, "Individual RANSAC fits"),
        ("common_fits", "yellow", "-", 2.5, "Point-weighted common-heading fits"),
    ):
        if key=="individual_fits":
            labeled = False
            for fit in result[key][:result["fit_count"]]:
                if fit is not None:
                    line = fit["line_xy"]
                    axes.plot(line[:, 0], line[:, 1], color="red", linestyle="-",
                            linewidth=3, zorder=4, label=None if labeled else label)
                    labeled = True

    rejected = result["ransac_outliers"]
    if len(rejected):
        axes.scatter(rejected[:, 0], rejected[:, 1], s=1, color="red",alpha=0.6, zorder=4,
                    label="Rejected points")
    axes.legend(loc="upper right", fontsize=8)


    # Common RANSAC
    for key, color, style, width, label in (
        ("individual_fits", "deepskyblue", "--", 1.5, "Individual RANSAC fits"),
        ("common_fits", "yellow", "-", 2.5, "Point-weighted common-heading fits"),
    ):
        if key=="common_fits":
            labeled = False
            for fit in result[key][:result["fit_count"]]:
                if fit is not None:
                    line = fit["line_xy"]
                    axes.plot(line[:, 0], line[:, 1], color="yellow", linestyle="-",
                            linewidth=3, zorder=4, label=None if labeled else label)
                    labeled = True
    axes.legend(loc="upper right", fontsize=8)


    # start of vines
    for density, color, marker, label in (
        (True, "lime", "P", "Segment starts passing density filter"),
        (False, "orange", "X", "Densest-point fallback starts"),
    ):
        selected = [entry for entry in result["segment_starts"] if entry[3] == density]
        if selected:
            points = np.asarray([entry[1] for entry in selected])
            axes.scatter(points[:, 0], points[:, 1], marker=marker, s=110, color=color,
                        edgecolors="black", zorder=6, label=label)
            for index, point, _, _ in selected:
                axes.annotate(str(index), point[:2], xytext=(5, 5),
                            textcoords="offset points", color=color, zorder=7)
    axes.legend(loc="upper right", fontsize=8)

  
    # start line fit
    start_line = result["start_line_fit"]
    if start_line is not None:
        line = start_line["line_xy"]
        axes.plot(line[:, 0], line[:, 1], color="magenta", linestyle="-.", linewidth=2.5,
                zorder=5, label="Robust least-squares start line (at most one excluded)")




    # midpoints derived from EoR
    transform = result["start_midpoint_transform"]
    if transform is not None:
        for start, midpoint in zip(transform["start_points"], transform["midpoint_points"]):
            axes.annotate("", xy=midpoint, xytext=start,
                        arrowprops={"arrowstyle": "->", "color": "white", "linestyle": ":"},
                        zorder=7)
        axes.plot([], [], color="white", linestyle=":", label="Start-to-midpoint matches")
        status = (
            f"Heading: {transform['heading_error_deg']:+.3f} deg\n"
            f"X: {transform['x_error_m']:+.3f} m, Y: {transform['y_error_m']:+.3f} m\n"
            f"Fit RMSE: {transform['rmse']:.3f} m"
        )
    else:
        status = "No start-to-midpoint transform (insufficient valid starts/midpoints)"
    axes.text(0.02, 0.02, status, transform=axes.transAxes, va="bottom",
            bbox={"facecolor": "black", "alpha": 0.7}, zorder=8)



    if save_path is not None:
        figure.savefig(save_path)
    if show_plot:
        plt.show()