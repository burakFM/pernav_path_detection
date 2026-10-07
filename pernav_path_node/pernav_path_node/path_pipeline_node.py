## TODO:
# [] - Check the frame of entrance line values
# [] - Talk with Joram about the message type entrance line and path message type
# [] - Ask Sam to provide node starting flag and ending flag for the path detection
# [] - Ask Sam to provide the estimated trajectory of the U turn
# [] - Write a sample subscriber to the check lidar data and the path detection output. 
# (maybe check the synchronization)
# [] - Check output data frame

#----------------------------------------------------------------------------------------#


import math
import struct
from collections import OrderedDict
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Empty, Float64
from visualization_msgs.msg import Marker, MarkerArray

from .path_start_tracker import PathStartTracker
from .start_line_tracker import StartLineTracker
from .pipeline_helpers import (  # type: ignore[reportMissingImports]
    apply_parallel_correction,
    build_group_start_lines,
    build_path_groups,
    build_paths_from_rows,
    detect_rows_from_xy,
)


class PathPipelineNode(Node):
    def __init__(self) -> None:
        super().__init__('path_pipeline_node')
        self.frame_count = 0

        # Centralized node parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('input_topic', '/pcl_world_roi'),
                ('row_start_ref_topic', '/pernav/row_start_ref_world'),
                ('row_start_ref_queue_size', 100),
                ('roi_axes_marker_topic', '/roi_markers_axes'),
                ('enable_roi_strip_detection', True),
                ('enable_fov_boundary_row_filter', False),
                ('fov_boundary_marker_topic', '/pernav/lidar_fov_marker_world'),
                ('fov_boundary_min_distance_m', 1.0),
                ('enable_discarded_fov_row_marker_publisher', True),
                ('discarded_fov_row_marker_topic', '/pernav/discarded_fov_row_markers'),
                ('enable_chassis_front_path_distance_publisher', True),
                ('chassis_front_marker_world_topic', '/pernav/chassis_front_marker_world'),
                ('chassis_front_path_distance_topic', '/pernav/chassis_front_to_path_entrance_distance'),
                ('chassis_front_path_distance_marker_topic', '/pernav/chassis_front_path_distance_marker'),
                ('chassis_front_path_distance_markers_topic', '/pernav/chassis_front_path_distance_markers'),
                ('enable_geometric_path_experiment', True),
                ('geometric_path_length', 3.0),
                ('geometric_path_output_topic', '/pernav/geometric_paths'),
                ('geometric_marker_output_topic', '/pernav/geometric_path_markers'),
                ('enable_row_detection', True),
                ('row_distance_threshold', 0.2),
                ('row_max_gap', 0.6),
                ('row_max_iterations', 400),
                ('row_min_segment_inliers', 40),
                ('row_max_rows', 10),
                ('row_remove_radius', 1.9),
                ('row_min_points_left', 30),
                ('enable_path_detection', True),
                ('min_path_length', 1.0),
                ('enable_path_start_tracking', True),
                ('tracking_max_distance', 1.0),
                ('tracking_max_heading_deg', 20.0),
                ('tracking_min_observations', 3),
                ('tracking_candidate_max_missed_scans', 5),
                ('tracking_ambiguity_margin', 0.15),
                ('enable_parallel_correction', True),
                ('group_angle_thresh_deg', 12.0),
                ('group_lateral_thresh', 5.0),
                ('group_midpoint_thresh', 8.0),
                ('enable_path_publisher', True),
                ('path_output_topic', '/pernav/paths'),
                ('path_sensor_id', 'pernav_path_node'),
                ('enable_marker_publisher', True),
                ('marker_output_topic', '/pernav/path_markers'),
                ('enable_row_marker_publisher', True),
                ('row_marker_output_topic', '/pernav/detected_row_markers'),
                ('marker_line_width', 0.08),
                ('marker_lifetime_sec', 0.2),
                ('enable_notebook_plot', False),
                ('enable_plot_video', False),
                ('plot_video_output_path', 'output/pernav_path_plot_{timestamp}.mp4'),
                ('plot_video_fps', 1.0),
                ('plot_every_n_frames', 1),
                ('enable_plot_autoscale', False),
                ('plot_padding_m', 1.0),
                ('plot_x_min', 0.0),
                ('plot_x_max', 20.0),
                ('plot_y_min', -10.0),
                ('plot_y_max', 10.0),
                ('start_line_extension_m', 2.0),
                ('start_line_eor_marker_topic', '/roi_markers'),
                ('start_line_residual_threshold', 0.5),
                ('start_line_min_anchors', 2),
                ('start_line_switch_confirmations', 5),
                ('enable_group_start_line_marker', True),
                ('enable_group_start_distance_publisher', True),
                ('group_start_distance_topic', '/pernav/group_start_line_distance'),
                ('enable_group_start_distance_stop', True),
                ('group_start_distance_stop_threshold', 0.2),
                ('path_pipeline_reset_topic', '/pernav/path_pipeline_reset'),
            ],
        )

        self.input_topic = self.get_parameter('input_topic').value
        self.row_start_ref_topic = str(self.get_parameter('row_start_ref_topic').value)
        self.row_start_ref_queue_size = max(1, int(self.get_parameter('row_start_ref_queue_size').value))
        self.roi_axes_marker_topic = str(self.get_parameter('roi_axes_marker_topic').value)
        self.enable_roi_strip_detection = bool(
            self.get_parameter('enable_roi_strip_detection').value
        )
        self._pending_clouds: OrderedDict[tuple[int, int], PointCloud2] = OrderedDict()
        self._pending_references: OrderedDict[tuple[int, int], PointStamped] = OrderedDict()
        self._pending_fov_markers: OrderedDict[tuple[int, int], Marker] = OrderedDict()
        self._pending_chassis_front_markers: OrderedDict[tuple[int, int], Marker] = OrderedDict()
        self._roi_strip_normal: np.ndarray | None = None
        self._roi_strip_boundaries: np.ndarray | None = None
        self._roi_driving_direction: np.ndarray | None = None
        self.enable_fov_boundary_row_filter = bool(
            self.get_parameter('enable_fov_boundary_row_filter').value
        )
        self.fov_boundary_marker_topic = str(
            self.get_parameter('fov_boundary_marker_topic').value
        )
        self.fov_boundary_min_distance_m = float(
            self.get_parameter('fov_boundary_min_distance_m').value
        )
        self.enable_discarded_fov_row_marker_publisher = bool(
            self.get_parameter('enable_discarded_fov_row_marker_publisher').value
        )
        self.discarded_fov_row_marker_topic = str(
            self.get_parameter('discarded_fov_row_marker_topic').value
        )
        if (
            not math.isfinite(self.fov_boundary_min_distance_m)
            or self.fov_boundary_min_distance_m < 0.0
        ):
            raise ValueError('FOV boundary minimum distance must be finite and nonnegative.')
        self.enable_chassis_front_path_distance_publisher = bool(
            self.get_parameter('enable_chassis_front_path_distance_publisher').value
        )
        self.chassis_front_marker_world_topic = str(
            self.get_parameter('chassis_front_marker_world_topic').value
        )
        self.chassis_front_path_distance_topic = str(
            self.get_parameter('chassis_front_path_distance_topic').value
        )
        self.chassis_front_path_distance_marker_topic = str(
            self.get_parameter('chassis_front_path_distance_marker_topic').value
        )
        self.chassis_front_path_distance_markers_topic = str(
            self.get_parameter('chassis_front_path_distance_markers_topic').value
        )
        self.enable_geometric_path_experiment = bool(
            self.get_parameter('enable_geometric_path_experiment').value
        )
        self.geometric_path_length = float(self.get_parameter('geometric_path_length').value)
        self.geometric_path_output_topic = str(
            self.get_parameter('geometric_path_output_topic').value
        )
        self.geometric_marker_output_topic = str(
            self.get_parameter('geometric_marker_output_topic').value
        )
        if not math.isfinite(self.geometric_path_length) or self.geometric_path_length <= 0.0:
            raise ValueError('Geometric path length must be finite and positive.')
        self.enable_row_detection = bool(self.get_parameter('enable_row_detection').value)
        self.row_distance_threshold = float(self.get_parameter('row_distance_threshold').value)
        self.row_max_gap = float(self.get_parameter('row_max_gap').value)
        self.row_max_iterations = int(self.get_parameter('row_max_iterations').value)
        self.row_min_segment_inliers = int(self.get_parameter('row_min_segment_inliers').value)
        self.row_max_rows = int(self.get_parameter('row_max_rows').value)
        self.row_remove_radius = float(self.get_parameter('row_remove_radius').value)
        self.row_min_points_left = int(self.get_parameter('row_min_points_left').value)
        self.enable_path_detection = bool(self.get_parameter('enable_path_detection').value)
        self.min_path_length = float(self.get_parameter('min_path_length').value)
        self.enable_path_start_tracking = bool(self.get_parameter('enable_path_start_tracking').value)
        self.path_start_tracker = PathStartTracker(
            max_distance=float(self.get_parameter('tracking_max_distance').value),
            max_heading_deg=float(self.get_parameter('tracking_max_heading_deg').value),
            min_observations=int(self.get_parameter('tracking_min_observations').value),
            candidate_max_missed_scans=int(self.get_parameter('tracking_candidate_max_missed_scans').value),
            ambiguity_margin=float(self.get_parameter('tracking_ambiguity_margin').value),
        )
        self.geometric_path_start_tracker = PathStartTracker(
            max_distance=float(self.get_parameter('tracking_max_distance').value),
            max_heading_deg=float(self.get_parameter('tracking_max_heading_deg').value),
            min_observations=int(self.get_parameter('tracking_min_observations').value),
            candidate_max_missed_scans=int(
                self.get_parameter('tracking_candidate_max_missed_scans').value
            ),
            ambiguity_margin=float(self.get_parameter('tracking_ambiguity_margin').value),
        )
        self.enable_parallel_correction = bool(self.get_parameter('enable_parallel_correction').value)
        self.group_angle_thresh_deg = float(self.get_parameter('group_angle_thresh_deg').value)
        self.group_lateral_thresh = float(self.get_parameter('group_lateral_thresh').value)
        self.group_midpoint_thresh = float(self.get_parameter('group_midpoint_thresh').value)
        self.enable_path_publisher = bool(self.get_parameter('enable_path_publisher').value)
        self.path_output_topic = str(self.get_parameter('path_output_topic').value)
        self.path_sensor_id = str(self.get_parameter('path_sensor_id').value)
        self.enable_marker_publisher = bool(self.get_parameter('enable_marker_publisher').value)
        self.marker_output_topic = str(self.get_parameter('marker_output_topic').value)
        self.enable_row_marker_publisher = bool(
            self.get_parameter('enable_row_marker_publisher').value
        )
        self.row_marker_output_topic = str(self.get_parameter('row_marker_output_topic').value)
        self.marker_line_width = float(self.get_parameter('marker_line_width').value)
        self.marker_lifetime_sec = float(self.get_parameter('marker_lifetime_sec').value)
        self.enable_notebook_plot = bool(self.get_parameter('enable_notebook_plot').value)
        self.enable_plot_video = bool(self.get_parameter('enable_plot_video').value)
        self.plot_video_output_path = str(self.get_parameter('plot_video_output_path').value)
        self.plot_video_fps = float(self.get_parameter('plot_video_fps').value)
        self.plot_every_n_frames = max(1, int(self.get_parameter('plot_every_n_frames').value))
        self.enable_plot_autoscale = bool(self.get_parameter('enable_plot_autoscale').value)
        self.plot_padding_m = max(0.0, float(self.get_parameter('plot_padding_m').value))
        self.plot_x_min = float(self.get_parameter('plot_x_min').value)
        self.plot_x_max = float(self.get_parameter('plot_x_max').value)
        self.plot_y_min = float(self.get_parameter('plot_y_min').value)
        self.plot_y_max = float(self.get_parameter('plot_y_max').value)
        self.start_line_extension_m = float(self.get_parameter('start_line_extension_m').value)
        self.start_line_eor_marker_topic = str(
            self.get_parameter('start_line_eor_marker_topic').value
        )
        self.start_line_tracker = StartLineTracker(
            residual_threshold=float(self.get_parameter('start_line_residual_threshold').value),
            min_anchors=int(self.get_parameter('start_line_min_anchors').value),
            switch_confirmations=int(self.get_parameter('start_line_switch_confirmations').value),
            extension_m=self.start_line_extension_m,
            max_eor_angle_deg=float(self.get_parameter('group_angle_thresh_deg').value),
        )
        self.geometric_start_line_tracker = StartLineTracker(
            residual_threshold=float(self.get_parameter('start_line_residual_threshold').value),
            min_anchors=int(self.get_parameter('start_line_min_anchors').value),
            switch_confirmations=int(self.get_parameter('start_line_switch_confirmations').value),
            extension_m=self.start_line_extension_m,
            max_eor_angle_deg=float(self.get_parameter('group_angle_thresh_deg').value),
        )
        self._eor_line_reference: np.ndarray | None = None
        self._eor_line_direction: np.ndarray | None = None
        self.enable_group_start_line_marker = bool(self.get_parameter('enable_group_start_line_marker').value)
        self.enable_group_start_distance_publisher = bool(
            self.get_parameter('enable_group_start_distance_publisher').value
        )
        self.group_start_distance_topic = str(self.get_parameter('group_start_distance_topic').value)
        self.enable_group_start_distance_stop = bool(
            self.get_parameter('enable_group_start_distance_stop').value
        )
        self.group_start_distance_stop_threshold = float(
            self.get_parameter('group_start_distance_stop_threshold').value
        )
        self.path_pipeline_reset_topic = str(
            self.get_parameter('path_pipeline_reset_topic').value
        )
        if (
            not math.isfinite(self.group_start_distance_stop_threshold)
            or self.group_start_distance_stop_threshold < 0.0
        ):
            raise ValueError('Group-start distance stop threshold must be finite and nonnegative.')
        self._retained_group_start_lines: dict[int, dict] = {}
        self._last_group_start_distance_stamp_ns: int | None = None
        self._previous_group_start_distance: float | None = None
        self._row_path_detection_stopped = False
        self._last_confirmed_output_paths: list[dict] = []
        self._last_confirmed_group_start_lines: list[dict] = []

        self._plt: Any = None
        self._fig: Any = None
        self._ax: Any = None
        self._video_recorder: Any = None

        self.path_publisher = self.create_publisher(PoseArray, self.path_output_topic, 10)
        self.marker_publisher = self.create_publisher(MarkerArray, self.marker_output_topic, 10)
        self.geometric_path_publisher = self.create_publisher(
            PoseArray, self.geometric_path_output_topic, 10,
        )
        self.geometric_marker_publisher = self.create_publisher(
            MarkerArray, self.geometric_marker_output_topic, 10,
        )
        self.row_marker_publisher = self.create_publisher(
            MarkerArray, self.row_marker_output_topic, 10,
        )
        self.discarded_fov_row_marker_publisher = self.create_publisher(
            MarkerArray, self.discarded_fov_row_marker_topic, 10,
        )
        self.group_start_distance_publisher = self.create_publisher(
            Float64, self.group_start_distance_topic, 10,
        )
        self.chassis_front_path_distance_publisher = self.create_publisher(
            Float64, self.chassis_front_path_distance_topic, 10,
        )
        self.chassis_front_path_distance_marker_publisher = self.create_publisher(
            Marker, self.chassis_front_path_distance_marker_topic, 10,
        )
        self.chassis_front_path_distance_markers_publisher = self.create_publisher(
            MarkerArray, self.chassis_front_path_distance_markers_topic, 10,
        )

        if self.enable_notebook_plot or self.enable_plot_video:
            self._init_notebook_plot()

        self.subscription = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.listener_callback,
            10,
        )
        self.subscription
        self.row_start_ref_subscription = self.create_subscription(
            PointStamped,
            self.row_start_ref_topic,
            self.row_start_ref_callback,
            10,
        )
        self.path_pipeline_reset_subscription = self.create_subscription(
            Empty,
            self.path_pipeline_reset_topic,
            self.path_pipeline_reset_callback,
            10,
        )
        self.eor_line_subscription = self.create_subscription(
            Marker,
            self.start_line_eor_marker_topic,
            self.eor_line_marker_callback,
            10,
        )
        self.roi_axes_subscription = self.create_subscription(
            Marker,
            self.roi_axes_marker_topic,
            self.roi_axes_marker_callback,
            10,
        )
        self.fov_boundary_marker_subscription = self.create_subscription(
            Marker,
            self.fov_boundary_marker_topic,
            self.fov_boundary_marker_callback,
            10,
        )
        self.chassis_front_marker_subscription = self.create_subscription(
            Marker,
            self.chassis_front_marker_world_topic,
            self.chassis_front_marker_callback,
            10,
        )
        self.get_logger().info(
            f'Path pipeline node listening on topic {self.input_topic} '
            f'| synchronized_reference={self.row_start_ref_topic} '
            f'| row_detection={self.enable_row_detection} '
            f'| roi_strip_detection={self.enable_roi_strip_detection} '
            f'({self.roi_axes_marker_topic}) '
            f'| fov_boundary_row_filter={self.enable_fov_boundary_row_filter} '
            f'({self.fov_boundary_marker_topic}, minimum={self.fov_boundary_min_distance_m} m) '
            f'| chassis_front_path_distance={self.enable_chassis_front_path_distance_publisher} '
            f'({self.chassis_front_path_distance_topic}) '
            f'| geometric_path_experiment={self.enable_geometric_path_experiment} '
            f'({self.geometric_path_output_topic}) '
            f'| path_detection={self.enable_path_detection} '
            f'| path_start_tracking={self.enable_path_start_tracking} '
            f'| parallel_correction={self.enable_parallel_correction} '
            f'| path_pub={self.enable_path_publisher} ({self.path_output_topic}) '
            f'| marker_pub={self.enable_marker_publisher} ({self.marker_output_topic}) '
            f'| row_marker_pub={self.enable_row_marker_publisher} '
            f'({self.row_marker_output_topic}) '
            f'| notebook_plot={self.enable_notebook_plot} '
            f'| plot_video={self.enable_plot_video} '
            f'| plot_autoscale={self.enable_plot_autoscale} '
            f'| group_start_line_marker={self.enable_group_start_line_marker} '
            f'| eor_line={self.start_line_eor_marker_topic} '
            f'| group_start_distance_pub={self.enable_group_start_distance_publisher} '
            f'({self.group_start_distance_topic}) '
            f'| group_start_distance_stop={self.enable_group_start_distance_stop} '
            f'(falling threshold={self.group_start_distance_stop_threshold} m) '
            f'| cycle_reset={self.path_pipeline_reset_topic}'
        )

    @staticmethod
    def _point_to_segment_distance(point_xy: np.ndarray, line: dict) -> float:
        """Return the shortest XY distance from a point to a finite line segment."""
        point = np.asarray(point_xy, dtype=float)
        p0 = np.array([line['start_x'], line['start_y']], dtype=float)
        p1 = np.array([line['end_x'], line['end_y']], dtype=float)
        segment = p1 - p0
        length_squared = float(segment @ segment)
        if length_squared <= 1e-12:
            return float(np.linalg.norm(point - p0))
        projection = float((point - p0) @ segment) / length_squared
        closest = p0 + np.clip(projection, 0.0, 1.0) * segment
        return float(np.linalg.norm(point - closest))

    @staticmethod
    def _point_to_ray_distance(point_xy: np.ndarray, origin_xy: np.ndarray, endpoint_xy: np.ndarray) -> float:
        """Return shortest XY distance from a point to an outward FOV boundary ray."""
        direction = endpoint_xy - origin_xy
        length_squared = float(direction @ direction)
        if length_squared <= 1e-12:
            return float(np.linalg.norm(point_xy - origin_xy))
        progress = max(0.0, float((point_xy - origin_xy) @ direction) / length_squared)
        return float(np.linalg.norm(point_xy - (origin_xy + progress * direction)))

    def _filter_rows_by_fov_boundary(
        self, row_records: list[dict], fov_marker: Marker,
    ) -> tuple[list[dict], list[dict]]:
        """Remove rows whose EOR-oriented start is too close to either FOV boundary ray."""
        if fov_marker.type != Marker.LINE_LIST or len(fov_marker.points) < 4:
            self.get_logger().warning('Ignoring malformed world FOV boundary marker.')
            return row_records, []
        rays = []
        for index in range(0, len(fov_marker.points) - 1, 2):
            origin = fov_marker.points[index]
            endpoint = fov_marker.points[index + 1]
            rays.append((
                np.array([origin.x, origin.y], dtype=float),
                np.array([endpoint.x, endpoint.y], dtype=float),
            ))
        accepted: list[dict] = []
        discarded: list[dict] = []
        for row in row_records:
            start = np.array([row['start_x'], row['start_y']], dtype=float)
            distance = min(
                self._point_to_ray_distance(start, origin, endpoint)
                for origin, endpoint in rays
            )
            if distance < self.fov_boundary_min_distance_m:
                rejected = dict(row)
                rejected['fov_boundary_distance'] = distance
                discarded.append(rejected)
            else:
                accepted.append(row)
        return accepted, discarded

    def _publish_group_start_distance(
        self,
        reference: PointStamped,
        current_lines: list[dict],
        stamp_ns: int,
    ) -> float | None:
        """Retain group lines and publish distance to the nearest retained segment."""
        self._prepare_group_start_distance_stamp(stamp_ns)

        for line in current_lines:
            values = np.array(
                [line['start_x'], line['start_y'], line['end_x'], line['end_y']], dtype=float,
            )
            if np.isfinite(values).all():
                self._retained_group_start_lines[int(line['group_id'])] = dict(line)

        if not self._retained_group_start_lines:
            return None

        reference_xy = np.array([reference.point.x, reference.point.y], dtype=float)
        distance = min(
            self._point_to_segment_distance(reference_xy, line)
            for line in self._retained_group_start_lines.values()
        )
        if self.enable_group_start_distance_publisher:
            msg = Float64()
            msg.data = distance
            self.group_start_distance_publisher.publish(msg)
        self._update_group_start_distance_stop(distance)
        return distance

    def _prepare_group_start_distance_stamp(self, stamp_ns: int) -> None:
        """Reset retained and latched state when a new bag rewinds time."""
        if (
            self._last_group_start_distance_stamp_ns is not None
            and stamp_ns < self._last_group_start_distance_stamp_ns
        ):
            PathPipelineNode._clear_path_pipeline_cycle_state(self)
        self._last_group_start_distance_stamp_ns = stamp_ns

    def _clear_path_pipeline_cycle_state(self) -> None:
        self._retained_group_start_lines.clear()
        self._last_group_start_distance_stamp_ns = None
        self._previous_group_start_distance = None
        self._row_path_detection_stopped = False
        self._last_confirmed_output_paths.clear()
        self._last_confirmed_group_start_lines.clear()
        self.path_start_tracker.reset()
        self.start_line_tracker.reset()
        self.geometric_path_start_tracker.reset()
        self.geometric_start_line_tracker.reset()
        self._eor_line_reference = None
        self._eor_line_direction = None
        self._roi_strip_normal = None
        self._roi_strip_boundaries = None
        self._roi_driving_direction = None

    def eor_line_marker_callback(self, marker: Marker) -> None:
        """Store the frozen ROI centerline direction as the start-line orientation prior."""
        if marker.action == Marker.DELETE or marker.action == Marker.DELETEALL:
            self._eor_line_reference = None
            self._eor_line_direction = None
            return
        if marker.ns != 'world_roi' or marker.type != Marker.LINE_LIST or len(marker.points) < 2:
            return
        direction = np.array([
            marker.points[1].x - marker.points[0].x,
            marker.points[1].y - marker.points[0].y,
        ], dtype=float)
        length = float(np.linalg.norm(direction))
        if np.isfinite(direction).all() and length > 1e-9:
            self._eor_line_reference = np.array(
                [marker.points[0].x, marker.points[0].y], dtype=float,
            )
            self._eor_line_direction = direction / length

    def roi_axes_marker_callback(self, marker: Marker) -> None:
        """Cache ordered infinite-line coordinates for ROI strip partitioning."""
        if marker.action == Marker.DELETE or marker.action == Marker.DELETEALL:
            self._roi_strip_normal = None
            self._roi_strip_boundaries = None
            self._roi_driving_direction = None
            return
        if (
            marker.ns != 'world_roi_axes'
            or marker.type != Marker.LINE_LIST
            or len(marker.points) < 4
            or len(marker.points) % 2 != 0
        ):
            return
        first_start = np.array([marker.points[0].x, marker.points[0].y], dtype=float)
        first_end = np.array([marker.points[1].x, marker.points[1].y], dtype=float)
        direction = first_end - first_start
        direction_length = float(np.linalg.norm(direction))
        if not np.isfinite(direction).all() or direction_length <= 1e-9:
            return
        direction /= direction_length
        normal = np.array([-direction[1], direction[0]], dtype=float)
        midpoints = np.array([
            [
                0.5 * (marker.points[index].x + marker.points[index + 1].x),
                0.5 * (marker.points[index].y + marker.points[index + 1].y),
            ]
            for index in range(0, len(marker.points), 2)
        ], dtype=float)
        boundaries = np.unique(np.round(midpoints @ normal, decimals=9))
        if boundaries.size < 2:
            return
        self._roi_strip_normal = normal
        self._roi_strip_boundaries = np.sort(boundaries)
        self._roi_driving_direction = direction

    def _build_geometric_paths(self, xy_roi: np.ndarray) -> tuple[list[np.ndarray], list[dict]]:
        """Build fixed-length reverse-driving paths from EOR-nearest strip points."""
        if (
            not self.enable_geometric_path_experiment
            or self._roi_strip_normal is None
            or self._roi_strip_boundaries is None
            or self._roi_driving_direction is None
            or self._eor_line_reference is None
            or self._eor_line_direction is None
        ):
            return [], []

        coordinates = xy_roi @ self._roi_strip_normal
        eor_normal = np.array(
            [-self._eor_line_direction[1], self._eor_line_direction[0]], dtype=float,
        )
        row_starts: list[np.ndarray] = []
        for lower, upper in zip(self._roi_strip_boundaries[:-1], self._roi_strip_boundaries[1:]):
            region = xy_roi[(coordinates >= lower) & (coordinates < upper)]
            if region.size == 0:
                continue
            eor_distances = np.abs((region - self._eor_line_reference) @ eor_normal)
            row_starts.append(region[int(np.argmin(eor_distances))].copy())

        path_direction = -self._roi_driving_direction
        path_records: list[dict] = []
        for path_id, (first_start, second_start) in enumerate(
            zip(row_starts[:-1], row_starts[1:]), start=1,
        ):
            entrance = 0.5 * (first_start + second_start)
            end = entrance + self.geometric_path_length * path_direction
            path_records.append({
                'path_id': path_id,
                'path_start_x': float(entrance[0]),
                'path_start_y': float(entrance[1]),
                'path_end_x': float(end[0]),
                'path_end_y': float(end[1]),
            })
        return row_starts, path_records

    def _detect_rows_in_roi_strips(
        self,
        xy_roi: np.ndarray,
    ) -> tuple[list[dict], int]:
        normal = self._roi_strip_normal
        boundaries = self._roi_strip_boundaries
        if self._eor_line_reference is None or self._eor_line_direction is None:
            return [], 0
        if (
            not self.enable_roi_strip_detection
            or normal is None
            or boundaries is None
            or boundaries.size < 2
        ):
            regions = [xy_roi]
        else:
            coordinates = xy_roi @ normal
            regions = [
                xy_roi[(coordinates >= lower) & (coordinates < upper)]
                for lower, upper in zip(boundaries[:-1], boundaries[1:])
            ]

        row_records: list[dict] = []
        processed_regions = 0
        for region_id, region_points in enumerate(regions, start=1):
            # Each ROI strip represents one physical row. Stop once the
            # whole ROI reaches its cap, rather than applying the cap anew
            # in every strip.
            if len(row_records) >= self.row_max_rows:
                break
            if region_points.shape[0] < self.row_min_segment_inliers:
                continue
            processed_regions += 1
            region_rows = detect_rows_from_xy(
                region_points,
                distance_threshold=self.row_distance_threshold,
                max_gap=self.row_max_gap,
                max_iterations=self.row_max_iterations,
                min_segment_inliers=self.row_min_segment_inliers,
                max_rows=1,
                remove_radius=self.row_remove_radius,
                min_points_left=self.row_min_points_left,
                eor_line_reference=tuple(self._eor_line_reference),
                eor_line_direction=tuple(self._eor_line_direction),
            )
            for row in region_rows:
                row['row_id'] = len(row_records) + 1
                row['roi_strip_id'] = region_id
                row_records.append(row)
        return row_records, processed_regions

    def path_pipeline_reset_callback(self, _msg: Empty) -> None:
        """Clear all state belonging to the completed ROI cycle."""
        self._clear_path_pipeline_cycle_state()
        self._pending_clouds.clear()
        self._pending_references.clear()

        if self.enable_path_publisher:
            paths = PoseArray()
            paths.header.frame_id = 'world'
            paths.header.stamp = self.get_clock().now().to_msg()
            self.path_publisher.publish(paths)

        geometric_paths = PoseArray()
        geometric_paths.header.frame_id = 'world'
        geometric_paths.header.stamp = self.get_clock().now().to_msg()
        self.geometric_path_publisher.publish(geometric_paths)

        delete_all = Marker()
        delete_all.header.frame_id = 'world'
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.action = Marker.DELETEALL
        geometric_markers = MarkerArray()
        geometric_markers.markers.append(delete_all)
        self.geometric_marker_publisher.publish(geometric_markers)

        if self.enable_marker_publisher:
            delete_all = Marker()
            delete_all.header.frame_id = 'world'
            delete_all.header.stamp = self.get_clock().now().to_msg()
            delete_all.action = Marker.DELETEALL
            markers = MarkerArray()
            markers.markers.append(delete_all)
            self.marker_publisher.publish(markers)

        if self.enable_row_marker_publisher:
            delete_all = Marker()
            delete_all.header.frame_id = 'world'
            delete_all.header.stamp = self.get_clock().now().to_msg()
            delete_all.action = Marker.DELETEALL
            markers = MarkerArray()
            markers.markers.append(delete_all)
            self.row_marker_publisher.publish(markers)

        self.get_logger().info(
            'Path pipeline cycle reset; waiting for point clouds from the next ROI.'
        )

    def _update_group_start_distance_stop(self, distance: float) -> bool:
        """Latch detection off on an above-to-below threshold crossing."""
        previous = self._previous_group_start_distance
        self._previous_group_start_distance = distance
        if (
            not self.enable_group_start_distance_stop
            or self._row_path_detection_stopped
            or previous is None
            or previous <= self.group_start_distance_stop_threshold
            or distance > self.group_start_distance_stop_threshold
        ):
            return False

        self._row_path_detection_stopped = True
        self.get_logger().info(
            'Group-start distance crossed the falling stop threshold '
            f'({previous:.3f} -> {distance:.3f} m, threshold='
            f'{self.group_start_distance_stop_threshold:.3f} m). '
            'Row and path detection are now latched off.'
        )
        return True

    def _init_notebook_plot(self) -> None:
        try:
            if self.enable_notebook_plot:
                import matplotlib.pyplot as plt

                self._plt = plt
                self._plt.ion()
                self._fig, self._ax = self._plt.subplots(figsize=(11, 9))
            else:
                # Recording also works without a display or a live plot window.
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                from matplotlib.figure import Figure

                self._fig = Figure(figsize=(11, 9))
                FigureCanvasAgg(self._fig)
                self._ax = self._fig.subplots()
        except Exception as exc:
            self.enable_notebook_plot = False
            self.enable_plot_video = False
            self.get_logger().warning(f'Plotting and video recording disabled: {exc}')
            return

        if self.enable_plot_video:
            try:
                from .plot_video import PlotVideoRecorder

                self._video_recorder = PlotVideoRecorder(self.plot_video_output_path, self.plot_video_fps)
                self.get_logger().info(
                    f'Plot video ready: {self._video_recorder.output_path} '
                    f'| fps={self.plot_video_fps} | every {self.plot_every_n_frames} scans'
                )
            except Exception as exc:
                self.enable_plot_video = False
                self.get_logger().warning(f'Plot video recording disabled: {exc}')

    def _update_notebook_plot(
        self,
        xy_roi: np.ndarray,
        row_records: list[dict],
        output_paths: list[dict],
        group_start_lines: list[dict],
    ) -> None:
        if self._ax is None or self._fig is None:
            return

        ax = self._ax
        ax.clear()
        confirmed_starts = (
            self.path_start_tracker.confirmed_starts() if self.enable_path_start_tracking else []
        )
        confirmed_xy = np.array(
            [[rec['path_start_x'], rec['path_start_y']] for rec in confirmed_starts], dtype=float,
        ).reshape(-1, 2)

        if xy_roi.size > 0:
            ax.scatter(xy_roi[:, 0], xy_roi[:, 1], s=2, color='lightgray', label='ROI points')

        used_row_ids = set()
        for rec in output_paths:
            if 'row_a_id' in rec:
                used_row_ids.add(int(rec['row_a_id']))
            if 'row_b_id' in rec:
                used_row_ids.add(int(rec['row_b_id']))

        for row in row_records:
            row_id = int(row.get('row_id', -1))
            used_by_path = row_id in used_row_ids
            sx, sy = float(row['start_x']), float(row['start_y'])
            ex, ey = float(row['end_x']), float(row['end_y'])
            color = 'tab:orange' if used_by_path else 'goldenrod'
            ax.plot(
                [sx, ex], [sy, ey],
                linestyle='-' if used_by_path else '--',
                linewidth=2.0 if used_by_path else 1.4,
                color=color,
                alpha=0.9 if used_by_path else 0.75,
            )
            ax.scatter([sx], [sy], s=28 if used_by_path else 18, color='red' if used_by_path else color)
            ax.text(sx, sy, f"R{row_id}", fontsize=8, color=color)

        for rec in output_paths:
            sx, sy = float(rec['path_start_x']), float(rec['path_start_y'])
            ex, ey = float(rec['path_end_x']), float(rec['path_end_y'])
            pid = int(rec['path_id'])
            is_reference = bool(rec.get('is_reference', False))
            color = 'tab:red' if is_reference else 'tab:blue'
            style = '-' if is_reference else '--'
            ax.plot([sx, ex], [sy, ey], linestyle=style, linewidth=2.0, color=color, alpha=0.95)
            if not self.enable_path_start_tracking:
                ax.scatter([sx], [sy], s=22, color='green')
            mx, my = 0.5 * (sx + ex), 0.5 * (sy + ey)
            label = f"p{pid}"
            if 'observation_count' in rec:
                label += f" (n={rec['observation_count']})"
            ax.text(mx, my, label, fontsize=8, color=color)

        if confirmed_starts:
            ax.scatter(confirmed_xy[:, 0], confirmed_xy[:, 1], s=22, color='green',
                       zorder=5, label='Confirmed path starts', gid='confirmed_path_starts')
            visible_path_ids = {int(rec['path_id']) for rec in output_paths}
            for rec in confirmed_starts:
                if rec['path_id'] not in visible_path_ids:
                    ax.annotate(
                        f"p{rec['path_id']} (n={rec['observation_count']})",
                        (rec['path_start_x'], rec['path_start_y']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, color='green',
                    )

        for ln in group_start_lines:
            ax.plot(
                [float(ln['start_x']), float(ln['end_x'])],
                [float(ln['start_y']), float(ln['end_y'])],
                linestyle='-.',
                linewidth=2.0,
                color='tab:green',
                alpha=0.9,
            )

        # Stored world positions also determine the viewport, including empty scans.
        bounds_xy = np.vstack((xy_roi, confirmed_xy))
        if self.enable_plot_autoscale and bounds_xy.size > 0:
            x_min = float(np.min(bounds_xy[:, 0]))
            x_max = float(np.max(bounds_xy[:, 0]))
            y_min = float(np.min(bounds_xy[:, 1]))
            y_max = float(np.max(bounds_xy[:, 1]))
            padding = max(0.1, self.plot_padding_m)
            ax.set_xlim(x_min - padding, x_max + padding)
            ax.set_ylim(y_min - padding, y_max + padding)
        else:
            ax.set_xlim(self.plot_x_min, self.plot_x_max)
            ax.set_ylim(self.plot_y_min, self.plot_y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'Notebook-style live view | frame {self.frame_count}')

        if self.enable_plot_video and self._video_recorder is not None:
            try:
                self._video_recorder.capture(self._fig)
            except Exception as exc:
                self.enable_plot_video = False
                self.get_logger().warning(f'Plot video recording stopped: {exc}')
                self._close_plot_video()

        if self.enable_notebook_plot and self._plt is not None:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            self._plt.pause(0.001)

    def _close_plot_video(self) -> None:
        recorder = self._video_recorder
        self._video_recorder = None
        if recorder is not None:
            try:
                recorder.close()
                if recorder.frame_count:
                    self.get_logger().info(
                        f'Saved plot video: {recorder.output_path} | frames={recorder.frame_count}'
                    )
            except Exception as exc:
                self.get_logger().warning(f'Could not finalize plot video: {exc}')

    def close_plot(self) -> None:
        self._close_plot_video()
        if self._plt is not None and self._fig is not None:
            self._plt.close(self._fig)
        self._ax = None
        self._fig = None

    def _publish_debug_markers(
        self,
        src_msg: PointCloud2,
        path_records: list[dict],
        group_start_lines: list[dict],
    ) -> int:
        marker_array = MarkerArray()

        # Publish every confirmed averaged entrance on every scan. Confirmed
        # tracks remain in the tracker when their path is absent from the
        # current detection, matching the persistent green dots in the plot.
        confirmed_starts = (
            self.path_start_tracker.confirmed_starts() if self.enable_path_start_tracking else []
        )
        for rec in confirmed_starts:
            marker = Marker()
            marker.header = src_msg.header
            marker.ns = 'pernav_confirmed_path_starts'
            marker.id = int(rec['path_id'])
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(rec['path_start_x'])
            marker.pose.position.y = float(rec['path_start_y'])
            marker.pose.position.z = 0.08
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.25
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime.sec = int(self.marker_lifetime_sec)
            marker.lifetime.nanosec = int((self.marker_lifetime_sec % 1.0) * 1e9)
            marker_array.markers.append(marker)

        for rec in path_records:
            marker = Marker()
            marker.header = src_msg.header
            marker.ns = 'pernav_paths'
            marker.id = int(rec['path_id'])
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            p1 = Point()
            p1.x = float(rec['path_start_x'])
            p1.y = float(rec['path_start_y'])
            p1.z = 0.05
            p2 = Point()
            p2.x = float(rec['path_end_x'])
            p2.y = float(rec['path_end_y'])
            p2.z = 0.05
            marker.points = [p1, p2]

            marker.scale.x = self.marker_line_width

            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.95

            marker.lifetime.sec = int(self.marker_lifetime_sec)
            marker.lifetime.nanosec = int((self.marker_lifetime_sec % 1.0) * 1e9)

            marker_array.markers.append(marker)

        if self.enable_group_start_line_marker:
            base_id = 10000
            for i, ln in enumerate(group_start_lines):
                marker = Marker()
                marker.header = src_msg.header
                marker.ns = 'pernav_group_start_lines'
                marker.id = base_id + i
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD

                p0 = Point()
                p0.x = float(ln['start_x'])
                p0.y = float(ln['start_y'])
                p0.z = 0.06
                p1 = Point()
                p1.x = float(ln['end_x'])
                p1.y = float(ln['end_y'])
                p1.z = 0.06
                marker.points = [p0, p1]

                marker.scale.x = max(0.02, self.marker_line_width * 0.8)
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.95
                marker.lifetime.sec = int(self.marker_lifetime_sec)
                marker.lifetime.nanosec = int((self.marker_lifetime_sec % 1.0) * 1e9)

                marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)
        return len(marker_array.markers)

    def _publish_row_markers(self, src_msg: PointCloud2, row_records: list[dict]) -> int:
        marker_array = MarkerArray()
        for row in row_records:
            marker = Marker()
            marker.header = src_msg.header
            marker.ns = 'pernav_detected_rows'
            marker.id = int(row['row_id'])
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            start = Point()
            start.x = float(row['start_x'])
            start.y = float(row['start_y'])
            start.z = 0.04
            end = Point()
            end.x = float(row['end_x'])
            end.y = float(row['end_y'])
            end.z = 0.04
            marker.points = [start, end]

            marker.scale.x = self.marker_line_width
            marker.color.r = 1.0
            marker.color.g = 0.65
            marker.color.b = 0.0
            marker.color.a = 0.95
            marker.lifetime.sec = int(self.marker_lifetime_sec)
            marker.lifetime.nanosec = int((self.marker_lifetime_sec % 1.0) * 1e9)
            marker_array.markers.append(marker)

        self.row_marker_publisher.publish(marker_array)
        return len(marker_array.markers)

    def _publish_discarded_fov_row_markers(
        self, src_msg: PointCloud2, row_records: list[dict],
    ) -> int:
        marker_array = MarkerArray()
        for row in row_records:
            marker = Marker()
            marker.header = src_msg.header
            marker.ns = 'pernav_discarded_fov_rows'
            marker.id = int(row['row_id'])
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.points = [
                Point(x=float(row['start_x']), y=float(row['start_y']), z=0.06),
                Point(x=float(row['end_x']), y=float(row['end_y']), z=0.06),
            ]
            marker.scale.x = self.marker_line_width
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 0.0, 1.0
            marker.lifetime.sec = int(self.marker_lifetime_sec)
            marker.lifetime.nanosec = int((self.marker_lifetime_sec % 1.0) * 1e9)
            marker_array.markers.append(marker)
        self.discarded_fov_row_marker_publisher.publish(marker_array)
        return len(marker_array.markers)

    def _publish_chassis_front_path_distance(
        self, src_msg: PointCloud2, chassis_front_marker: Marker, path_records: list[dict],
    ) -> float | None:
        """Publish the XY distance from the chassis front to the nearest path entrance."""
        if not path_records:
            return None
        front = np.array([
            chassis_front_marker.pose.position.x,
            chassis_front_marker.pose.position.y,
        ], dtype=float)
        closest = min(
            path_records,
            key=lambda record: float(np.linalg.norm(front - np.array([
                record['path_start_x'], record['path_start_y'],
            ], dtype=float))),
        )
        entrance = np.array([closest['path_start_x'], closest['path_start_y']], dtype=float)
        distance = float(np.linalg.norm(front - entrance))
        self.chassis_front_path_distance_publisher.publish(Float64(data=distance))

        marker = Marker()
        marker.header = src_msg.header
        marker.ns = 'pernav_chassis_front_path_distance'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        # Use the marker's exact 3D centre so the line visibly starts at the
        # red chassis-front sphere in an oblique RViz camera view.
        z = float(chassis_front_marker.pose.position.z)
        marker.points = [
            Point(x=float(front[0]), y=float(front[1]), z=z),
            Point(x=float(entrance[0]), y=float(entrance[1]), z=z),
        ]
        marker.scale.x = max(0.02, self.marker_line_width)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.9, 0.1, 1.0, 1.0
        marker.lifetime.sec = int(self.marker_lifetime_sec)
        marker.lifetime.nanosec = int((self.marker_lifetime_sec % 1.0) * 1e9)
        self.chassis_front_path_distance_marker_publisher.publish(marker)

        # Publish both elements together so RViz cannot render a chassis point
        # from one scan with a distance line from another scan.
        start_marker = Marker()
        start_marker.header = src_msg.header
        start_marker.ns = 'pernav_chassis_front_path_distance_start'
        start_marker.id = 0
        start_marker.type = Marker.SPHERE
        start_marker.action = Marker.ADD
        start_marker.pose.position.x = float(front[0])
        start_marker.pose.position.y = float(front[1])
        start_marker.pose.position.z = float(chassis_front_marker.pose.position.z)
        start_marker.pose.orientation.w = 1.0
        start_marker.scale.x = start_marker.scale.y = start_marker.scale.z = 0.20
        start_marker.color.r, start_marker.color.g, start_marker.color.b, start_marker.color.a = 1.0, 0.0, 0.0, 1.0
        start_marker.lifetime = marker.lifetime

        synchronized_markers = MarkerArray()
        synchronized_markers.markers = [start_marker, marker]
        self.chassis_front_path_distance_markers_publisher.publish(synchronized_markers)
        return distance

    def _publish_paths(self, src_msg: PointCloud2, path_records: list[dict]) -> int:
        msg = PoseArray()
        msg.header = src_msg.header

        for rec in path_records:
            pose = Pose()
            pose.position.x = float(rec['path_start_x'])
            pose.position.y = float(rec['path_start_y'])
            pose.position.z = 0.0

            dx = float(rec['path_end_x']) - pose.position.x
            dy = float(rec['path_end_y']) - pose.position.y
            yaw = math.atan2(dy, dx)
            pose.orientation.z = math.sin(0.5 * yaw)
            pose.orientation.w = math.cos(0.5 * yaw)

            msg.poses.append(pose)

        self.path_publisher.publish(msg)
        return len(msg.poses)

    def _publish_geometric_paths(
        self,
        src_msg: PointCloud2,
        path_records: list[dict],
        start_lines: list[dict],
    ) -> int:
        paths_msg = PoseArray()
        paths_msg.header = src_msg.header
        markers = MarkerArray()

        for record in path_records:
            marker = Marker()
            marker.header = src_msg.header
            marker.ns = 'pernav_geometric_path_starts'
            marker.id = int(record['path_id'])
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(record['path_start_x'])
            marker.pose.position.y = float(record['path_start_y'])
            marker.pose.position.z = 0.08
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.22
            marker.scale.y = 0.22
            marker.scale.z = 0.22
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.95
            marker.lifetime.sec = int(self.marker_lifetime_sec)
            marker.lifetime.nanosec = int((self.marker_lifetime_sec % 1.0) * 1e9)
            markers.markers.append(marker)

        for record in path_records:
            pose = Pose()
            pose.position.x = float(record['path_start_x'])
            pose.position.y = float(record['path_start_y'])
            dx = float(record['path_end_x']) - pose.position.x
            dy = float(record['path_end_y']) - pose.position.y
            yaw = math.atan2(dy, dx)
            pose.orientation.z = math.sin(0.5 * yaw)
            pose.orientation.w = math.cos(0.5 * yaw)
            paths_msg.poses.append(pose)

            marker = Marker()
            marker.header = src_msg.header
            marker.ns = 'pernav_geometric_paths'
            marker.id = int(record['path_id'])
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.points = [
                Point(x=pose.position.x, y=pose.position.y, z=0.05),
                Point(
                    x=float(record['path_end_x']),
                    y=float(record['path_end_y']),
                    z=0.05,
                ),
            ]
            marker.scale.x = self.marker_line_width
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.95
            marker.lifetime.sec = int(self.marker_lifetime_sec)
            marker.lifetime.nanosec = int((self.marker_lifetime_sec % 1.0) * 1e9)
            markers.markers.append(marker)

        for line_id, line in enumerate(start_lines, start=1):
            marker = Marker()
            marker.header = src_msg.header
            marker.ns = 'pernav_geometric_sor_lines'
            marker.id = line_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.points = [
                Point(x=float(line['start_x']), y=float(line['start_y']), z=0.06),
                Point(x=float(line['end_x']), y=float(line['end_y']), z=0.06),
            ]
            marker.scale.x = max(0.02, self.marker_line_width * 0.8)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.95
            marker.lifetime.sec = int(self.marker_lifetime_sec)
            marker.lifetime.nanosec = int((self.marker_lifetime_sec % 1.0) * 1e9)
            markers.markers.append(marker)

        self.geometric_path_publisher.publish(paths_msg)
        self.geometric_marker_publisher.publish(markers)
        return len(paths_msg.poses)

    @staticmethod
    def _pc2_to_xyz(msg: PointCloud2) -> tuple[int, np.ndarray]:
        raw_count = int(msg.width) * int(msg.height)

        offsets = {field.name: int(field.offset) for field in msg.fields}
        if 'x' not in offsets or 'y' not in offsets or 'z' not in offsets:
            return raw_count, np.empty((0, 3), dtype=np.float32)

        x_off = offsets['x']
        y_off = offsets['y']
        z_off = offsets['z']

        fmt = '>f' if msg.is_bigendian else '<f'
        step = int(msg.point_step)
        data = msg.data

        points = []
        for i in range(0, len(data), step):
            x = struct.unpack_from(fmt, data, i + x_off)[0]
            y = struct.unpack_from(fmt, data, i + y_off)[0]
            z = struct.unpack_from(fmt, data, i + z_off)[0]
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                points.append((x, y, z))

        if not points:
            return raw_count, np.empty((0, 3), dtype=np.float32)

        return raw_count, np.asarray(points, dtype=np.float32)

    def listener_callback(self, msg: PointCloud2) -> None:
        self._queue_for_reference_sync(msg, is_cloud=True)

    def row_start_ref_callback(self, msg: PointStamped) -> None:
        if not all(math.isfinite(value) for value in (msg.point.x, msg.point.y, msg.point.z)):
            self.get_logger().warning('Ignoring non-finite row-start reference.')
            return
        self._queue_for_reference_sync(msg, is_cloud=False)

    def fov_boundary_marker_callback(self, msg: Marker) -> None:
        """Buffer world FOV border rays until their matching ROI cloud arrives."""
        if not self.enable_fov_boundary_row_filter:
            return
        if msg.header.frame_id != 'world' or msg.ns != 'lidar_fov_world':
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._pending_fov_markers[stamp] = msg
        self._pending_fov_markers.move_to_end(stamp)
        self._try_process_synchronized_stamp(stamp)
        while len(self._pending_fov_markers) > self.row_start_ref_queue_size:
            self._pending_fov_markers.popitem(last=False)
            self.get_logger().warning('Dropping FOV marker without a matching world ROI cloud.')

    def chassis_front_marker_callback(self, msg: Marker) -> None:
        """Buffer the front-centre marker for the matching world-frame cloud."""
        if not self.enable_chassis_front_path_distance_publisher:
            return
        if msg.header.frame_id != 'world' or msg.ns != 'chassis_front_world':
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._pending_chassis_front_markers[stamp] = msg
        self._pending_chassis_front_markers.move_to_end(stamp)
        while len(self._pending_chassis_front_markers) > self.row_start_ref_queue_size:
            self._pending_chassis_front_markers.popitem(last=False)

    def _queue_for_reference_sync(self, msg: PointCloud2 | PointStamped, *, is_cloud: bool) -> None:
        """Pair by exact scan timestamp, allowing either topic to arrive first."""
        if msg.header.frame_id != 'world':
            self.get_logger().warning(
                f'Ignoring {"cloud" if is_cloud else "row-start reference"} '
                f'in frame {msg.header.frame_id!r}; expected world.'
            )
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        queue = self._pending_clouds if is_cloud else self._pending_references
        queue[stamp] = msg
        queue.move_to_end(stamp)

        self._try_process_synchronized_stamp(stamp)

        # Evict by arrival order, so earlier timestamps from bag replay can still match.
        while len(queue) > self.row_start_ref_queue_size:
            queue.popitem(last=False)
            if is_cloud:
                self.get_logger().warning('Dropping cloud without a matching world row-start reference.')

    def _try_process_synchronized_stamp(self, stamp: tuple[int, int]) -> None:
        if stamp not in self._pending_clouds or stamp not in self._pending_references:
            return
        if self.enable_fov_boundary_row_filter and stamp not in self._pending_fov_markers:
            return
        cloud = self._pending_clouds.pop(stamp)
        reference = self._pending_references.pop(stamp)
        fov_marker = self._pending_fov_markers.pop(stamp, None)
        chassis_front_marker = self._pending_chassis_front_markers.pop(stamp, None)
        fov_filter_enabled = bool(getattr(self, 'enable_fov_boundary_row_filter', False))
        chassis_distance_enabled = bool(
            getattr(self, 'enable_chassis_front_path_distance_publisher', False)
        )
        if fov_filter_enabled or chassis_distance_enabled:
            self._process_cloud(cloud, reference, fov_marker, chassis_front_marker)
        else:
            self._process_cloud(cloud, reference)

    def _process_cloud(
        self, msg: PointCloud2, reference: PointStamped, fov_marker: Marker | None = None,
        chassis_front_marker: Marker | None = None,
    ) -> None:
        self.frame_count += 1
        stamp_sec = msg.header.stamp.sec
        stamp_nsec = msg.header.stamp.nanosec
        stamp_ns = stamp_sec * 1_000_000_000 + stamp_nsec
        self._prepare_group_start_distance_stamp(stamp_ns)
        raw_count, xyz = self._pc2_to_xyz(msg)
        xy = xyz[:, :2]
        xy_roi = xy

        valid_xyz_count = int(xyz.shape[0])
        roi_xy_count = int(xy_roi.shape[0])

        _, geometric_path_records = self._build_geometric_paths(xy_roi)
        geometric_output_paths: list[dict] = []
        geometric_start_lines: list[dict] = []
        if self.enable_geometric_path_experiment:
            geometric_output_paths = self.geometric_path_start_tracker.update(
                geometric_path_records,
                stamp_ns,
                self._eor_line_reference,
                self._eor_line_direction,
            )
            if self._eor_line_reference is not None and self._eor_line_direction is not None:
                geometric_start_lines = self.geometric_start_line_tracker.update(
                    self.geometric_path_start_tracker.confirmed_starts(),
                    self._eor_line_reference,
                    self._eor_line_direction,
                )
        geometric_paths_published = self._publish_geometric_paths(
            msg, geometric_output_paths, geometric_start_lines,
        )

        rows_detected = -1
        rows_discarded_fov = 0
        roi_strips_processed = 0
        row_records: list[dict] = []
        if (
            self.enable_row_detection
            and not self._row_path_detection_stopped
            and roi_xy_count >= self.row_min_segment_inliers
        ):
            row_records, roi_strips_processed = self._detect_rows_in_roi_strips(xy_roi)
            rows_detected = len(row_records)
        discarded_fov_rows: list[dict] = []
        fov_filter_enabled = bool(getattr(self, 'enable_fov_boundary_row_filter', False))
        if fov_filter_enabled and fov_marker is not None and row_records:
            row_records, discarded_fov_rows = self._filter_rows_by_fov_boundary(row_records, fov_marker)
            rows_discarded_fov = len(discarded_fov_rows)

        paths_detected = -1
        groups_detected = -1
        published_paths = -1
        published_markers = -1
        published_row_markers = -1
        published_group_start_distance: float | None = None
        chassis_front_path_distance: float | None = None
        group_start_lines_count = -1
        output_paths: list[dict] = []
        group_start_lines: list[dict] = []
        if self.enable_row_marker_publisher:
            published_row_markers = self._publish_row_markers(msg, row_records)
        if bool(getattr(self, 'enable_discarded_fov_row_marker_publisher', False)):
            self._publish_discarded_fov_row_markers(msg, discarded_fov_rows)

        if self.enable_path_detection and not self._row_path_detection_stopped and row_records:
            path_records = build_paths_from_rows(
                row_records,
                min_path_length=self.min_path_length,
            )
            paths_detected = len(path_records)

            if self.enable_parallel_correction and path_records:
                groups = build_path_groups(
                    path_records,
                    angle_thresh_deg=self.group_angle_thresh_deg,
                    midpoint_thresh=self.group_midpoint_thresh,
                    lateral_thresh=self.group_lateral_thresh,
                )
                groups_detected = len(groups)
                output_paths = apply_parallel_correction(path_records, groups)
            else:
                groups_detected = 0 if path_records else -1
                output_paths = path_records

        if self.enable_path_detection:
            if self._row_path_detection_stopped:
                output_paths = [dict(record) for record in self._last_confirmed_output_paths]
                group_start_lines = [
                    dict(line) for line in self._last_confirmed_group_start_lines
                ]
                group_start_lines_count = len(group_start_lines)
            else:
                if self.enable_path_start_tracking:
                    output_paths = self.path_start_tracker.update(
                        output_paths,
                        stamp_ns,
                        self._eor_line_reference,
                        self._eor_line_direction,
                    )
                if self.enable_parallel_correction:
                    if (
                        self.enable_path_start_tracking
                        and self._eor_line_reference is not None
                        and self._eor_line_direction is not None
                    ):
                        group_start_lines = self.start_line_tracker.update(
                            self.path_start_tracker.confirmed_starts(),
                            self._eor_line_reference,
                            self._eor_line_direction,
                        )
                    else:
                        group_start_lines = build_group_start_lines(
                            output_paths, extension_m=self.start_line_extension_m,
                        )
                    group_start_lines_count = len(group_start_lines)

                # Keep the last nonempty confirmed result. If the falling
                # threshold is crossed on this scan, this is the geometry that
                # will be republished after detection is latched off.
                if output_paths:
                    self._last_confirmed_output_paths = [
                        dict(record) for record in output_paths
                    ]
                if group_start_lines:
                    self._last_confirmed_group_start_lines = [
                        dict(line) for line in group_start_lines
                    ]

            published_group_start_distance = self._publish_group_start_distance(
                reference, group_start_lines, stamp_ns,
            )

            # The threshold can be crossed on a scan without a new detection.
            # Freeze and publish the cached result immediately on that scan.
            if self._row_path_detection_stopped:
                output_paths = [dict(record) for record in self._last_confirmed_output_paths]
                group_start_lines = [
                    dict(line) for line in self._last_confirmed_group_start_lines
                ]
                group_start_lines_count = len(group_start_lines)

            if self.enable_path_publisher:
                published_paths = self._publish_paths(msg, output_paths)

            if self.enable_marker_publisher:
                published_markers = self._publish_debug_markers(msg, output_paths, group_start_lines)

        if (
            bool(getattr(self, 'enable_chassis_front_path_distance_publisher', False))
            and chassis_front_marker is not None
            and output_paths
        ):
            chassis_front_path_distance = self._publish_chassis_front_path_distance(
                msg, chassis_front_marker, output_paths,
            )

        self.get_logger().info(
            f'Lidar messages are received | frame={self.frame_count} '
            f'| stamp={stamp_sec}.{stamp_nsec:09d} '
            f'| width={msg.width} | height={msg.height} | point_step={msg.point_step} '
            f'| raw_points={raw_count} | valid_xyz_points={valid_xyz_count} '
            f'| roi_xy_points={roi_xy_count} '
            f'| roi_strips_processed={roi_strips_processed} '
            f'| rows_detected={rows_detected} '
            f'| rows_discarded_fov={rows_discarded_fov} '
            f'| paths_detected={paths_detected} '
            f'| groups_detected={groups_detected} '
            f'| group_start_lines={group_start_lines_count} '
            f'| group_start_distance={published_group_start_distance} '
            f'| chassis_front_path_distance={chassis_front_path_distance} '
            f'| detection_stopped={self._row_path_detection_stopped} '
            f'| published_paths={published_paths} '
            f'| published_markers={published_markers} '
            f'| published_row_markers={published_row_markers}'
        )

        if (self.enable_notebook_plot or self.enable_plot_video) and (self.frame_count % self.plot_every_n_frames == 0):
            self._update_notebook_plot(xy_roi, row_records, output_paths, group_start_lines)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathPipelineNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_plot()
        node.destroy_node()
        # Ctrl+C can already shut down the default context via ROS signal handling.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
