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
                ('input_topic', '/pcl_world_fov'),
                ('row_start_ref_topic', '/pernav/row_start_ref_world'),
                ('row_start_ref_queue_size', 100),
                ('enable_row_detection', True),
                ('row_distance_threshold', 0.25),
                ('row_max_gap', 0.6),
                ('row_max_iterations', 400),
                ('row_min_segment_inliers', 40),
                ('row_max_rows', 8),
                ('row_remove_radius', 1.9),
                ('row_min_points_left', 30),
                ('enable_path_detection', True),
                ('path_width', 3.4),
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
        self._pending_clouds: OrderedDict[tuple[int, int], PointCloud2] = OrderedDict()
        self._pending_references: OrderedDict[tuple[int, int], PointStamped] = OrderedDict()
        self.enable_row_detection = bool(self.get_parameter('enable_row_detection').value)
        self.row_distance_threshold = float(self.get_parameter('row_distance_threshold').value)
        self.row_max_gap = float(self.get_parameter('row_max_gap').value)
        self.row_max_iterations = int(self.get_parameter('row_max_iterations').value)
        self.row_min_segment_inliers = int(self.get_parameter('row_min_segment_inliers').value)
        self.row_max_rows = int(self.get_parameter('row_max_rows').value)
        self.row_remove_radius = float(self.get_parameter('row_remove_radius').value)
        self.row_min_points_left = int(self.get_parameter('row_min_points_left').value)
        self.enable_path_detection = bool(self.get_parameter('enable_path_detection').value)
        self.path_width = float(self.get_parameter('path_width').value)
        self.min_path_length = float(self.get_parameter('min_path_length').value)
        self.enable_path_start_tracking = bool(self.get_parameter('enable_path_start_tracking').value)
        self.path_start_tracker = PathStartTracker(
            max_distance=float(self.get_parameter('tracking_max_distance').value),
            max_heading_deg=float(self.get_parameter('tracking_max_heading_deg').value),
            min_observations=int(self.get_parameter('tracking_min_observations').value),
            candidate_max_missed_scans=int(self.get_parameter('tracking_candidate_max_missed_scans').value),
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
        self.group_start_distance_publisher = self.create_publisher(
            Float64, self.group_start_distance_topic, 10,
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
        self.get_logger().info(
            f'Path pipeline node listening on topic {self.input_topic} '
            f'| synchronized_reference={self.row_start_ref_topic} '
            f'| row_detection={self.enable_row_detection} '
            f'| path_detection={self.enable_path_detection} '
            f'| path_start_tracking={self.enable_path_start_tracking} '
            f'| parallel_correction={self.enable_parallel_correction} '
            f'| path_pub={self.enable_path_publisher} ({self.path_output_topic}) '
            f'| marker_pub={self.enable_marker_publisher} ({self.marker_output_topic}) '
            f'| notebook_plot={self.enable_notebook_plot} '
            f'| plot_video={self.enable_plot_video} '
            f'| plot_autoscale={self.enable_plot_autoscale} '
            f'| group_start_line_marker={self.enable_group_start_line_marker} '
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

    def path_pipeline_reset_callback(self, _msg: Empty) -> None:
        """Clear all state belonging to the completed FOV cycle."""
        self._clear_path_pipeline_cycle_state()
        self._pending_clouds.clear()
        self._pending_references.clear()

        if self.enable_path_publisher:
            paths = PoseArray()
            paths.header.frame_id = 'world'
            paths.header.stamp = self.get_clock().now().to_msg()
            self.path_publisher.publish(paths)

        if self.enable_marker_publisher:
            delete_all = Marker()
            delete_all.header.frame_id = 'world'
            delete_all.header.stamp = self.get_clock().now().to_msg()
            delete_all.action = Marker.DELETEALL
            markers = MarkerArray()
            markers.markers.append(delete_all)
            self.marker_publisher.publish(markers)

        self.get_logger().info(
            'Path pipeline cycle reset; waiting for point clouds from the next FOV.'
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
        xy_fov: np.ndarray,
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

        if xy_fov.size > 0:
            ax.scatter(xy_fov[:, 0], xy_fov[:, 1], s=2, color='lightgray', label='FOV points')

        used_row_ids = set()
        for rec in output_paths:
            if 'row_a_id' in rec:
                used_row_ids.add(int(rec['row_a_id']))
            if 'row_b_id' in rec:
                used_row_ids.add(int(rec['row_b_id']))

        rows_to_plot = [row for row in row_records if int(row.get('row_id', -1)) in used_row_ids]

        for row in rows_to_plot:
            sx, sy = float(row['start_x']), float(row['start_y'])
            ex, ey = float(row['end_x']), float(row['end_y'])
            ax.plot([sx, ex], [sy, ey], linewidth=2.0, color='tab:orange', alpha=0.9)
            ax.scatter([sx], [sy], s=28, color='red')
            ax.text(sx, sy, f"R{int(row['row_id'])}", fontsize=8, color='tab:orange')

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
        bounds_xy = np.vstack((xy_fov, confirmed_xy))
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
            marker.color.r = 0.1
            marker.color.g = 1.0
            marker.color.b = 0.1
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

            is_reference = bool(rec.get('is_reference', False))
            if is_reference:
                marker.color.r = 1.0
                marker.color.g = 0.15
                marker.color.b = 0.15
            else:
                marker.color.r = 0.15
                marker.color.g = 0.85
                marker.color.b = 1.0
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
                marker.color.r = 0.2
                marker.color.g = 1.0
                marker.color.b = 0.2
                marker.color.a = 0.95
                marker.lifetime.sec = int(self.marker_lifetime_sec)
                marker.lifetime.nanosec = int((self.marker_lifetime_sec % 1.0) * 1e9)

                marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)
        return len(marker_array.markers)

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

        if stamp in self._pending_clouds and stamp in self._pending_references:
            cloud = self._pending_clouds.pop(stamp)
            reference = self._pending_references.pop(stamp)
            self._process_cloud(cloud, reference)

        # Evict by arrival order, so earlier timestamps from bag replay can still match.
        while len(queue) > self.row_start_ref_queue_size:
            queue.popitem(last=False)
            if is_cloud:
                self.get_logger().warning('Dropping cloud without a matching world row-start reference.')

    def _process_cloud(self, msg: PointCloud2, reference: PointStamped) -> None:
        self.frame_count += 1
        stamp_sec = msg.header.stamp.sec
        stamp_nsec = msg.header.stamp.nanosec
        stamp_ns = stamp_sec * 1_000_000_000 + stamp_nsec
        self._prepare_group_start_distance_stamp(stamp_ns)
        raw_count, xyz = self._pc2_to_xyz(msg)
        xy = xyz[:, :2]
        xy_fov = xy

        valid_xyz_count = int(xyz.shape[0])
        fov_xy_count = int(xy_fov.shape[0])

        rows_detected = -1
        row_records: list[dict] = []
        if (
            self.enable_row_detection
            and not self._row_path_detection_stopped
            and fov_xy_count >= self.row_min_segment_inliers
        ):
            row_records = detect_rows_from_xy(
                xy_fov,
                distance_threshold=self.row_distance_threshold,
                max_gap=self.row_max_gap,
                max_iterations=self.row_max_iterations,
                min_segment_inliers=self.row_min_segment_inliers,
                max_rows=self.row_max_rows,
                remove_radius=self.row_remove_radius,
                start_ref_point=(reference.point.x, reference.point.y),
                min_points_left=self.row_min_points_left,
            )
            rows_detected = len(row_records)

        paths_detected = -1
        groups_detected = -1
        published_paths = -1
        published_markers = -1
        published_group_start_distance: float | None = None
        group_start_lines_count = -1
        output_paths: list[dict] = []
        group_start_lines: list[dict] = []
        if self.enable_path_detection and not self._row_path_detection_stopped and row_records:
            path_records = build_paths_from_rows(
                row_records,
                path_width=self.path_width,
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
                    output_paths = self.path_start_tracker.update(output_paths, stamp_ns)
                if self.enable_parallel_correction:
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

        self.get_logger().info(
            f'Lidar messages are received | frame={self.frame_count} '
            f'| stamp={stamp_sec}.{stamp_nsec:09d} '
            f'| width={msg.width} | height={msg.height} | point_step={msg.point_step} '
            f'| raw_points={raw_count} | valid_xyz_points={valid_xyz_count} '
            f'| fov_xy_points={fov_xy_count} '
            f'| rows_detected={rows_detected} '
            f'| paths_detected={paths_detected} '
            f'| groups_detected={groups_detected} '
            f'| group_start_lines={group_start_lines_count} '
            f'| group_start_distance={published_group_start_distance} '
            f'| detection_stopped={self._row_path_detection_stopped} '
            f'| published_paths={published_paths} '
            f'| published_markers={published_markers}'
        )

        if (self.enable_notebook_plot or self.enable_plot_video) and (self.frame_count % self.plot_every_n_frames == 0):
            self._update_notebook_plot(xy_fov, row_records, output_paths, group_start_lines)


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
