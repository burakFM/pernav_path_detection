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
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

from .pipeline_helpers import (  # type: ignore[reportMissingImports]
    apply_parallel_correction,
    build_group_start_lines,
    build_path_groups,
    build_paths_from_rows,
    detect_rows_from_xy,
    filter_fov,
    remove_xy_box,
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
                ('enable_rectangular_fov_filter', True),
                ('fov_x_min', 0.0),
                ('fov_x_max', 20.0),
                ('fov_y_min', -10.0),
                ('fov_y_max', 10.0),
                ('enable_chassis_exclusion', True),
                ('chassis_exclusion_x_min', 0.0),
                ('chassis_exclusion_x_max', 3.0),
                ('chassis_exclusion_y_min', -1.0),
                ('chassis_exclusion_y_max', 1.0),
                ('enable_row_detection', True),
                ('row_distance_threshold', 0.25),
                ('row_max_gap', 0.6),
                ('row_max_iterations', 400),
                ('row_min_segment_inliers', 40),
                ('row_max_rows', 8),
                ('row_remove_radius', 1.9),
                ('row_min_points_left', 30),
                ('row_start_ref_x', 3.0),
                ('row_start_ref_y', 0.0),
                ('enable_path_detection', True),
                ('path_width', 3.4),
                ('min_path_length', 1.0),
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
                ('plot_every_n_frames', 1),
                ('enable_plot_autoscale', False),
                ('plot_padding_m', 1.0),
                ('plot_x_min', 0.0),
                ('plot_x_max', 20.0),
                ('plot_y_min', -10.0),
                ('plot_y_max', 10.0),
                ('start_line_extension_m', 2.0),
                ('enable_group_start_line_marker', True),
            ],
        )

        self.input_topic = self.get_parameter('input_topic').value
        self.enable_rectangular_fov_filter = bool(
            self.get_parameter('enable_rectangular_fov_filter').value
        )
        self.fov_x_min = float(self.get_parameter('fov_x_min').value)
        self.fov_x_max = float(self.get_parameter('fov_x_max').value)
        self.fov_y_min = float(self.get_parameter('fov_y_min').value)
        self.fov_y_max = float(self.get_parameter('fov_y_max').value)
        self.enable_chassis_exclusion = bool(self.get_parameter('enable_chassis_exclusion').value)
        self.chassis_exclusion_x_min = float(self.get_parameter('chassis_exclusion_x_min').value)
        self.chassis_exclusion_x_max = float(self.get_parameter('chassis_exclusion_x_max').value)
        self.chassis_exclusion_y_min = float(self.get_parameter('chassis_exclusion_y_min').value)
        self.chassis_exclusion_y_max = float(self.get_parameter('chassis_exclusion_y_max').value)
        self.enable_row_detection = bool(self.get_parameter('enable_row_detection').value)
        self.row_distance_threshold = float(self.get_parameter('row_distance_threshold').value)
        self.row_max_gap = float(self.get_parameter('row_max_gap').value)
        self.row_max_iterations = int(self.get_parameter('row_max_iterations').value)
        self.row_min_segment_inliers = int(self.get_parameter('row_min_segment_inliers').value)
        self.row_max_rows = int(self.get_parameter('row_max_rows').value)
        self.row_remove_radius = float(self.get_parameter('row_remove_radius').value)
        self.row_min_points_left = int(self.get_parameter('row_min_points_left').value)
        self.row_start_ref_x = float(self.get_parameter('row_start_ref_x').value)
        self.row_start_ref_y = float(self.get_parameter('row_start_ref_y').value)
        self.enable_path_detection = bool(self.get_parameter('enable_path_detection').value)
        self.path_width = float(self.get_parameter('path_width').value)
        self.min_path_length = float(self.get_parameter('min_path_length').value)
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
        self.plot_every_n_frames = max(1, int(self.get_parameter('plot_every_n_frames').value))
        self.enable_plot_autoscale = bool(self.get_parameter('enable_plot_autoscale').value)
        self.plot_padding_m = max(0.0, float(self.get_parameter('plot_padding_m').value))
        self.plot_x_min = float(self.get_parameter('plot_x_min').value)
        self.plot_x_max = float(self.get_parameter('plot_x_max').value)
        self.plot_y_min = float(self.get_parameter('plot_y_min').value)
        self.plot_y_max = float(self.get_parameter('plot_y_max').value)
        self.start_line_extension_m = float(self.get_parameter('start_line_extension_m').value)
        self.enable_group_start_line_marker = bool(self.get_parameter('enable_group_start_line_marker').value)

        self._plt: Any = None
        self._fig: Any = None
        self._ax: Any = None

        self.path_publisher = self.create_publisher(PoseArray, self.path_output_topic, 10)
        self.marker_publisher = self.create_publisher(MarkerArray, self.marker_output_topic, 10)

        if self.enable_notebook_plot:
            self._init_notebook_plot()

        self.subscription = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.listener_callback,
            10,
        )
        self.subscription
        self.get_logger().info(
            f'Path pipeline node listening on topic {self.input_topic} '
            f'| rectangular_fov_filter={self.enable_rectangular_fov_filter} '
            f'| FOV x:[{self.fov_x_min}, {self.fov_x_max}] '
            f'y:[{self.fov_y_min}, {self.fov_y_max}] '
            f'| row_detection={self.enable_row_detection} '
            f'| path_detection={self.enable_path_detection} '
            f'| parallel_correction={self.enable_parallel_correction} '
            f'| path_pub={self.enable_path_publisher} ({self.path_output_topic}) '
            f'| marker_pub={self.enable_marker_publisher} ({self.marker_output_topic}) '
            f'| notebook_plot={self.enable_notebook_plot} '
            f'| plot_autoscale={self.enable_plot_autoscale} '
            f'| group_start_line_marker={self.enable_group_start_line_marker}'
        )

    def _init_notebook_plot(self) -> None:
        try:
            import matplotlib.pyplot as plt  # Imported lazily to keep runtime deps optional when disabled.

            self._plt = plt
            self._plt.ion()
            self._fig, self._ax = self._plt.subplots(figsize=(11, 9))
        except Exception as exc:
            self.enable_notebook_plot = False
            self.get_logger().warning(f'Notebook-style plotting disabled: {exc}')

    def _update_notebook_plot(
        self,
        xy_fov: np.ndarray,
        row_records: list[dict],
        output_paths: list[dict],
        group_start_lines: list[dict],
    ) -> None:
        if self._ax is None or self._fig is None or self._plt is None:
            return

        ax = self._ax
        ax.clear()

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
            ax.scatter([sx], [sy], s=22, color='green')
            mx, my = 0.5 * (sx + ex), 0.5 * (sy + ey)
            ax.text(mx, my, f"p{pid}", fontsize=8, color=color)

        for ln in group_start_lines:
            ax.plot(
                [float(ln['start_x']), float(ln['end_x'])],
                [float(ln['start_y']), float(ln['end_y'])],
                linestyle='-.',
                linewidth=2.0,
                color='tab:green',
                alpha=0.9,
            )

        if self.enable_plot_autoscale and xy_fov.size > 0:
            x_min = float(np.min(xy_fov[:, 0]))
            x_max = float(np.max(xy_fov[:, 0]))
            y_min = float(np.min(xy_fov[:, 1]))
            y_max = float(np.max(xy_fov[:, 1]))
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

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)

    def close_plot(self) -> None:
        if self._plt is not None and self._fig is not None:
            self._plt.close(self._fig)

    def _publish_debug_markers(
        self,
        src_msg: PointCloud2,
        path_records: list[dict],
        group_start_lines: list[dict],
    ) -> int:
        marker_array = MarkerArray()

        for i, rec in enumerate(path_records):
            marker = Marker()
            marker.header = src_msg.header
            marker.ns = 'pernav_paths'
            marker.id = i
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
        self.frame_count += 1
        stamp_sec = msg.header.stamp.sec
        stamp_nsec = msg.header.stamp.nanosec
        raw_count, xyz = self._pc2_to_xyz(msg)
        xy = xyz[:, :2]
        xy_fov = xy
        if self.enable_rectangular_fov_filter:
            xy_fov = filter_fov(
                xy,
                x_min=self.fov_x_min,
                x_max=self.fov_x_max,
                y_min=self.fov_y_min,
                y_max=self.fov_y_max,
            )
        if self.enable_chassis_exclusion:
            xy_fov = remove_xy_box(
                xy_fov,
                x_min=self.chassis_exclusion_x_min,
                x_max=self.chassis_exclusion_x_max,
                y_min=self.chassis_exclusion_y_min,
                y_max=self.chassis_exclusion_y_max,
            )

        valid_xyz_count = int(xyz.shape[0])
        fov_xy_count = int(xy_fov.shape[0])

        rows_detected = -1
        row_records: list[dict] = []
        if self.enable_row_detection and fov_xy_count >= self.row_min_segment_inliers:
            row_records = detect_rows_from_xy(
                xy_fov,
                distance_threshold=self.row_distance_threshold,
                max_gap=self.row_max_gap,
                max_iterations=self.row_max_iterations,
                min_segment_inliers=self.row_min_segment_inliers,
                max_rows=self.row_max_rows,
                remove_radius=self.row_remove_radius,
                start_ref_point=(self.row_start_ref_x, self.row_start_ref_y),
                min_points_left=self.row_min_points_left,
            )
            rows_detected = len(row_records)

        paths_detected = -1
        groups_detected = -1
        published_paths = -1
        published_markers = -1
        group_start_lines_count = -1
        output_paths: list[dict] = []
        group_start_lines: list[dict] = []
        if self.enable_path_detection and row_records:
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
                group_start_lines = build_group_start_lines(
                    output_paths,
                    extension_m=self.start_line_extension_m,
                )
                group_start_lines_count = len(group_start_lines)
            else:
                groups_detected = 0 if path_records else -1
                output_paths = path_records

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
            f'| published_paths={published_paths} '
            f'| published_markers={published_markers}'
        )

        if self.enable_notebook_plot and (self.frame_count % self.plot_every_n_frames == 0):
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
