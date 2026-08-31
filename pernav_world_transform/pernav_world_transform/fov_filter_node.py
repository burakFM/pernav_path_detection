from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from example_interfaces.msg import Float64
from geometry_msgs.msg import Point, PoseArray
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker


@dataclass
class FovGeometry:
    reference: np.ndarray
    direction: np.ndarray
    normal: np.ndarray
    min_along: float
    max_along: float
    lower_offset: float
    upper_offset: float


class FovFilterNode(Node):
    def __init__(self) -> None:
        super().__init__('fov_filter_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('input_pointcloud_topic', '/pcl_world'),
                ('end_of_row_topic', '/end_of_row'),
                ('distance_topic', '/distance_to_eor'),
                ('output_pointcloud_topic', '/pcl_world_fov'),
                ('activation_distance', 4.0),
                ('upper_offset', 4.0),
                ('lower_offset', -0.5),
                ('fov_line_extension', 0.0),
                ('debug', True),
                ('plot_output_path', 'fov_world_snapshot.png'),
            ],
        )

        self.input_pointcloud_topic = str(self.get_parameter('input_pointcloud_topic').value)
        self.end_of_row_topic = str(self.get_parameter('end_of_row_topic').value)
        self.distance_topic = str(self.get_parameter('distance_topic').value)
        self.output_pointcloud_topic = str(self.get_parameter('output_pointcloud_topic').value)
        self.activation_distance = float(self.get_parameter('activation_distance').value)
        self.upper_offset = float(self.get_parameter('upper_offset').value)
        self.lower_offset = float(self.get_parameter('lower_offset').value)
        self.fov_line_extension = max(0.0, float(self.get_parameter('fov_line_extension').value))
        self.debug = bool(self.get_parameter('debug').value)
        self.plot_output_path = str(self.get_parameter('plot_output_path').value)

        self.latest_end_of_row: np.ndarray | None = None
        self.latest_world_cloud: np.ndarray | None = None
        self.previous_distance: float | None = None
        self.activation_requested = False
        self.fov_active = False
        self.fov_geometry: FovGeometry | None = None
        self._last_debug_log_sec = -1.0
        self._debug_interval_sec = 1.0

        self.cloud_publisher = self.create_publisher(PointCloud2, self.output_pointcloud_topic, 10)
        self.marker_publisher = self.create_publisher(Marker, '/fov_markers', 10)
        self.create_subscription(PoseArray, self.end_of_row_topic, self.end_of_row_callback, 10)
        self.create_subscription(Float64, self.distance_topic, self.distance_callback, 10)
        self.create_subscription(PointCloud2, self.input_pointcloud_topic, self.pointcloud_callback, 10)

        self.get_logger().info('World-frame FOV filter started')
        self.get_logger().info(f'Input cloud: {self.input_pointcloud_topic}')
        self.get_logger().info(f'End-of-row: {self.end_of_row_topic}')
        self.get_logger().info(f'Distance trigger: {self.distance_topic}')
        self.get_logger().info(f'Output cloud: {self.output_pointcloud_topic}')
        self.get_logger().info(f'Activation threshold: {self.activation_distance:.3f} m')

    @staticmethod
    def _pointcloud2_xyz(msg: PointCloud2) -> np.ndarray:
        field_names = {field.name for field in msg.fields}
        if not {'x', 'y', 'z'}.issubset(field_names):
            return np.empty((0, 3), dtype=np.float32)

        if hasattr(pc2, 'dtype_from_fields'):
            dtype = pc2.dtype_from_fields(msg.fields, msg.point_step)
            raw = np.frombuffer(msg.data, dtype=dtype, count=int(msg.width) * int(msg.height))
            xyz = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(np.float32, copy=False)
            return xyz[np.isfinite(xyz).all(axis=1)]

        points = np.asarray(
            list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)),
            dtype=np.float32,
        )
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return points.reshape(-1, 3)

    @staticmethod
    def _build_xyz_cloud(points_xyz: np.ndarray, source_msg: PointCloud2) -> PointCloud2:
        points = np.asarray(points_xyz, dtype=np.float32)
        cloud = PointCloud2()
        cloud.header.stamp = source_msg.header.stamp
        cloud.header.frame_id = 'world'
        cloud.height = 1
        cloud.width = int(points.shape[0])
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        cloud.data = points.tobytes()
        return cloud

    def end_of_row_callback(self, msg: PoseArray) -> None:
        positions = np.array(
            [[pose.position.x, pose.position.y] for pose in msg.poses],
            dtype=np.float64,
        )
        if positions.shape[0] < 2 or not np.isfinite(positions).all():
            self.get_logger().warning('Received invalid end-of-row estimate; need at least two finite poses.')
            return

        self.latest_end_of_row = positions
        if self.activation_requested and not self.fov_active:
            self.create_fov()

    def distance_callback(self, msg: Float64) -> None:
        current_distance = float(msg.data)
        if not np.isfinite(current_distance):
            return

        previous_distance = self.previous_distance
        self.previous_distance = current_distance
        if (
            not self.activation_requested
            and previous_distance is not None
            and previous_distance > self.activation_distance
            and current_distance <= self.activation_distance
        ):
            self.activation_requested = True
            self.get_logger().info(
                'FOV activation threshold crossed: '
                f'previous distance = {previous_distance:.3f} m, '
                f'current distance = {current_distance:.3f} m'
            )
            if self.latest_end_of_row is None:
                self.get_logger().warning(
                    'Threshold crossed before a valid end-of-row estimate; waiting for one.'
                )
            else:
                self.create_fov()

    @staticmethod
    def _fit_end_of_row_line(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        reference = np.mean(positions, axis=0)
        centered = positions - reference
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]

        ordered_direction = positions[-1] - positions[0]
        if np.dot(direction, ordered_direction) < 0.0:
            direction = -direction
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        projections = centered @ direction
        return reference, direction, normal, float(np.min(projections)), float(np.max(projections))

    def create_fov(self) -> None:
        if self.fov_active or self.latest_end_of_row is None:
            return

        reference, direction, normal, min_along, max_along = self._fit_end_of_row_line(self.latest_end_of_row)
        min_along -= self.fov_line_extension
        max_along += self.fov_line_extension
        self.fov_geometry = FovGeometry(
            reference=reference,
            direction=direction,
            normal=normal,
            min_along=min_along,
            max_along=max_along,
            lower_offset=self.lower_offset,
            upper_offset=self.upper_offset,
        )
        self.fov_active = True
        self.get_logger().info(
            'FOV created and frozen in world frame\n'
            f'Number of end-of-row poses: {len(self.latest_end_of_row)}\n'
            f'Direction: {direction.tolist()}\n'
            f'Normal: {normal.tolist()}\n'
            f'Along range: [{min_along:.3f}, {max_along:.3f}]\n'
            f'Across range: [{self.lower_offset:.3f}, {self.upper_offset:.3f}]'
        )
        self.publish_fov_markers()
        self.save_fov_plot()

    def filter_points(self, points_xyz: np.ndarray) -> np.ndarray:
        if self.fov_geometry is None:
            return np.empty((0, 3), dtype=np.float32)

        geometry = self.fov_geometry
        relative_xy = points_xyz[:, :2].astype(np.float64, copy=False) - geometry.reference
        along = relative_xy @ geometry.direction
        across = relative_xy @ geometry.normal
        mask = (
            (along >= geometry.min_along)
            & (along <= geometry.max_along)
            & (across >= geometry.lower_offset)
            & (across <= geometry.upper_offset)
        )
        return points_xyz[mask]

    def publish_fov_markers(self) -> None:
        if self.fov_geometry is None:
            return

        geometry = self.fov_geometry
        along_values = np.array([geometry.min_along, geometry.max_along])
        centerline = [geometry.reference + along * geometry.direction for along in along_values]
        lower = [point + geometry.lower_offset * geometry.normal for point in centerline]
        upper = [point + geometry.upper_offset * geometry.normal for point in centerline]

        marker = Marker()
        marker.header.frame_id = 'world'
        marker.ns = 'world_fov'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.06
        marker.color.r = 0.1
        marker.color.g = 0.9
        marker.color.b = 0.2
        marker.color.a = 1.0
        marker.lifetime.sec = 0
        marker.points = [
            Point(x=float(centerline[0][0]), y=float(centerline[0][1]), z=0.0),
            Point(x=float(centerline[1][0]), y=float(centerline[1][1]), z=0.0),
            Point(x=float(lower[0][0]), y=float(lower[0][1]), z=0.0),
            Point(x=float(lower[1][0]), y=float(lower[1][1]), z=0.0),
            Point(x=float(upper[0][0]), y=float(upper[0][1]), z=0.0),
            Point(x=float(upper[1][0]), y=float(upper[1][1]), z=0.0),
        ]
        self.marker_publisher.publish(marker)

    def save_fov_plot(self) -> None:
        if self.fov_geometry is None or self.latest_end_of_row is None:
            return
        if self.latest_world_cloud is None:
            self.get_logger().warning('FOV created, but no /pcl_world scan is available for the plot.')
            return

        try:
            import matplotlib

            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.get_logger().warning(f'Could not save FOV plot because matplotlib is unavailable: {exc}')
            return

        geometry = self.fov_geometry
        along_values = np.array([geometry.min_along, geometry.max_along])
        centerline = np.asarray(
            [geometry.reference + along * geometry.direction for along in along_values],
            dtype=np.float64,
        )
        lower = centerline + geometry.lower_offset * geometry.normal
        upper = centerline + geometry.upper_offset * geometry.normal

        output_path = Path(self.plot_output_path).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure, axis = plt.subplots(figsize=(12, 8))
        cloud_xy = self.latest_world_cloud[:, :2]
        axis.scatter(cloud_xy[:, 0], cloud_xy[:, 1], s=1, c='lightgray', alpha=0.55, label='/pcl_world')
        axis.scatter(
            self.latest_end_of_row[:, 0],
            self.latest_end_of_row[:, 1],
            s=28,
            c='black',
            marker='x',
            label='end_of_row points',
        )
        axis.plot(centerline[:, 0], centerline[:, 1], color='gold', linewidth=2.5, label='FOV centerline')
        axis.plot(upper[:, 0], upper[:, 1], color='red', linewidth=2.0, label='+upper offset')
        axis.plot(lower[:, 0], lower[:, 1], color='blue', linewidth=2.0, label='-lower offset')
        axis.set_title('Frozen world-frame FOV')
        axis.set_xlabel('world X [m]')
        axis.set_ylabel('world Y [m]')
        axis.set_aspect('equal', adjustable='box')
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        self.get_logger().info(f'Saved frozen FOV plot to {output_path}')

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        points_xyz = self._pointcloud2_xyz(msg)
        if points_xyz.size == 0:
            return
        self.latest_world_cloud = points_xyz.copy()

        if not self.fov_active:
            return

        filtered_points = self.filter_points(points_xyz)
        self.cloud_publisher.publish(self._build_xyz_cloud(filtered_points, msg))

        if self.debug:
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            if now_sec - self._last_debug_log_sec >= self._debug_interval_sec:
                self.get_logger().info(
                    f'input points: {len(points_xyz)} | points inside FOV: {len(filtered_points)}'
                )
                self._last_debug_log_sec = now_sec


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FovFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()