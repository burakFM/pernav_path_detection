from __future__ import annotations

from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker


class RoiCloudAccumulatorNode(Node):
    """Publish a rolling accumulation of every ROI cloud."""

    def __init__(self) -> None:
        super().__init__('roi_cloud_accumulator_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('input_pointcloud_topic', '/pcl_world_roi'),
                ('output_pointcloud_topic', '/pernav/pcl_world_roi_accumulated'),
                ('roi_marker_topic', '/roi_markers'),
                # Zero means accept every input cloud.  A positive value can
                # optionally be used to decimate an unusually fast source.
                ('sampling_period_sec', 0.0),
                ('accumulated_cloud_count', 10),
                ('path_pipeline_reset_topic', '/pernav/path_pipeline_reset'),
            ],
        )

        self.input_pointcloud_topic = str(self.get_parameter('input_pointcloud_topic').value)
        self.output_pointcloud_topic = str(self.get_parameter('output_pointcloud_topic').value)
        self.roi_marker_topic = str(self.get_parameter('roi_marker_topic').value)
        self.sampling_period_sec = float(self.get_parameter('sampling_period_sec').value)
        self.accumulated_cloud_count = int(self.get_parameter('accumulated_cloud_count').value)
        self.path_pipeline_reset_topic = str(
            self.get_parameter('path_pipeline_reset_topic').value
        )
        if not np.isfinite(self.sampling_period_sec) or self.sampling_period_sec < 0.0:
            raise ValueError('sampling_period_sec must be finite and non-negative.')
        if self.accumulated_cloud_count < 1:
            raise ValueError('accumulated_cloud_count must be at least one.')

        self._clouds: deque[np.ndarray] = deque(maxlen=self.accumulated_cloud_count)
        self._last_accepted_stamp_ns: int | None = None
        self._last_log_sec = -1.0
        self._roi_reference: np.ndarray | None = None
        self._roi_direction: np.ndarray | None = None
        self._roi_along_max: float | None = None
        self._roi_across_min: float | None = None
        self._roi_across_max: float | None = None
        self._last_source_msg: PointCloud2 | None = None

        self.publisher = self.create_publisher(PointCloud2, self.output_pointcloud_topic, 10)
        self.create_subscription(
            PointCloud2,
            self.input_pointcloud_topic,
            self.pointcloud_callback,
            10,
        )
        self.create_subscription(
            Empty,
            self.path_pipeline_reset_topic,
            self.reset_callback,
            10,
        )
        self.create_subscription(Marker, self.roi_marker_topic, self.roi_marker_callback, 10)

        self.get_logger().info(
            'ROI cloud accumulator started: '
            f'{self.input_pointcloud_topic} -> {self.output_pointcloud_topic} '
            f'| roi_marker={self.roi_marker_topic} '
            f'| sample_period={self.sampling_period_sec:.3f}s (0=every cloud) '
            f'| cloud_count={self.accumulated_cloud_count}'
        )

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
        return points.reshape(-1, 3) if points.size else np.empty((0, 3), dtype=np.float32)

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

    def roi_marker_callback(self, msg: Marker) -> None:
        """Store the frozen rectangle published by roi_filter_node."""
        if msg.ns != 'world_roi':
            return
        if msg.action in (Marker.DELETE, Marker.DELETEALL):
            self._clear_roi_geometry()
            return
        if msg.type != Marker.LINE_LIST or len(msg.points) < 6:
            return

        center_start = np.array([msg.points[0].x, msg.points[0].y], dtype=np.float64)
        center_end = np.array([msg.points[1].x, msg.points[1].y], dtype=np.float64)
        direction = center_end - center_start
        length = float(np.linalg.norm(direction))
        if not np.isfinite(length) or length <= 1e-9:
            self.get_logger().warning('Ignoring ROI marker with a zero-length centerline.')
            return
        direction /= length
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        lower = np.array([msg.points[2].x, msg.points[2].y], dtype=np.float64)
        upper = np.array([msg.points[4].x, msg.points[4].y], dtype=np.float64)
        across_values = np.array([
            float((lower - center_start) @ normal),
            float((upper - center_start) @ normal),
        ])
        if not np.isfinite(across_values).all():
            self.get_logger().warning('Ignoring ROI marker with non-finite boundaries.')
            return
        self._roi_reference = center_start
        self._roi_direction = direction
        self._roi_along_max = length
        self._roi_across_min = float(np.min(across_values))
        self._roi_across_max = float(np.max(across_values))

    def _clear_roi_geometry(self) -> None:
        self._roi_reference = None
        self._roi_direction = None
        self._roi_along_max = None
        self._roi_across_min = None
        self._roi_across_max = None

    def _clip_to_roi(self, points_xyz: np.ndarray) -> np.ndarray:
        """Enforce the active frozen ROI on the final merged point cloud."""
        if points_xyz.size == 0:
            return points_xyz
        if (
            self._roi_reference is None
            or self._roi_direction is None
            or self._roi_along_max is None
            or self._roi_across_min is None
            or self._roi_across_max is None
        ):
            return np.empty((0, 3), dtype=np.float32)
        normal = np.array([-self._roi_direction[1], self._roi_direction[0]])
        relative_xy = points_xyz[:, :2].astype(np.float64, copy=False) - self._roi_reference
        along = relative_xy @ self._roi_direction
        across = relative_xy @ normal
        mask = (
            (along >= 0.0)
            & (along <= self._roi_along_max)
            & (across >= self._roi_across_min)
            & (across <= self._roi_across_max)
        )
        return points_xyz[mask]

    def reset_callback(self, _msg: Empty) -> None:
        self._clouds.clear()
        self._last_accepted_stamp_ns = None
        self._clear_roi_geometry()
        if self._last_source_msg is not None:
            self.publisher.publish(
                self._build_xyz_cloud(np.empty((0, 3), dtype=np.float32), self._last_source_msg)
            )
        self.get_logger().info('Cleared accumulated ROI clouds after path-pipeline reset.')

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        if msg.header.frame_id != 'world':
            self.get_logger().warning(
                f'Ignoring cloud in frame {msg.header.frame_id!r}; expected world.'
            )
            return
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if self._last_accepted_stamp_ns is not None:
            elapsed_sec = (stamp_ns - self._last_accepted_stamp_ns) * 1e-9
            if elapsed_sec < 0.0:
                self._clouds.clear()
                self._last_accepted_stamp_ns = None
                self.get_logger().info('Timestamp moved backward; cleared accumulated ROI clouds.')
            elif self.sampling_period_sec > 0.0 and elapsed_sec < self.sampling_period_sec:
                return

        points_xyz = self._pointcloud2_xyz(msg)
        self._clouds.append(points_xyz)
        self._last_accepted_stamp_ns = stamp_ns
        self._last_source_msg = msg
        merged = np.concatenate(tuple(self._clouds), axis=0)
        clipped = self._clip_to_roi(merged)
        self.publisher.publish(self._build_xyz_cloud(clipped, msg))

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec - self._last_log_sec >= 1.0:
            self.get_logger().info(
                f'Published accumulated ROI cloud | clouds={len(self._clouds)}/'
                f'{self.accumulated_cloud_count} | merged_points={len(merged)} '
                f'| roi_points={len(clipped)} '
                f'| published_points={len(clipped)}'
            )
            self._last_log_sec = now_sec


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoiCloudAccumulatorNode()
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
