from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2


@dataclass
class PoseSample:
    stamp_sec: float
    rotation: np.ndarray
    translation: np.ndarray


class RearAxleToWorldNode(Node):
    def __init__(self) -> None:
        super().__init__('rearaxle_to_world_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('input_pointcloud_topic', '/pcl_prep'),
                ('pose_topic', '/pose_rearAxle2worldGNSS'),
                ('output_pointcloud_topic', '/pcl_world'),
                ('max_pose_time_difference', 0.1),
                ('pose_buffer_size', 100),
            ],
        )

        self.input_pointcloud_topic = str(self.get_parameter('input_pointcloud_topic').value)
        self.pose_topic = str(self.get_parameter('pose_topic').value)
        self.output_pointcloud_topic = str(self.get_parameter('output_pointcloud_topic').value)
        self.max_pose_time_difference = float(self.get_parameter('max_pose_time_difference').value)
        self.pose_buffer_size = max(2, int(self.get_parameter('pose_buffer_size').value))

        self._pose_buffer: list[PoseSample] = []
        self._last_missing_pose_warn_sec = -1.0
        self._last_debug_log_sec = -1.0
        self._warn_throttle_sec = 2.0
        self._debug_interval_sec = 1.0

        self.cloud_publisher = self.create_publisher(PointCloud2, self.output_pointcloud_topic, 10)

        self.pose_subscription = self.create_subscription(
            TransformStamped,
            self.pose_topic,
            self._pose_callback,
            100,
        )
        self.cloud_subscription = self.create_subscription(
            PointCloud2,
            self.input_pointcloud_topic,
            self._cloud_callback,
            10,
        )

        self.get_logger().info('RearAxle-to-World point cloud transformer started')
        self.get_logger().info(f'Input cloud: {self.input_pointcloud_topic}')
        self.get_logger().info(f'Pose topic: {self.pose_topic}')
        self.get_logger().info(f'Output cloud: {self.output_pointcloud_topic}')

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _pointcloud2_xyz(msg: PointCloud2) -> np.ndarray:
        field_names = {field.name for field in msg.fields}
        if not {'x', 'y', 'z'}.issubset(field_names):
            return np.empty((0, 3), dtype=np.float32)

        if hasattr(pc2, 'dtype_from_fields'):
            dtype = pc2.dtype_from_fields(msg.fields, msg.point_step)
            raw = np.frombuffer(msg.data, dtype=dtype, count=int(msg.width) * int(msg.height))
            xyz = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(np.float32, copy=False)
            finite_mask = np.isfinite(xyz).all(axis=1)
            return xyz[finite_mask]

        points = np.asarray(list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)), dtype=np.float32)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return points.reshape(-1, 3)

    @staticmethod
    def _build_xyz_cloud(points_xyz: np.ndarray, source_msg: PointCloud2) -> PointCloud2:
        points_xyz_f32 = np.asarray(points_xyz, dtype=np.float32)

        cloud = PointCloud2()
        cloud.header.stamp = source_msg.header.stamp
        cloud.header.frame_id = 'world'
        cloud.height = 1
        cloud.width = int(points_xyz_f32.shape[0])
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        cloud.data = points_xyz_f32.tobytes()
        return cloud

    def _pose_callback(self, msg: TransformStamped) -> None:
        stamp_sec = self._stamp_to_sec(msg.header.stamp)

        quat_xyzw = np.array(
            [
                float(msg.transform.rotation.x),
                float(msg.transform.rotation.y),
                float(msg.transform.rotation.z),
                float(msg.transform.rotation.w),
            ],
            dtype=np.float64,
        )
        quat_norm = np.linalg.norm(quat_xyzw)
        if quat_norm <= 0.0:
            self.get_logger().warning('Received pose with zero-norm quaternion; skipping pose sample.')
            return

        quat_xyzw /= quat_norm
        rotation = Rotation.from_quat(quat_xyzw).as_matrix()
        translation = np.array(
            [
                float(msg.transform.translation.x),
                float(msg.transform.translation.y),
                float(msg.transform.translation.z),
            ],
            dtype=np.float64,
        )

        sample = PoseSample(stamp_sec=stamp_sec, rotation=rotation, translation=translation)
        self._insert_pose_sample(sample)

    def _insert_pose_sample(self, sample: PoseSample) -> None:
        if not self._pose_buffer or sample.stamp_sec >= self._pose_buffer[-1].stamp_sec:
            self._pose_buffer.append(sample)
        else:
            stamps = [entry.stamp_sec for entry in self._pose_buffer]
            insert_at = bisect.bisect_left(stamps, sample.stamp_sec)
            self._pose_buffer.insert(insert_at, sample)

        overflow = len(self._pose_buffer) - self.pose_buffer_size
        if overflow > 0:
            del self._pose_buffer[:overflow]

    def _find_closest_pose(self, stamp_sec: float) -> tuple[PoseSample | None, float]:
        if not self._pose_buffer:
            return None, float('inf')

        stamps = [entry.stamp_sec for entry in self._pose_buffer]
        idx = bisect.bisect_left(stamps, stamp_sec)

        candidates: list[PoseSample] = []
        if idx < len(self._pose_buffer):
            candidates.append(self._pose_buffer[idx])
        if idx > 0:
            candidates.append(self._pose_buffer[idx - 1])

        if not candidates:
            return None, float('inf')

        best = min(candidates, key=lambda entry: abs(entry.stamp_sec - stamp_sec))
        delta = abs(best.stamp_sec - stamp_sec)
        return best, delta

    def _throttled_warn(self, message: str) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec - self._last_missing_pose_warn_sec >= self._warn_throttle_sec:
            self.get_logger().warning(message)
            self._last_missing_pose_warn_sec = now_sec

    def _cloud_callback(self, msg: PointCloud2) -> None:
        lidar_stamp_sec = self._stamp_to_sec(msg.header.stamp)
        pose_sample, delta_sec = self._find_closest_pose(lidar_stamp_sec)

        if pose_sample is None:
            self._throttled_warn('No pose samples in buffer yet; skipping LiDAR scan.')
            return

        if delta_sec > self.max_pose_time_difference:
            self._throttled_warn(
                'Closest pose too far in time '
                f'({delta_sec:.3f}s > {self.max_pose_time_difference:.3f}s); skipping LiDAR scan.'
            )
            return

        points_rear = self._pointcloud2_xyz(msg)
        if points_rear.size == 0:
            self._throttled_warn('Incoming PointCloud2 has no valid XYZ points; skipping scan.')
            return

        points_world = points_rear @ pose_sample.rotation.T + pose_sample.translation

        output_msg = self._build_xyz_cloud(points_world, msg)
        self.cloud_publisher.publish(output_msg)

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec - self._last_debug_log_sec >= self._debug_interval_sec:
            self.get_logger().info(
                'LiDAR stamp=%.6f | pose stamp=%.6f | dt=%.4fs | transformed_points=%d'
                % (lidar_stamp_sec, pose_sample.stamp_sec, delta_sec, int(points_world.shape[0]))
            )
            self._last_debug_log_sec = now_sec


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RearAxleToWorldNode()

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
