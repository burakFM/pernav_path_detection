"""Publish raw-frame and world-frame RViz outlines for the LiDAR field of view."""

from __future__ import annotations

import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker


@dataclass
class PoseSample:
    stamp_ns: int
    translation: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


class FovMarkerNode(Node):
    """Draw LiDAR FOV border rays in the cloud frame and in the world frame."""

    def __init__(self) -> None:
        super().__init__('fov_marker_node')
        self.declare_parameters(namespace='', parameters=[
            ('input_pointcloud_topic', '/pcl_prep'),
            ('marker_topic', '/pernav/lidar_fov_marker'),
            ('world_marker_topic', '/pernav/lidar_fov_marker_world'),
            ('pose_topic', '/pose_rearAxle2worldGNSS'),
            ('world_frame_id', 'world'),
            ('max_pose_time_difference', 0.1),
            ('pose_buffer_size', 100),
            ('fov_degrees', 110.0),
            ('range_m', 20.0),
            ('marker_z', 0.0),
            ('line_width_m', 0.08),
        ])
        self.input_pointcloud_topic = str(self.get_parameter('input_pointcloud_topic').value)
        self.marker_topic = str(self.get_parameter('marker_topic').value)
        self.world_marker_topic = str(self.get_parameter('world_marker_topic').value)
        self.pose_topic = str(self.get_parameter('pose_topic').value)
        self.world_frame_id = str(self.get_parameter('world_frame_id').value)
        self.max_pose_time_difference_ns = int(
            float(self.get_parameter('max_pose_time_difference').value) * 1e9
        )
        self.pose_buffer_size = max(1, int(self.get_parameter('pose_buffer_size').value))
        self.fov_degrees = float(self.get_parameter('fov_degrees').value)
        self.range_m = float(self.get_parameter('range_m').value)
        self.marker_z = float(self.get_parameter('marker_z').value)
        self.line_width_m = float(self.get_parameter('line_width_m').value)
        if not 0.0 < self.fov_degrees <= 360.0:
            raise ValueError('fov_degrees must be in (0, 360].')
        if not math.isfinite(self.range_m) or self.range_m <= 0.0:
            raise ValueError('range_m must be finite and positive.')
        if not math.isfinite(self.marker_z):
            raise ValueError('marker_z must be finite.')
        if not math.isfinite(self.line_width_m) or self.line_width_m <= 0.0:
            raise ValueError('line_width_m must be finite and positive.')
        if self.max_pose_time_difference_ns < 0:
            raise ValueError('max_pose_time_difference must be non-negative.')
        if not self.world_frame_id:
            raise ValueError('world_frame_id must not be empty.')
        self._pose_buffer: list[PoseSample] = []
        self.marker_publisher = self.create_publisher(Marker, self.marker_topic, 10)
        self.world_marker_publisher = self.create_publisher(Marker, self.world_marker_topic, 10)
        self.pose_subscription = self.create_subscription(
            TransformStamped, self.pose_topic, self._pose_callback, 100,
        )
        self.cloud_subscription = self.create_subscription(
            PointCloud2, self.input_pointcloud_topic, self._cloud_callback, 10,
        )
        self.get_logger().info(
            f'LiDAR FOV marker started: {self.fov_degrees:g} degrees, '
            f'{self.range_m:g} m range; {self.input_pointcloud_topic} -> '
            f'{self.marker_topic} and {self.world_marker_topic} ({self.world_frame_id})',
        )

    @staticmethod
    def _stamp_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _pose_callback(self, pose: TransformStamped) -> None:
        transform = pose.transform
        quaternion = transform.rotation
        norm = math.sqrt(
            quaternion.x * quaternion.x + quaternion.y * quaternion.y
            + quaternion.z * quaternion.z + quaternion.w * quaternion.w
        )
        if norm == 0.0 or not math.isfinite(norm):
            return
        self._pose_buffer.append(PoseSample(
            stamp_ns=self._stamp_ns(pose.header.stamp),
            translation=(transform.translation.x, transform.translation.y, transform.translation.z),
            quaternion=(quaternion.x / norm, quaternion.y / norm, quaternion.z / norm, quaternion.w / norm),
        ))
        if len(self._pose_buffer) > self.pose_buffer_size:
            del self._pose_buffer[:-self.pose_buffer_size]

    def _closest_pose(self, stamp_ns: int) -> PoseSample | None:
        if not self._pose_buffer:
            return None
        pose = min(self._pose_buffer, key=lambda sample: abs(sample.stamp_ns - stamp_ns))
        if abs(pose.stamp_ns - stamp_ns) > self.max_pose_time_difference_ns:
            return None
        return pose

    @staticmethod
    def _rotate(point: Point, quaternion: tuple[float, float, float, float]) -> Point:
        """Rotate a point by a normalized quaternion without an extra dependency."""
        qx, qy, qz, qw = quaternion
        tx = 2.0 * (qy * point.z - qz * point.y)
        ty = 2.0 * (qz * point.x - qx * point.z)
        tz = 2.0 * (qx * point.y - qy * point.x)
        return Point(
            x=point.x + qw * tx + (qy * tz - qz * ty),
            y=point.y + qw * ty + (qz * tx - qx * tz),
            z=point.z + qw * tz + (qx * ty - qy * tx),
        )

    def _marker(self, cloud: PointCloud2, points: list[Point], *, namespace: str) -> Marker:
        marker = Marker()
        marker.header = cloud.header
        marker.ns = namespace
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.line_width_m
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.55, 0.0, 1.0
        marker.points = points
        return marker

    def _cloud_callback(self, cloud: PointCloud2) -> None:
        """Publish two FOV rays, then transform them with the matched world pose."""
        if not cloud.header.frame_id:
            self.get_logger().warning('Skipping FOV marker: /pcl_prep cloud has no frame_id.')
            return
        half_angle = math.radians(self.fov_degrees * 0.5)
        points: list[Point] = []
        for angle in (-half_angle, half_angle):
            points.append(Point(x=0.0, y=0.0, z=self.marker_z))
            points.append(Point(
                x=self.range_m * math.cos(angle),
                y=self.range_m * math.sin(angle),
                z=self.marker_z,
            ))
        self.marker_publisher.publish(self._marker(cloud, points, namespace='lidar_fov'))

        pose = self._closest_pose(self._stamp_ns(cloud.header.stamp))
        if pose is None:
            return
        tx, ty, tz = pose.translation
        world_points = []
        for point in points:
            rotated = self._rotate(point, pose.quaternion)
            world_points.append(Point(x=rotated.x + tx, y=rotated.y + ty, z=rotated.z + tz))
        world_marker = self._marker(cloud, world_points, namespace='lidar_fov_world')
        world_marker.header.frame_id = self.world_frame_id
        self.world_marker_publisher.publish(world_marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FovMarkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # ros2 launch may already have shut down the default context after SIGINT.
        if rclpy.ok():
            rclpy.shutdown()
