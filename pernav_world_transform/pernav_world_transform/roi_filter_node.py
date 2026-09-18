from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from example_interfaces.msg import Float64 as EorDistance
from geometry_msgs.msg import Point, PoseArray, TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Empty, Float64 as StdFloat64
from visualization_msgs.msg import Marker


@dataclass
class RoiGeometry:
    reference: np.ndarray
    direction: np.ndarray
    normal: np.ndarray
    min_along: float
    max_along: float
    lower_offset: float
    upper_offset: float


class RoiFilterNode(Node):
    def __init__(self) -> None:
        super().__init__('roi_filter_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('input_pointcloud_topic', '/pcl_world'),
                ('end_of_row_topic', '/end_of_row'),
                ('distance_topic', '/distance_to_eor'),
                ('distance_message_type', 'example_interfaces/msg/Float64'),
                ('output_pointcloud_topic', '/pcl_world_roi'),
                ('pose_topic', '/pose_rearAxle2worldGNSS'),
                ('axes_marker_topic', '/roi_markers_axes'),
                ('activation_distance', 1.0),
                ('upper_offset', 4.0),
                ('lower_offset', -0.5),
                ('roi_line_extension', 10.0),
                ('group_start_distance_topic', '/pernav/group_start_line_distance'),
                ('group_start_near_threshold', 0.2),
                ('deactivation_distance', 5.0),
                ('eor_deactivation_distance', 10.0),
                ('path_pipeline_reset_topic', '/pernav/path_pipeline_reset'),
                ('debug', True),
                ('plot_output_path', 'roi_world_snapshot.png'),
            ],
        )

        self.input_pointcloud_topic = str(self.get_parameter('input_pointcloud_topic').value)
        self.end_of_row_topic = str(self.get_parameter('end_of_row_topic').value)
        self.distance_topic = str(self.get_parameter('distance_topic').value)
        self.distance_message_type = str(self.get_parameter('distance_message_type').value)
        self.output_pointcloud_topic = str(self.get_parameter('output_pointcloud_topic').value)
        self.pose_topic = str(self.get_parameter('pose_topic').value)
        self.axes_marker_topic = str(self.get_parameter('axes_marker_topic').value)
        self.activation_distance = float(self.get_parameter('activation_distance').value)
        self.upper_offset = float(self.get_parameter('upper_offset').value)
        self.lower_offset = float(self.get_parameter('lower_offset').value)
        self.roi_line_extension = max(0.0, float(self.get_parameter('roi_line_extension').value))
        self.group_start_distance_topic = str(
            self.get_parameter('group_start_distance_topic').value
        )
        self.group_start_near_threshold = float(
            self.get_parameter('group_start_near_threshold').value
        )
        self.deactivation_distance = float(self.get_parameter('deactivation_distance').value)
        self.eor_deactivation_distance = float(
            self.get_parameter('eor_deactivation_distance').value
        )
        self.path_pipeline_reset_topic = str(
            self.get_parameter('path_pipeline_reset_topic').value
        )
        if (
            not np.isfinite(self.group_start_near_threshold)
            or self.group_start_near_threshold < 0.0
            or not np.isfinite(self.deactivation_distance)
            or self.deactivation_distance <= self.group_start_near_threshold
        ):
            raise ValueError(
                'ROI deactivation distance must be finite and greater than the '
                'nonnegative group-start near threshold.'
            )
        if (
            not np.isfinite(self.eor_deactivation_distance)
            or self.eor_deactivation_distance <= self.activation_distance
        ):
            raise ValueError(
                'End-of-row deactivation distance must be finite and greater than '
                'the ROI activation distance.'
            )
        self.debug = bool(self.get_parameter('debug').value)
        self.plot_output_path = str(self.get_parameter('plot_output_path').value)

        self.latest_end_of_row: np.ndarray | None = None
        self.latest_end_of_row_axes: np.ndarray | None = None
        self.latest_tractor_xy: np.ndarray | None = None
        self.roi_axis_base: np.ndarray | None = None
        self.roi_axis_direction: np.ndarray | None = None
        self.latest_world_cloud: np.ndarray | None = None
        self.previous_distance: float | None = None
        self.activation_requested = False
        self.roi_active = False
        self.roi_geometry: RoiGeometry | None = None
        self.previous_group_start_distance: float | None = None
        self.deactivation_armed = False
        self.eor_deactivation_armed = False
        self.roi_cycle_count = 0
        self._last_debug_log_sec = -1.0
        self._debug_interval_sec = 1.0

        self.cloud_publisher = self.create_publisher(PointCloud2, self.output_pointcloud_topic, 10)
        self.marker_publisher = self.create_publisher(Marker, '/roi_markers', 10)
        self.axes_marker_publisher = self.create_publisher(Marker, self.axes_marker_topic, 10)
        self.path_pipeline_reset_publisher = self.create_publisher(
            Empty, self.path_pipeline_reset_topic, 10,
        )
        self.create_subscription(PoseArray, self.end_of_row_topic, self.end_of_row_callback, 10)
        self.create_subscription(TransformStamped, self.pose_topic, self.pose_callback, 10)
        distance_types = {
            'example_interfaces/msg/Float64': EorDistance,
            'std_msgs/msg/Float64': StdFloat64,
        }
        if self.distance_message_type not in distance_types:
            raise ValueError(
                'distance_message_type must be example_interfaces/msg/Float64 '
                'or std_msgs/msg/Float64.'
            )
        self.create_subscription(
            distance_types[self.distance_message_type],
            self.distance_topic,
            self.distance_callback,
            10,
        )
        self.create_subscription(
            StdFloat64,
            self.group_start_distance_topic,
            self.group_start_distance_callback,
            10,
        )
        self.create_subscription(PointCloud2, self.input_pointcloud_topic, self.pointcloud_callback, 10)

        self.get_logger().info('World-frame ROI filter started')
        self.get_logger().info(f'Input cloud: {self.input_pointcloud_topic}')
        self.get_logger().info(f'End-of-row: {self.end_of_row_topic}')
        self.get_logger().info(
            f'Distance trigger: {self.distance_topic} ({self.distance_message_type})'
        )
        self.get_logger().info(f'Output cloud: {self.output_pointcloud_topic}')
        self.get_logger().info(f'ROI axis grid: {self.axes_marker_topic}')
        self.get_logger().info(f'Activation threshold: {self.activation_distance:.3f} m')
        self.get_logger().info(
            f'Cycle deactivation: arm at <= {self.group_start_near_threshold:.3f} m on '
            f'{self.group_start_distance_topic}, deactivate on rise to '
            f'{self.deactivation_distance:.3f} m; or rearm {self.distance_topic} at >= '
            f'{self.eor_deactivation_distance:.3f} m and deactivate on its next falling edge'
        )
        self.get_logger().info(f'Path-cycle reset topic: {self.path_pipeline_reset_topic}')

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
        positions = []
        axes = []
        for pose in msg.poses:
            position = np.array([pose.position.x, pose.position.y], dtype=np.float64)
            quaternion = np.array([
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ], dtype=np.float64)
            quaternion_norm = float(np.linalg.norm(quaternion))
            if not np.isfinite(position).all() or not np.isfinite(quaternion).all() or quaternion_norm <= 0.0:
                continue
            rotation = Rotation.from_quat(quaternion / quaternion_norm)
            local_axes = rotation.apply([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ])[:, :2]
            axis_norms = np.linalg.norm(local_axes, axis=1)
            if np.any(axis_norms <= 1e-9):
                continue
            positions.append(position)
            axes.append(local_axes / axis_norms[:, np.newaxis])
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        if positions.shape[0] < 2:
            self.get_logger().warning('Received invalid end-of-row estimate; need at least two finite poses.')
            return

        self.latest_end_of_row = positions
        self.latest_end_of_row_axes = np.asarray(axes, dtype=np.float64).reshape(-1, 2, 2)
        if self.activation_requested and not self.roi_active:
            self.create_roi()

    def pose_callback(self, msg: TransformStamped) -> None:
        position = np.array([
            msg.transform.translation.x,
            msg.transform.translation.y,
        ], dtype=np.float64)
        if np.isfinite(position).all():
            self.latest_tractor_xy = position

    def distance_callback(self, msg: EorDistance | StdFloat64) -> None:
        current_distance = float(msg.data)
        if not np.isfinite(current_distance):
            return

        previous_distance = self.previous_distance
        self.previous_distance = current_distance

        if self.roi_active:
            if not self.eor_deactivation_armed:
                if current_distance >= self.eor_deactivation_distance:
                    self.eor_deactivation_armed = True
                    self.get_logger().info(
                        'ROI distance-to-EOR fallback armed: '
                        f'distance reached {current_distance:.3f} m '
                        f'(threshold {self.eor_deactivation_distance:.3f} m)'
                    )
            elif (
                previous_distance is not None
                and previous_distance >= self.eor_deactivation_distance
                and current_distance < self.eor_deactivation_distance
            ):
                self.deactivate_roi(
                    trigger='distance_to_eor falling edge',
                    current_distance=current_distance,
                )
            return

        if (
            not self.activation_requested
            and previous_distance is not None
            and previous_distance > self.activation_distance
            and current_distance <= self.activation_distance
        ):
            self.activation_requested = True
            self.get_logger().info(
                'ROI activation threshold crossed: '
                f'previous distance = {previous_distance:.3f} m, '
                f'current distance = {current_distance:.3f} m'
            )
            if self.latest_end_of_row is None:
                self.get_logger().warning(
                    'Threshold crossed before a valid end-of-row estimate; waiting for one.'
                )
            else:
                self.create_roi()

    def group_start_distance_callback(self, msg: StdFloat64) -> None:
        """Arm near the start line, then end this ROI cycle on the 5 m rise."""
        if not self.roi_active:
            return

        current_distance = float(msg.data)
        if not np.isfinite(current_distance):
            return
        previous_distance = self.previous_group_start_distance
        self.previous_group_start_distance = current_distance

        if not self.deactivation_armed:
            if (
                previous_distance is not None
                and previous_distance > self.group_start_near_threshold
                and current_distance <= self.group_start_near_threshold
            ):
                self.deactivation_armed = True
                self.get_logger().info(
                    'ROI deactivation armed at the group start line: '
                    f'{previous_distance:.3f} -> {current_distance:.3f} m'
                )
            return

        if (
            previous_distance is not None
            and previous_distance < self.deactivation_distance
            and current_distance >= self.deactivation_distance
        ):
            self.deactivate_roi(
                trigger='group-start rising edge',
                current_distance=current_distance,
            )

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

    def create_roi(self) -> None:
        if self.roi_active or self.latest_end_of_row is None:
            return

        reference, direction, normal, min_along, max_along = self._fit_end_of_row_line(self.latest_end_of_row)
        min_along -= self.roi_line_extension
        max_along += self.roi_line_extension
        self.roi_geometry = RoiGeometry(
            reference=reference,
            direction=direction,
            normal=normal,
            min_along=min_along,
            max_along=max_along,
            lower_offset=self.lower_offset,
            upper_offset=self.upper_offset,
        )
        self._freeze_roi_axis_direction()
        self.roi_cycle_count += 1
        self.roi_active = True
        self.deactivation_armed = False
        self.eor_deactivation_armed = False
        self.previous_group_start_distance = None
        self.get_logger().info(
            f'ROI cycle {self.roi_cycle_count} created and frozen in world frame\n'
            f'Number of end-of-row poses: {len(self.latest_end_of_row)}\n'
            f'Direction: {direction.tolist()}\n'
            f'Normal: {normal.tolist()}\n'
            f'Along range: [{min_along:.3f}, {max_along:.3f}]\n'
            f'Across range: [{self.lower_offset:.3f}, {self.upper_offset:.3f}]'
        )
        self.publish_roi_markers()
        self.publish_roi_axes_marker()
        self.save_roi_plot()

    def deactivate_roi(self, *, trigger: str, current_distance: float) -> None:
        """Clear this ROI cycle and reset the path pipeline for the next one."""
        if not self.roi_active:
            return

        completed_cycle = self.roi_cycle_count
        self.roi_active = False
        self.roi_geometry = None
        self.activation_requested = False
        self.deactivation_armed = False
        self.eor_deactivation_armed = False
        self.previous_group_start_distance = None
        # Require a fresh end-of-row estimate for the next ROI geometry.
        self.latest_end_of_row = None
        self.latest_end_of_row_axes = None
        self.roi_axis_base = None
        self.roi_axis_direction = None

        self.publish_roi_marker_delete()
        self.publish_roi_axes_marker_delete()
        self.path_pipeline_reset_publisher.publish(Empty())
        self.get_logger().info(
            f'ROI cycle {completed_cycle} deactivated by {trigger} at '
            f'{current_distance:.3f} m; /pcl_world_roi paused and path pipeline reset.'
        )

    def filter_points(self, points_xyz: np.ndarray) -> np.ndarray:
        if self.roi_geometry is None:
            return np.empty((0, 3), dtype=np.float32)

        geometry = self.roi_geometry
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

    def publish_roi_markers(self) -> None:
        if self.roi_geometry is None:
            return

        geometry = self.roi_geometry
        along_values = np.array([geometry.min_along, geometry.max_along])
        centerline = [geometry.reference + along * geometry.direction for along in along_values]
        lower = [point + geometry.lower_offset * geometry.normal for point in centerline]
        upper = [point + geometry.upper_offset * geometry.normal for point in centerline]

        marker = Marker()
        marker.header.frame_id = 'world'
        marker.ns = 'world_roi'
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

    @staticmethod
    def _clip_line_to_roi(
        base: np.ndarray,
        line_direction: np.ndarray,
        geometry: RoiGeometry,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        relative = base - geometry.reference
        origin_coordinates = np.array([
            relative @ geometry.direction,
            relative @ geometry.normal,
        ], dtype=float)
        direction_coordinates = np.array([
            line_direction @ geometry.direction,
            line_direction @ geometry.normal,
        ], dtype=float)
        minimums = np.array([geometry.min_along, geometry.lower_offset], dtype=float)
        maximums = np.array([geometry.max_along, geometry.upper_offset], dtype=float)
        t_min = -np.inf
        t_max = np.inf
        for origin, delta, minimum, maximum in zip(
            origin_coordinates, direction_coordinates, minimums, maximums,
        ):
            if abs(delta) < 1e-12:
                if origin < minimum or origin > maximum:
                    return None
                continue
            first = (minimum - origin) / delta
            second = (maximum - origin) / delta
            t_min = max(t_min, min(first, second))
            t_max = min(t_max, max(first, second))
            if t_min > t_max:
                return None
        return base + t_min * line_direction, base + t_max * line_direction

    def _build_roi_axis_segments(self) -> list[tuple[np.ndarray, np.ndarray]]:
        if (
            self.roi_geometry is None
            or self.latest_end_of_row is None
            or self.roi_axis_base is None
            or self.roi_axis_direction is None
        ):
            return []
        geometry = self.roi_geometry
        positions = self.latest_end_of_row
        projections = (positions - geometry.reference) @ geometry.direction
        ordered = np.sort(projections)
        gaps = np.diff(ordered)
        valid_gaps = gaps[gaps > 1e-6]
        if valid_gaps.size == 0:
            return []
        spacing = float(np.median(valid_gaps))
        base = self.roi_axis_base
        axis = self.roi_axis_direction
        base_projection = float((base - geometry.reference) @ geometry.direction)
        first_step = int(np.ceil((geometry.min_along - base_projection) / spacing))
        last_step = int(np.floor((geometry.max_along - base_projection) / spacing))

        segments = []
        for step in range(first_step, last_step + 1):
            line_base = base + step * spacing * geometry.direction
            segment = self._clip_line_to_roi(line_base, axis, geometry)
            if segment is not None:
                segments.append(segment)
        return segments

    def _freeze_roi_axis_direction(self) -> None:
        self.roi_axis_base = None
        self.roi_axis_direction = None
        if (
            self.latest_end_of_row is None
            or self.latest_end_of_row_axes is None
            or self.latest_tractor_xy is None
        ):
            return
        positions = self.latest_end_of_row
        closest_index = int(np.argmin(np.linalg.norm(positions - self.latest_tractor_xy, axis=1)))
        base = positions[closest_index]
        approach = base - self.latest_tractor_xy
        approach_norm = float(np.linalg.norm(approach))
        if approach_norm <= 1e-9:
            return
        approach /= approach_norm
        axes = self.latest_end_of_row_axes[closest_index]
        selected_index = int(np.argmax(np.abs(axes @ approach)))
        selected_axis = axes[selected_index].copy()
        if selected_axis @ approach < 0.0:
            selected_axis = -selected_axis
        self.roi_axis_base = base.copy()
        self.roi_axis_direction = selected_axis
        self.get_logger().info(
            f'ROI axis grid selected local {"X" if selected_index == 0 else "Y"} axis '
            f'at closest EOR point; |alignment|={abs(float(selected_axis @ approach)):.3f}'
        )

    def publish_roi_axes_marker(self) -> None:
        segments = self._build_roi_axis_segments()
        if not segments:
            self.get_logger().warning(
                'ROI created, but axis-grid marker needs valid EOR spacing and tractor pose.'
            )
            return
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'world_roi_axes'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.04
        marker.color.r = 0.0
        marker.color.g = 0.75
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.points = [
            Point(x=float(point[0]), y=float(point[1]), z=0.03)
            for segment in segments for point in segment
        ]
        self.axes_marker_publisher.publish(marker)
        self.get_logger().info(
            f'Published ROI axis grid: {len(segments)} lines on {self.axes_marker_topic}'
        )

    def publish_roi_marker_delete(self) -> None:
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'world_roi'
        marker.id = 0
        marker.action = Marker.DELETE
        self.marker_publisher.publish(marker)

    def publish_roi_axes_marker_delete(self) -> None:
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'world_roi_axes'
        marker.id = 0
        marker.action = Marker.DELETE
        self.axes_marker_publisher.publish(marker)

    def _plot_output_path_for_cycle(self) -> Path:
        output_path = Path(self.plot_output_path).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        if self.roi_cycle_count > 1:
            output_path = output_path.with_name(
                f'{output_path.stem}_cycle_{self.roi_cycle_count:03d}{output_path.suffix}'
            )
        return output_path

    def save_roi_plot(self) -> None:
        if self.roi_geometry is None or self.latest_end_of_row is None:
            return
        if self.latest_world_cloud is None:
            self.get_logger().warning('ROI created, but no /pcl_world scan is available for the plot.')
            return

        try:
            import matplotlib

            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.get_logger().warning(f'Could not save ROI plot because matplotlib is unavailable: {exc}')
            return

        geometry = self.roi_geometry
        along_values = np.array([geometry.min_along, geometry.max_along])
        centerline = np.asarray(
            [geometry.reference + along * geometry.direction for along in along_values],
            dtype=np.float64,
        )
        lower = centerline + geometry.lower_offset * geometry.normal
        upper = centerline + geometry.upper_offset * geometry.normal

        output_path = self._plot_output_path_for_cycle()
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
        axis.plot(centerline[:, 0], centerline[:, 1], color='gold', linewidth=2.5, label='ROI centerline')
        axis.plot(upper[:, 0], upper[:, 1], color='red', linewidth=2.0, label='+upper offset')
        axis.plot(lower[:, 0], lower[:, 1], color='blue', linewidth=2.0, label='-lower offset')
        axis.set_title('Frozen world-frame ROI')
        axis.set_xlabel('world X [m]')
        axis.set_ylabel('world Y [m]')
        axis.set_aspect('equal', adjustable='box')
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        self.get_logger().info(f'Saved frozen ROI plot to {output_path}')

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        points_xyz = self._pointcloud2_xyz(msg)
        if points_xyz.size == 0:
            return
        self.latest_world_cloud = points_xyz.copy()

        if not self.roi_active:
            return

        filtered_points = self.filter_points(points_xyz)
        self.cloud_publisher.publish(self._build_xyz_cloud(filtered_points, msg))

        if self.debug:
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            if now_sec - self._last_debug_log_sec >= self._debug_interval_sec:
                self.get_logger().info(
                    f'input points: {len(points_xyz)} | points inside ROI: {len(filtered_points)}'
                )
                self._last_debug_log_sec = now_sec


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoiFilterNode()
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
