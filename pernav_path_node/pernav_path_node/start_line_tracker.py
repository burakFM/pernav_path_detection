"""Track a stable path-start line using historical anchors and EOR orientation."""

from __future__ import annotations

import math

import numpy as np


class StartLineTracker:
    def __init__(
        self,
        residual_threshold: float = 0.5,
        min_anchors: int = 2,
        switch_confirmations: int = 5,
        extension_m: float = 2.0,
        max_eor_angle_deg: float = 12.0,
    ) -> None:
        if not math.isfinite(residual_threshold) or residual_threshold <= 0.0:
            raise ValueError('Start-line residual threshold must be finite and positive.')
        if min_anchors < 2 or switch_confirmations < 1:
            raise ValueError('Start-line anchor and confirmation counts are invalid.')
        if not math.isfinite(extension_m) or extension_m < 0.0:
            raise ValueError('Start-line extension must be finite and nonnegative.')
        if not math.isfinite(max_eor_angle_deg) or not 0.0 < max_eor_angle_deg <= 90.0:
            raise ValueError('Start-line EOR angle threshold must be in (0, 90] degrees.')
        self.residual_threshold = residual_threshold
        self.min_anchors = min_anchors
        self.switch_confirmations = switch_confirmations
        self.extension_m = extension_m
        self.max_eor_angle = math.radians(max_eor_angle_deg)
        self.reset()

    def reset(self) -> None:
        self._active_offset: float | None = None
        self._active_centroid: np.ndarray | None = None
        self._active_direction: np.ndarray | None = None
        self._challenger_offset: float | None = None
        self._challenger_frames = 0

    @staticmethod
    def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]], dtype=float)
        residual = float(np.median(np.abs(centered @ normal)))
        return centroid, direction, residual

    def _best_consensus(
        self,
        anchors: list[dict],
        eor_reference: np.ndarray,
        eor_direction: np.ndarray,
        eor_normal: np.ndarray,
    ) -> tuple[float, list[dict], tuple[int, int], np.ndarray, np.ndarray] | None:
        valid = []
        for anchor in anchors:
            point = np.array(
                [anchor['path_start_x'], anchor['path_start_y']], dtype=float,
            )
            if np.isfinite(point).all():
                valid.append((anchor, point, float((point - eor_reference) @ eor_normal)))
        if len(valid) < self.min_anchors:
            return None

        best = None
        for first_index in range(len(valid) - 1):
            for second_index in range(first_index + 1, len(valid)):
                pair_direction = valid[second_index][1] - valid[first_index][1]
                pair_length = float(np.linalg.norm(pair_direction))
                if pair_length < 1e-9:
                    continue
                pair_direction /= pair_length
                pair_angle = math.acos(float(np.clip(
                    abs(pair_direction @ eor_direction), 0.0, 1.0,
                )))
                if pair_angle > self.max_eor_angle:
                    continue
                pair_normal = np.array([-pair_direction[1], pair_direction[0]], dtype=float)
                pair_origin = valid[first_index][1]
                inliers = [
                    item for item in valid
                    if abs(float((item[1] - pair_origin) @ pair_normal))
                    <= self.residual_threshold
                ]
                if len(inliers) < self.min_anchors:
                    continue

                points = np.asarray([item[1] for item in inliers], dtype=float)
                centroid, direction, residual = self._fit_line(points)
                angle = math.acos(float(np.clip(abs(direction @ eor_direction), 0.0, 1.0)))
                if angle > self.max_eor_angle:
                    continue
                score = (
                    len(inliers),
                    sum(min(5, max(1, int(item[0].get('observation_count', 1))))
                        for item in inliers),
                )
                offset = float((centroid - eor_reference) @ eor_normal)
                eor_distance = abs(offset)
                rank = (score[0], score[1], -eor_distance, -residual)
                candidate = (rank, offset, inliers, centroid, direction)
                if best is None or candidate[0] > best[0]:
                    best = candidate

        if best is None or best[0][0] < self.min_anchors:
            return None
        return best[1], [item[0] for item in best[2]], best[0][:2], best[3], best[4]

    def update(
        self,
        anchors: list[dict],
        eor_reference: np.ndarray,
        eor_direction: np.ndarray,
    ) -> list[dict]:
        reference = np.asarray(eor_reference, dtype=float)
        direction = np.asarray(eor_direction, dtype=float)
        length = float(np.linalg.norm(direction))
        if (
            reference.shape != (2,) or not np.isfinite(reference).all()
            or direction.shape != (2,) or not np.isfinite(direction).all() or length < 1e-9
        ):
            return []
        direction /= length
        normal = np.array([-direction[1], direction[0]], dtype=float)
        candidate = self._best_consensus(anchors, reference, direction, normal)
        if candidate is None:
            return []
        candidate_offset, candidate_anchors, candidate_score, candidate_centroid, candidate_direction = candidate

        if self._active_offset is None:
            self._active_offset = candidate_offset
            self._active_centroid = candidate_centroid
            self._active_direction = candidate_direction
        elif abs(candidate_offset - self._active_offset) <= self.residual_threshold:
            self._active_offset = candidate_offset
            self._active_centroid = candidate_centroid
            self._active_direction = candidate_direction
            self._challenger_offset = None
            self._challenger_frames = 0
        else:
            assert self._active_centroid is not None and self._active_direction is not None
            active_normal = np.array(
                [-self._active_direction[1], self._active_direction[0]], dtype=float,
            )
            active_anchors = [
                anchor for anchor in anchors
                if abs(float((np.array([
                    anchor['path_start_x'], anchor['path_start_y'],
                ], dtype=float) - self._active_centroid) @ active_normal))
                <= self.residual_threshold
            ]
            active_score = (
                len(active_anchors),
                sum(min(5, max(1, int(anchor.get('observation_count', 1))))
                    for anchor in active_anchors),
            )
            if candidate_score > active_score:
                if (
                    self._challenger_offset is not None
                    and abs(candidate_offset - self._challenger_offset) <= self.residual_threshold
                ):
                    self._challenger_frames += 1
                else:
                    self._challenger_offset = candidate_offset
                    self._challenger_frames = 1
                if self._challenger_frames >= self.switch_confirmations:
                    self._active_offset = candidate_offset
                    self._active_centroid = candidate_centroid
                    self._active_direction = candidate_direction
                    self._challenger_offset = None
                    self._challenger_frames = 0
            else:
                self._challenger_offset = None
                self._challenger_frames = 0

        assert self._active_centroid is not None and self._active_direction is not None
        active_normal = np.array(
            [-self._active_direction[1], self._active_direction[0]], dtype=float,
        )
        line_anchors = []
        for anchor in anchors:
            point = np.array([anchor['path_start_x'], anchor['path_start_y']], dtype=float)
            if (
                np.isfinite(point).all()
                and abs(float((point - self._active_centroid) @ active_normal))
                <= self.residual_threshold
            ):
                line_anchors.append(point)
        if len(line_anchors) < self.min_anchors:
            return []

        points = np.asarray(line_anchors, dtype=float)
        centroid, active_direction, _ = self._fit_line(points)
        if active_direction @ direction < 0.0:
            active_direction = -active_direction
        projections = (points - centroid) @ active_direction
        start = centroid + (float(projections.min()) - self.extension_m) * active_direction
        end = centroid + (float(projections.max()) + self.extension_m) * active_direction
        return [{
            'group_id': 0,
            'start_x': float(start[0]),
            'start_y': float(start[1]),
            'end_x': float(end[0]),
            'end_y': float(end[1]),
            'anchor_count': len(line_anchors),
        }]