"""Associate world-frame path starts across scans and average their XY positions."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class PathTrack:
    track_id: int
    mean_xy: np.ndarray
    heading: float
    observations: int = 1
    missed_scans: int = 0


class PathStartTracker:
    def __init__(
        self,
        max_distance: float = 1.0,
        max_heading_deg: float = 20.0,
        min_observations: int = 3,
        candidate_max_missed_scans: int = 5,
        ambiguity_margin: float = 0.15,
    ) -> None:
        if not math.isfinite(max_distance) or max_distance <= 0:
            raise ValueError('Tracking match distance must be finite and positive.')
        if not math.isfinite(max_heading_deg) or not 0 < max_heading_deg <= 180:
            raise ValueError('Tracking heading threshold must be in (0, 180] degrees.')
        if min_observations < 1 or candidate_max_missed_scans < 0:
            raise ValueError('Tracking observations must be >= 1 and missed scans >= 0.')
        if not math.isfinite(ambiguity_margin) or ambiguity_margin < 0:
            raise ValueError('Tracking ambiguity margin must be finite and nonnegative.')
        self.max_distance = max_distance
        self.max_heading = math.radians(max_heading_deg)
        self.min_observations = min_observations
        self.candidate_max_missed_scans = candidate_max_missed_scans
        self.ambiguity_margin = ambiguity_margin
        self.reset()

    def reset(self) -> None:
        self.tracks: dict[int, PathTrack] = {}
        self._next_id = 1
        self._last_stamp_ns: int | None = None

    def confirmed_starts(self) -> list[dict]:
        """Snapshot of all confirmed mean positions, including missing tracks."""
        return [
            dict(path_id=track.track_id, path_start_x=float(track.mean_xy[0]),
                 path_start_y=float(track.mean_xy[1]), observation_count=track.observations)
            for track in sorted(self.tracks.values(), key=lambda item: item.track_id)
            if track.observations >= self.min_observations
        ]

    @staticmethod
    def _geometry(record: dict) -> tuple[np.ndarray, np.ndarray, float] | None:
        start = np.array([record['path_start_x'], record['path_start_y']], dtype=float)
        end = np.array([record['path_end_x'], record['path_end_y']], dtype=float)
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            return None
        delta = end - start
        if np.linalg.norm(delta) < 1e-9:
            return None
        return start, delta, math.atan2(delta[1], delta[0])

    def update(self, paths: list[dict], stamp_ns: int) -> list[dict]:
        """Return confirmed paths seen in this scan, sorted by persistent ID.

        Duplicate timestamps do not count again. Backward time starts a fresh
        tracking session (e.g. bag replay). Confirmed missing tracks are retained
        until reset; tentative tracks expire after too many consecutive misses.
        """
        if self._last_stamp_ns is not None:
            if stamp_ns == self._last_stamp_ns:
                return []
            if stamp_ns < self._last_stamp_ns:
                self.reset()
        self._last_stamp_ns = stamp_ns
        valid = [(record, self._geometry(record)) for record in paths]
        valid = [(record, geometry) for record, geometry in valid if geometry is not None]
        tracks = list(self.tracks.values())
        for track in tracks:
            track.missed_scans += 1

        # Costs are distance plus a small heading penalty. Heading differences
        # wrap at 360 degrees: reversed paths are not equivalent directions.
        costs = np.full((len(tracks), len(valid)), np.inf)
        for i, track in enumerate(tracks):
            for j, (_, (start, _, heading)) in enumerate(valid):
                distance = float(np.linalg.norm(start - track.mean_xy))
                angle = abs(math.atan2(math.sin(heading - track.heading), math.cos(heading - track.heading)))
                if distance <= self.max_distance and angle <= self.max_heading:
                    costs[i, j] = distance + 0.25 * self.max_distance * angle / self.max_heading

        # A competing near-equal match is left unresolved rather than averaged
        # into an uncertain identity. Such detections must not spawn duplicates.
        eligible = np.isfinite(costs)
        ambiguous_tracks = set()
        ambiguous_detections = set()
        if self.ambiguity_margin > 0:
            for i in range(len(tracks)):
                ordered = np.sort(costs[i, eligible[i]])
                if len(ordered) >= 2 and ordered[1] - ordered[0] < self.ambiguity_margin:
                    ambiguous_tracks.add(i)
                    ambiguous_detections.update(np.flatnonzero(eligible[i]).tolist())
            for j in range(len(valid)):
                ordered = np.sort(costs[eligible[:, j], j])
                if len(ordered) >= 2 and ordered[1] - ordered[0] < self.ambiguity_margin:
                    ambiguous_detections.add(j)
                    ambiguous_tracks.update(np.flatnonzero(eligible[:, j]).tolist())
        for i in ambiguous_tracks:
            costs[i, :] = np.inf
        for j in ambiguous_detections:
            costs[:, j] = np.inf

        matches: dict[int, PathTrack] = {}
        if tracks and valid:
            # Dummy columns allow every track to remain unmatched. This penalty
            # prioritizes the number of valid matches, then minimizes total cost.
            unmatched_cost = (len(tracks) + 1) * 2 * self.max_distance
            assignment = np.hstack((costs, np.full((len(tracks), len(tracks)), unmatched_cost)))
            rows, columns = linear_sum_assignment(assignment)
            for i, j in zip(rows, columns):
                if j < len(valid) and math.isfinite(costs[i, j]):
                    track = tracks[i]
                    start, _, heading = valid[j][1]
                    track.observations += 1
                    track.mean_xy += (start - track.mean_xy) / track.observations
                    track.heading = heading
                    track.missed_scans = 0
                    matches[j] = track

        # Unmatched observations near an existing admissible track are withheld:
        # one path cannot contribute twice in a scan or create a duplicate track.
        for j, (_, (start, _, heading)) in enumerate(valid):
            if j in matches or eligible[:, j].any():
                continue
            track = PathTrack(self._next_id, start.copy(), heading)
            self._next_id += 1
            self.tracks[track.track_id] = track
            matches[j] = track

        for track in tracks:
            if (track.observations < self.min_observations
                    and track.missed_scans > self.candidate_max_missed_scans):
                del self.tracks[track.track_id]

        output = []
        id_map = {int(valid[j][0]['path_id']): track.track_id for j, track in matches.items()}
        for j, track in matches.items():
            if track.observations < self.min_observations:
                continue
            record, (start, delta, _) = valid[j]
            result = dict(record)
            result['detection_path_id'] = record['path_id']
            result['path_id'] = track.track_id
            result['observation_count'] = track.observations
            result['detected_start_x'], result['detected_start_y'] = map(float, start)
            result['path_start_x'], result['path_start_y'] = map(float, track.mean_xy)
            # Translate the entire current segment to its averaged start. This
            # preserves the latest heading/length and parallel correction.
            result['path_end_x'], result['path_end_y'] = map(float, track.mean_xy + delta)
            if 'ref_path_id' in result:
                result['ref_path_id'] = id_map.get(int(record['ref_path_id']))
            output.append(result)
        return sorted(output, key=lambda record: record['path_id'])
