"""Stream Matplotlib plot frames to an MP4 without keeping frames in memory."""
from datetime import datetime
import math
from pathlib import Path
import subprocess

from matplotlib.animation import FFMpegWriter


class _SessionFFMpegWriter(FFMpegWriter):
    def _run(self) -> None:
        # ROS launch/terminal Ctrl+C signals the node's process group. Let the
        # node finish the encoder via EOF, instead of interrupting it mid-frame.
        self._proc = subprocess.Popen(
            self._args(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, start_new_session=True,
        )


class PlotVideoRecorder:
    def __init__(self, output_path: str, fps: float) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('plot_video_fps must be finite and positive.')
        if not output_path.strip():
            raise ValueError('plot_video_output_path must not be empty.')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.output_path = Path(output_path.replace('{timestamp}', timestamp)).expanduser().resolve()
        if self.output_path.suffix.lower() != '.mp4':
            raise ValueError('plot_video_output_path must end in .mp4.')
        if not FFMpegWriter.isAvailable():
            raise RuntimeError('ffmpeg is unavailable; install ffmpeg to record MP4 plots.')
        self._writer = _SessionFFMpegWriter(
            fps=fps, codec='h264',
            extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-threads', '1'],
        )
        self.frame_count = 0
        self._started = False
        self._closed = False

    def capture(self, figure) -> None:
        if self._closed:
            raise RuntimeError('Video recorder is already closed.')
        if not self._started:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.output_path.exists():
                raise FileExistsError(f'Refusing to overwrite existing video: {self.output_path}')
            self._writer.setup(figure, str(self.output_path), dpi=100)
            self._started = True
        self._writer.grab_frame()
        self.frame_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._started:
            self._writer.finish()
