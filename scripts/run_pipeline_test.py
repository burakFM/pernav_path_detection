#!/usr/bin/env python3
"""Run the PerNav pipeline against one rosbag and collect test artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Any


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY = SCRIPT_PATH.parent.parent
WORKSPACE = REPOSITORY.parent.parent
PROJECT_ROOT = next(
    (parent for parent in WORKSPACE.parents if (parent / "data").is_dir()),
    WORKSPACE.parent,
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_PARAMS = REPOSITORY / "pernav_path_node/config/path_pipeline.params.yaml"
ROS_SETUP = Path("/opt/ros/jazzy/setup.bash")

LAUNCH_TEMPLATE = """from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    run = Path(__file__).resolve().parent
    return LaunchDescription([
        Node(
            package='pernav_world_transform',
            executable='rearaxle_to_world_node',
            output='screen',
            parameters=[{{
                'enable_chassis_exclusion': True,
                'chassis_exclusion_x_min': 0.0,
                'chassis_exclusion_x_max': 4.0,
                'chassis_exclusion_y_min': -1.0,
                'chassis_exclusion_y_max': 1.0,
            }}],
        ),
        Node(
            package='pernav_world_transform',
            executable='roi_filter_node',
            output='screen',
            parameters=[{{
                'activation_distance': 1.0,
                'distance_message_type': '{distance_message_type}',
                'roi_line_extension': 10.0,
                'group_start_near_threshold': 0.2,
                'deactivation_distance': 5.0,
                'eor_deactivation_distance': 10.0,
                'path_pipeline_reset_topic': '/pernav/path_pipeline_reset',
                'plot_output_path': str(run / 'roi_snapshot.png'),
            }}],
        ),
        Node(
            package='pernav_path_node',
            executable='path_pipeline',
            name='path_pipeline_node',
            output='screen',
            parameters=[
                str(run / 'path_pipeline.params.yaml'),
                {{
                    'enable_notebook_plot': False,
                    'enable_plot_video': {enable_video},
                    'plot_video_output_path': str(run / 'path_plot.mp4'),
                    'plot_video_fps': 10.0,
                    'plot_every_n_frames': 1,
                }},
            ],
        ),
    ])
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PerNav ROS 2 pipeline against one bag and create a self-contained output folder."
    )
    parser.add_argument("bag", type=Path, help="Input .mcap file or rosbag directory")
    parser.add_argument(
        "--output-name",
        help="Output folder name (default: pipeline_<bag-name>_<timestamp>)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Parent output directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--domain-id", type=int, default=192, help="ROS_DOMAIN_ID (default: 192)")
    parser.add_argument("--rate", type=float, default=1.0, help="Rosbag playback rate (default: 1.0)")
    parser.add_argument("--skip-build", action="store_true", help="Use the existing workspace build")
    parser.add_argument("--no-record", action="store_true", help="Do not record all ROS topics")
    parser.add_argument("--no-video", action="store_true", help="Disable MP4 plot recording")
    args = parser.parse_args()

    if not args.bag.expanduser().exists():
        parser.error(f"bag does not exist: {args.bag}")
    if not 0 <= args.domain_id <= 232:
        parser.error("--domain-id must be between 0 and 232")
    if args.rate <= 0:
        parser.error("--rate must be positive")
    return args


def require_commands(commands: list[str], environment: dict[str, str]) -> None:
    missing = [
        command
        for command in commands
        if shutil.which(command, path=environment.get("PATH")) is None
    ]
    if missing:
        raise RuntimeError(f"required command(s) not found: {', '.join(missing)}")


def resolve_bag_path(requested_path: Path) -> Path:
    path = requested_path.expanduser().resolve()
    if path.is_file():
        return path
    if (path / "metadata.yaml").is_file():
        return path

    candidates = sorted(metadata.parent for metadata in path.rglob("metadata.yaml"))
    if len(candidates) == 1:
        print(f"Resolved parent directory to rosbag: {candidates[0]}")
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            f"no rosbag metadata.yaml found in directory or its descendants: {path}"
        )
    candidate_list = "\n  ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        f"multiple rosbags found below {path}; pass one specific bag directory:\n  {candidate_list}"
    )


def detect_distance_message_type(
    bag_path: Path,
    environment: dict[str, str],
) -> str:
    result = subprocess.run(
        ["ros2", "bag", "info", str(bag_path)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"ros2 bag info failed for {bag_path}: {detail}")

    match = re.search(
        r"Topic: /distance_to_eor \| Type: (\S+) \|",
        result.stdout,
    )
    if match is None:
        raise RuntimeError("required topic /distance_to_eor is missing from the input bag")
    message_type = match.group(1)
    supported_types = {
        "example_interfaces/msg/Float64",
        "std_msgs/msg/Float64",
    }
    if message_type not in supported_types:
        raise RuntimeError(
            f"unsupported /distance_to_eor type {message_type}; "
            f"supported types: {', '.join(sorted(supported_types))}"
        )
    print(f"Detected /distance_to_eor type: {message_type}")
    return message_type


def sourced_environment(
    domain_id: int,
    ros_log_dir: Path,
    *,
    include_workspace: bool = True,
) -> dict[str, str]:
    workspace_setup = WORKSPACE / "install/setup.bash"
    if not ROS_SETUP.exists():
        raise RuntimeError(f"ROS setup file not found: {ROS_SETUP}")
    if include_workspace and not workspace_setup.exists():
        raise RuntimeError(f"workspace setup file not found: {workspace_setup}; run without --skip-build")

    setup_commands = [f"source {ROS_SETUP}"]
    if include_workspace:
        setup_commands.append(f"source {workspace_setup}")
    command = " && ".join([*setup_commands, "env -0"])
    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        stdout=subprocess.PIPE,
    )
    environment = {}
    for entry in result.stdout.split(b"\0"):
        if entry and b"=" in entry:
            key, value = entry.split(b"=", 1)
            environment[key.decode()] = value.decode()
    environment["ROS_DOMAIN_ID"] = str(domain_id)
    environment["ROS_LOG_DIR"] = str(ros_log_dir)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def run_logged(
    command: list[str],
    log_path: Path,
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(map(str, command))}")
    with log_path.open("w", encoding="utf-8") as log_file:
        return subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )


def start_logged(
    command: list[str],
    log_path: Path,
    *,
    cwd: Path,
    environment: dict[str, str],
) -> tuple[subprocess.Popen[str], IO[str]]:
    print(f"$ {' '.join(map(str, command))}")
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return process, log_file


def stop_process(process: subprocess.Popen[str] | None, name: str, timeout: float = 30.0) -> None:
    if process is None or process.poll() is not None:
        return
    print(f"Stopping {name}...")
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def create_run_folder(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_name = args.bag.expanduser().resolve().stem.replace("rosbag2_", "")
    output_name = args.output_name or f"pipeline_{bag_name}_{timestamp}"
    run_folder = args.output_root.expanduser().resolve() / output_name
    run_folder.mkdir(parents=True, exist_ok=False)
    (run_folder / "ros_logs").mkdir()
    return run_folder


def prepare_run(
    run_folder: Path,
    args: argparse.Namespace,
    distance_message_type: str,
) -> str:
    shutil.copy2(DEFAULT_PARAMS, run_folder / "path_pipeline.params.yaml")
    (run_folder / "pipeline.launch.py").write_text(
        LAUNCH_TEMPLATE.format(
            enable_video=str(not args.no_video),
            distance_message_type=distance_message_type,
        ),
        encoding="utf-8",
    )

    commit = subprocess.run(
        ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (run_folder / "source_commit.txt").write_text(f"{commit}\n", encoding="utf-8")

    snapshot = run_folder / "source_snapshot"
    snapshot.mkdir()
    for relative_path in (
        "pernav_path_node/pernav_path_node/path_pipeline_node.py",
        "pernav_path_node/pernav_path_node/pipeline_helpers.py",
        "pernav_path_node/pernav_path_node/path_start_tracker.py",
        "pernav_path_node/pernav_path_node/plot_video.py",
        "pernav_world_transform/pernav_world_transform/rearaxle_to_world_node.py",
        "pernav_world_transform/pernav_world_transform/roi_filter_node.py",
    ):
        shutil.copy2(REPOSITORY / relative_path, snapshot)
    return commit


def count_pipeline_metrics(log_text: str) -> dict[str, int]:
    frame_lines = [line for line in log_text.splitlines() if "Lidar messages are received" in line]
    return {
        "processed_frames": len(frame_lines),
        "frames_with_rows": sum(bool(re.search(r"rows_detected=[1-9]", line)) for line in frame_lines),
        "frames_with_paths": sum(bool(re.search(r"paths_detected=[1-9]", line)) for line in frame_lines),
        "frames_detection_stopped": sum("detection_stopped=True" in line for line in frame_lines),
        "roi_cycles_created": log_text.count("created and frozen in world frame"),
        "roi_cycles_deactivated": log_text.count("deactivated by"),
        "pipeline_resets": log_text.count("Path pipeline cycle reset"),
        "detection_stop_events": log_text.count("Row and path detection are now latched off"),
    }


def parse_bag_info(text: str) -> dict[str, Any]:
    def match_number(label: str, number_pattern: str = r"([0-9]+)") -> str | None:
        match = re.search(rf"^{re.escape(label)}:\s+{number_pattern}", text, re.MULTILINE)
        return match.group(1) if match else None

    topic_counts = {
        match.group(1): int(match.group(2))
        for match in re.finditer(r"Topic: (\S+) \|.*?Count: ([0-9]+)", text)
    }
    return {
        "topics": len(topic_counts),
        "messages": int(match_number("Messages") or 0),
        "duration_seconds": float(match_number("Duration", r"([0-9.]+)s") or 0),
        "topic_message_counts": topic_counts,
    }


def validate_video(run_folder: Path) -> dict[str, Any]:
    video_path = run_folder / "path_plot.mp4"
    if not video_path.exists():
        return {"enabled": False, "full_decode_passed": False}

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration",
            "-of", "json", str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video_path), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    stream = json.loads(probe.stdout)["streams"][0] if probe.returncode == 0 else {}
    return {
        "enabled": True,
        "path": video_path.name,
        "size_bytes": video_path.stat().st_size,
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": stream.get("r_frame_rate"),
        "frames": int(stream.get("nb_read_frames", 0)),
        "duration_seconds": float(stream.get("duration", 0)),
        "full_decode_passed": probe.returncode == 0 and decode.returncode == 0,
        "decode_error": decode.stderr.strip() or None,
    }


def log_error_count(text: str) -> int:
    return sum(
        bool(re.search(r"\[ERROR\]|Traceback|process has died", line, re.IGNORECASE))
        for line in text.splitlines()
    )


def write_summary(
    run_folder: Path,
    args: argparse.Namespace,
    commit: str,
    playback_returncode: int,
    requested_bag: Path,
) -> tuple[dict[str, Any], bool]:
    def read_log(name: str) -> str:
        path = run_folder / name
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    pipeline_text = read_log("pipeline.log")
    player_text = read_log("rosbag_play.log")
    recorder_text = read_log("recorder.log")

    recorded_bag: dict[str, Any] = {"enabled": not args.no_record}
    bag_valid = args.no_record
    if not args.no_record:
        bag_info = run_logged(
            ["ros2", "bag", "info", "recorded_all_topics"],
            run_folder / "bag_info.txt",
            cwd=run_folder,
            environment=sourced_environment(args.domain_id, run_folder / "ros_logs"),
        )
        bag_text = (run_folder / "bag_info.txt").read_text(encoding="utf-8", errors="replace")
        recorded_bag.update(parse_bag_info(bag_text))
        recorded_bag["path"] = "recorded_all_topics"
        recorded_bag["integrity_check_passed"] = bag_info.returncode == 0
        bag_valid = bag_info.returncode == 0

    video = {"enabled": False, "full_decode_passed": True} if args.no_video else validate_video(run_folder)
    errors = {
        "pipeline_log_errors": log_error_count(pipeline_text),
        "recorder_log_errors": log_error_count(recorder_text),
        "player_log_errors": log_error_count(player_text),
    }
    summary = {
        "run": {
            "source_bag": str(args.bag.expanduser().resolve()),
            "requested_bag": str(requested_bag),
            "source_commit": commit,
            "ros_domain_id": args.domain_id,
            "playback_rate": args.rate,
            "playback_completed": playback_returncode == 0,
            "all_processes_stopped": True,
        },
        "recorded_bag": recorded_bag,
        "processing": count_pipeline_metrics(pipeline_text),
        "video": video,
        "snapshots": sorted(path.name for path in run_folder.glob("roi_snapshot*.png")),
        "validation": errors,
        "warnings": [
            "The recorder may not embed the std_msgs/msg/Empty definition for /pernav/path_pipeline_reset."
        ] if "Message definition for topic '/pernav/path_pipeline_reset'" in recorder_text else [],
    }
    success = (
        playback_returncode == 0
        and bag_valid
        and video["full_decode_passed"]
        and not any(errors.values())
    )
    summary["validation"]["passed"] = success
    (run_folder / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary, success


def main() -> int:
    args = parse_args()
    requested_bag = args.bag.expanduser().resolve()
    try:
        args.bag = resolve_bag_path(requested_bag)
    except RuntimeError as error:
        print(f"Input bag error: {error}", file=sys.stderr)
        return 2
    base_environment = sourced_environment(
        args.domain_id,
        args.output_root.expanduser().resolve(),
        include_workspace=False,
    )
    require_commands(
        ["bash", "git", "ros2"]
        + ([] if args.skip_build else ["colcon"])
        + ([] if args.no_video else ["ffmpeg", "ffprobe"]),
        base_environment,
    )
    try:
        distance_message_type = detect_distance_message_type(args.bag, base_environment)
    except RuntimeError as error:
        print(f"Input bag error: {error}", file=sys.stderr)
        return 2

    run_folder = create_run_folder(args)
    commit = prepare_run(run_folder, args, distance_message_type)

    if not args.skip_build:
        build_log = run_folder / "build.log"
        build = run_logged(
            [
                "colcon", "build", "--packages-select",
                "pernav_world_transform", "pernav_path_node", "--symlink-install",
            ],
            build_log,
            cwd=WORKSPACE,
            environment={**base_environment, "ROS_LOG_DIR": str(run_folder / "ros_logs")},
        )
        if build.returncode != 0:
            print(f"Build failed. See {build_log}", file=sys.stderr)
            return build.returncode

    environment = sourced_environment(args.domain_id, run_folder / "ros_logs")
    launch_process: subprocess.Popen[str] | None = None
    recorder_process: subprocess.Popen[str] | None = None
    log_files: list[IO[str]] = []
    playback_returncode = 1

    print(f"Output folder: {run_folder}")
    try:
        launch_process, launch_log = start_logged(
            ["ros2", "launch", str(run_folder / "pipeline.launch.py")],
            run_folder / "pipeline.log",
            cwd=run_folder,
            environment=environment,
        )
        log_files.append(launch_log)
        time.sleep(3)
        if launch_process.poll() is not None:
            raise RuntimeError(f"pipeline launch exited early with code {launch_process.returncode}")

        if not args.no_record:
            recorder_process, recorder_log = start_logged(
                [
                    "ros2", "bag", "record", "-a", "--include-hidden-topics",
                    "--storage", "mcap", "--storage-preset-profile", "zstd_fast",
                    "-o", "recorded_all_topics",
                ],
                run_folder / "recorder.log",
                cwd=run_folder,
                environment=environment,
            )
            log_files.append(recorder_log)
            time.sleep(2)
            if recorder_process.poll() is not None:
                raise RuntimeError(f"recorder exited early with code {recorder_process.returncode}")

        playback = run_logged(
            ["ros2", "bag", "play", str(args.bag), "--rate", str(args.rate)],
            run_folder / "rosbag_play.log",
            cwd=run_folder,
            environment=environment,
        )
        playback_returncode = playback.returncode
    except KeyboardInterrupt:
        print("Interrupted; finalizing available artifacts...", file=sys.stderr)
        playback_returncode = 130
    except Exception as error:
        print(f"Run failed: {error}", file=sys.stderr)
    finally:
        stop_process(launch_process, "pipeline")
        stop_process(recorder_process, "recorder")
        for log_file in log_files:
            log_file.close()

    summary, success = write_summary(
        run_folder,
        args,
        commit,
        playback_returncode,
        requested_bag,
    )
    print(f"Summary: {run_folder / 'summary.json'}")
    print(
        f"Frames: {summary['processing']['processed_frames']} | "
        f"ROI cycles: {summary['processing']['roi_cycles_created']} | "
        f"Validation: {'PASSED' if success else 'FAILED'}"
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())