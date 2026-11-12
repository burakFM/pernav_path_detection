#!/usr/bin/env python3
"""Copy a timestamp window from an MCAP rosbag without replaying ROS topics."""

from __future__ import annotations

import argparse
from pathlib import Path

import rosbag2_py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Copy a relative time window from one MCAP rosbag to another.'
    )
    parser.add_argument('source', type=Path, help='Input MCAP file or bag directory')
    parser.add_argument('destination', type=Path, help='New output bag directory')
    parser.add_argument('--start-offset', type=float, required=True, help='Start offset in seconds')
    parser.add_argument('--end-offset', type=float, required=True, help='End offset in seconds')
    args = parser.parse_args()
    if not args.source.exists():
        parser.error(f'source does not exist: {args.source}')
    if args.destination.exists():
        parser.error(f'destination already exists: {args.destination}')
    if args.start_offset < 0.0 or args.end_offset <= args.start_offset:
        parser.error('offsets must satisfy 0 <= start-offset < end-offset')
    return args


def main() -> int:
    args = parse_args()
    source_uri = args.source.parent if args.source.is_file() else args.source
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(source_uri), storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''),
    )

    start_time_ns = reader.get_metadata().starting_time.nanoseconds
    window_start_ns = start_time_ns + int(args.start_offset * 1_000_000_000)
    window_end_ns = start_time_ns + int(args.end_offset * 1_000_000_000)

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(args.destination), storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''),
    )
    for topic in reader.get_all_topics_and_types():
        writer.create_topic(topic)

    copied_messages = 0
    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if timestamp_ns >= window_end_ns:
            break
        if timestamp_ns >= window_start_ns:
            writer.write(topic, data, timestamp_ns)
            copied_messages += 1

    print(
        f'Copied {copied_messages} messages from '
        f'{args.start_offset:.3f}s to {args.end_offset:.3f}s into {args.destination}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())