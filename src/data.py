from logging import getLogger
from pathlib import Path

import numpy as np

from camera import Camera
from event import Event

logger = getLogger("__main__." + __name__)


def read_data_file(
    file_name: Path,
    n_events: int,
    E0: float,
    cameras: list[Camera],
    start_position: int = 0,
) -> np.ndarray:
    events = []
    skipped_events = 0
    with open(file_name, "r") as data_fh:
        for line_n, line in enumerate(data_fh):
            if line_n >= start_position:
                try:
                    event = Event(line_n, line, E0)
                    # This links the event to the camera(s) in which it occurred
                    event.set_camera_index(cameras)
                    events.append(event)
                except ValueError as e:
                    skipped_events += 1
                    logger.warning(f"Skipping event {line.strip()} REASON: {e}")
                    continue
            if n_events != -1 and line_n >= n_events + start_position - 1:
                break

    logger.info(f"Got {str(len(events))} events")
    logger.warning(f"Skipped {str(skipped_events)} events in data parsing")
    return np.array(events)
