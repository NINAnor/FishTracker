import logging
import os
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from fishtracker.core.processing.batch_track import BatchTrack, ProcessState
from fishtracker.core.tracking.tracker import (
    AllTrackerParameters,
    FilterParameters,
    TrackerParameters,
)
from fishtracker.parameters.detector_parameters import DetectorParameters


@hydra.main(version_base=None, config_path="configs", config_name="default.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.input.file_paths is None:
        raise ValueError("No input file paths provided in the configuration.")
    if cfg.output.directory is None or cfg.output.directory == "<output_dir>":
        raise ValueError("No output directory provided in the configuration.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    # this environement varaible needs to be set for Qt to work in headless mode
    # (e.g., when running in a server environment without a display)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    logger = logging.getLogger(__name__)

    logger.info("Batch processing started")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Accept a list of image paths from the config
    files = [Path(p) for p in cfg.input.file_paths]
    logger.info(f"Found {len(files)} files to process")

    output_path = Path(cfg.output.directory)
    if not output_path.exists():
        logger.info(f"Creating output directory: {output_path}")
        os.makedirs(output_path)

    detector_params = DetectorParameters(**cfg.detector)

    tracker_params = AllTrackerParameters(
        primary=TrackerParameters(**cfg.tracker.primary_tracking),
        filter=FilterParameters(**cfg.tracker.filtering),
        secondary=TrackerParameters(**cfg.tracker.secondary_tracking),
    )

    batch_track = BatchTrack(
        display=False,
        files=files,
        save_directory=cfg.output.directory,
        parallel=cfg.batch_processing.parallel,
        params_detector=detector_params,
        params_tracker=tracker_params,
        secondary_track=True,
        save_detections=cfg.output.save_detections,
        save_tracks=cfg.output.save_tracks,
        save_complete=cfg.output.save_fish,
        flow_direction=cfg.input.flow_direction,
    )

    time.sleep(0.1)
    batch_track.beginTrack()

    while batch_track.state != ProcessState.FINISHED:
        time.sleep(0.1)

    # Start app for signal brokering
    logger.info("Batch processing finished")


if __name__ == "__main__":
    main()
