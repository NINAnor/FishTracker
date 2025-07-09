"""
This file is part of Fish Tracker.
Copyright 2021, VTT Technical research centre of Finland Ltd.
Developed by: Mikael Uimonen.

Fish Tracker is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Fish Tracker is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Fish Tracker.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass

import cv2
from PyQt5 import QtCore, QtWidgets

import fishtracker.utils.file_handler as fh
from fishtracker.core.detection.detector import Detector, DetectorParameters
from fishtracker.core.fish.fish_manager import FishManager
from fishtracker.core.tracking.tracker import Tracker, TrackingState
from fishtracker.managers.playback_manager import PlaybackManager, TestFigure
from fishtracker.parameters.filter_parameters import FilterParameters
from fishtracker.utils.save_manager import SaveManager


def getDefaultParser(getArgs=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--display",
        default=False,
        action="store_true",
        help="display frames as the patch is processed",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=argparse.FileType("r"),
        nargs="*",
        help=".aris file(s) to be processed",
    )
    parser.add_argument(
        "-t",
        "--test",
        default=False,
        action="store_true",
        help="use test file (if exists)",
    )
    if getArgs:
        return parser.parse_args()
    else:
        return parser


def getFiles(args=None):
    files = []
    if args is not None and args.file:
        files = [f.name for f in args.file]
    elif args is not None and args.test:
        files = [fh.getTestFilePath()]
    else:
        dir = fh.getLatestDirectory()
        filePathTuple = QtWidgets.QFileDialog.getOpenFileNames(
            None, "Open File", dir, "Sonar Files (*.aris *.ddf)"
        )
        files = [f for f in filePathTuple[0]]
        try:
            fh.setLatestDirectory(os.path.dirname(files[0]))
        except IndexError:
            pass

    return files


def writeToFile(value, mode="a"):
    with open("track_process_io.txt", mode) as f:
        f.write(str(value) + "\n")


@dataclass
class TrackProcessInfo:
    display: bool
    file: str
    save_directory: str
    connection: mp.connection._ConnectionBase
    params_detector_dict: dict = None
    params_tracker_dict: dict = None
    secondary_tracking: bool = False
    test_file: bool = False

    # Save detections to a text file
    save_detections: bool = False

    # Save tracks to a text file
    save_tracks: bool = False

    # Save complete results with save manager
    save_complete: bool = False

    # Save results as a binary file
    as_binary: bool = True

    # Flow direction setting
    flow_direction: str = "left-to-right"

    # Export all tracks or apply filtering thresholds
    export_all_tracks: bool = False


class TrackProcess(QtCore.QObject):
    """
    TrackProcess launches individual PlaybackManager, Detector and Tracker,
    separate from the ones associated with the UI.
    These are used for the tracking process of the file provided in the track method.
    Each file is intended to be processed with its own TrackProcess instance.
    """

    exit_signal = QtCore.pyqtSignal()

    def __init__(
        self, app: QtWidgets.QApplication, info: TrackProcessInfo, params_tracker=None
    ):
        super().__init__()
        self.app = app
        self.info = info
        self.display = info.display
        self.file = info.file
        self.save_directory = os.path.abspath(info.save_directory)
        self.connection = info.connection
        self.test_file = info.test_file
        self.secondary_tracking = info.secondary_tracking
        self.secondary_tracking_started = False
        self.alive = True
        self.figure = None

        self.save_detections = info.save_detections
        self.save_tracks = info.save_tracks
        self.save_complete = info.save_complete
        self.binary = info.as_binary

        self.logger = logging.getLogger(__name__)

        if info.display:
            self.main_window = QtWidgets.QMainWindow()
        else:
            self.main_window = None

        self.playback_manager = PlaybackManager(self.app, self.main_window)

        self.detector = Detector(self.playback_manager)
        self.tracker = Tracker(self.detector)

        if params_tracker is not None:
            self.tracker.setAllParameters(params_tracker)
        self.setParametersFromDict(info.params_detector_dict, info.params_tracker_dict)

        self.fish_manager = FishManager(self.playback_manager, self.tracker)

        # Set flow direction based on configuration
        if info.flow_direction == "right-to-left":
            self.fish_manager.setUpDownInversion(True)
        else:  # default to left-to-right
            self.fish_manager.setUpDownInversion(False)

        # Apply filter parameters from tracker configuration if not exporting all tracks
        if not info.export_all_tracks:
            filter_params = self.tracker.filter_parameters
            min_duration = filter_params.getParameter(
                FilterParameters.ParametersEnum.min_duration
            )
            mad_limit = filter_params.getParameter(
                FilterParameters.ParametersEnum.mad_limit
            )
            self.fish_manager.setMinDetections(min_duration)
            self.fish_manager.setMAD(mad_limit)
            self.logger.info(
                f"Applied filter parameters - min_duration: {min_duration}, "
                f"mad_limit: {mad_limit}"
            )

        self.save_manager = SaveManager(
            self.playback_manager, self.detector, self.tracker, self.fish_manager
        )

        self.playback_manager.fps = 100
        self.playback_manager.runInThread(self.listenConnection)

        self.logger = logging.getLogger(__name__)

    def setParametersFromDict(
        self, params_detector_dict: dict, params_tracker_dict: dict
    ):
        if params_detector_dict is not None:
            params_detector = DetectorParameters(
                self.detector.bg_subtractor.mog_parameters
            )
            params_detector.setParameterDict(params_detector_dict)
            self.detector.parameters = params_detector

        if params_tracker_dict is not None:
            # params_tracker = AllTrackerParameters(
            #     TrackerParameters(), FilterParameters(), TrackerParameters()
            # )
            # params_tracker.setParameterDict(params_tracker_dict)
            # self.tracker.setAllParameters(params_tracker)
            self.tracker.setAllParametersFromDict(params_tracker_dict)

    def writeToConnection(self, value):
        if self.connection:
            self.connection.send(value)

    def forwardImage(self, tuple):
        """
        Default function for forwarding the image, does not visualize the result.
        """
        ind, frame = tuple
        detections = self.detector.getDetection(ind)

    def forwardImageDisplay(self, tuple):
        """
        If the progress is visualized, this is used to forward the image.
        """
        ind, frame = tuple
        detections = self.detector.getDetection(ind)

        image = cv2.applyColorMap(frame, cv2.COLORMAP_OCEAN)
        image = self.tracker.visualize(image, ind)
        self.figure.displayImage((ind, image))

    def startTrackingProcess(self):
        """
        Initiates detecting and tracking. Called from an event (mapping_done)
        when the playback_manager is ready to feed frames.
        """
        self.detector.initMOG()
        self.detector.computeAll()
        self.tracker.primaryTrack()

        if self.secondary_tracking:
            self.secondary_tracking_started = True
            self.fish_manager.secondaryTrack(self.tracker.filter_parameters)

        if self.display:
            self.playback_manager.play()

    def track(self):
        """
        Handles the tracking process. Opens file and connects detection and tracking
        calls to the appropriate signals, so that they can be started when the file
        has been loaded.
        """
        if self.test_file:
            self.playback_manager.openTestFile()
        else:
            self.playback_manager.loadFile(self.file)

        if self.display:
            self.playback_manager.frame_available.connect(self.forwardImageDisplay)
        else:
            self.playback_manager.frame_available.connect(self.forwardImage)

        self.detector.bg_subtractor.mog_parameters.nof_bg_frames = 500
        self.detector._show_detections = True
        self.playback_manager.mapping_done.connect(self.startTrackingProcess)
        self.tracker.all_computed_signal.connect(self.onAllComputed)

        if self.display:
            self.figure = TestFigure(self.playback_manager.togglePlay)
            self.main_window.setCentralWidget(self.figure)

        if self.display:
            self.main_window.show()

    def listenConnection(self):
        """
        Listens the connection for messages. Currently, only terminate message (-1) is
        supported, but others should be easy to add when needed.
        """
        while self.alive:
            if self.connection.poll():
                id, msg = self.connection.recv()
                if id == -1:
                    self.connection.send((-1, "Terminating"))
                    self.quit()
            else:
                time.sleep(0.5)

    def getSaveFilePath(self, end_string):
        """
        Formats the save file path. Detections and tracks are separated based on
        end_string.
        """
        base_name = os.path.basename(self.file)
        file_name = os.path.splitext(base_name)[0]
        file_path = os.path.join(self.save_directory, f"{file_name}{end_string}")
        return file_path

    def saveResults(self):
        """
        Saves and/or exports results to the directory provided earlier.
        """
        file_name = os.path.splitext(self.file)[0]
        if self.save_detections:
            det_path = self.getSaveFilePath("_dets.txt")
            self.detector.saveDetectionsToFile(det_path)

        if self.save_tracks:
            track_path = self.getSaveFilePath("_tracks.txt")
            self.fish_manager.saveToFile(track_path)

        if self.save_complete:
            save_path = self.getSaveFilePath(".fish")
            self.save_manager.saveFile(save_path, self.binary)

        self.logger.info(f"Results saved to {self.save_directory}")

    def onAllComputed(self, tracking_state):
        """
        Saves and quits the process.
        """
        if not self.secondary_tracking or tracking_state == TrackingState.SECONDARY:
            self.saveResults()
            self.quit()

    def quit(self):
        self.alive = False
        self.app.quit()


def trackProcess(process_info: TrackProcessInfo, params_tracker=None):
    app = QtWidgets.QApplication(sys.argv)
    process = TrackProcess(app, process_info, params_tracker=params_tracker)
    logger = logging.getLogger(__name__)
    logger.info("Starting track process")
    process.track()
    sys.exit(app.exec_())


# TODO: Fix test code
if __name__ == "__main__":
    save_directory = fh.getLatestSaveDirectory()
    args = getDefaultParser(getArgs=True)
    file = getFiles(args)
    info = TrackProcessInfo(
        file=file[0], save_directory=save_directory, test_file=args.test
    )
    trackProcess(args.display, info)
