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

import sys

from PyQt5 import QtCore, QtWidgets

import file_handler as fh
from batch_dialog import BatchDialog
from detection_list import DetectionDataModel, DetectionList
from detector import Detector
from detector_parameters_view import DetectorParametersView
from echogram_widget import EchogramViewer
from fish_list import FishList
from fish_manager import FishManager
from log_object import LogObject
from main_window import MainWindow
from output_widget import LogToFile, OutputViewer
from parameter_list import ParameterList
from playback_manager import PlaybackManager
from save_manager import SaveManager
from sonar_view3 import Ui_MainWindow
from sonar_widget import SonarViewer
from tracker import Tracker
from tracker_parameters_view import TrackerParametersView
from user_preferences import UserPreferencesDialog


class UIManager:
    def __init__(
        self,
        main_window,
        playback_manager,
        detector,
        tracker,
        fish_manager,
        save_manager,
    ):
        self.widgets_initialized = False
        self.main_window = main_window

        self.playback = playback_manager
        self.detector = detector
        self.tracker = tracker
        self.fish_manager = fish_manager
        self.save_manager = save_manager

        self.ui = Ui_MainWindow()
        self.ui.setupUi(main_window)
        self.main_window.setupStatusBar()
        self.setUpFunctions()

        self.main_window.show()
        self.setupWidgets()
        self.playback.setTitle()

    def setupWidgets(self):
        _translate = QtCore.QCoreApplication.translate

        echo = EchogramViewer(self.playback, self.detector, self.fish_manager)
        self.ui.splitter_2.replaceWidget(0, echo)
        self.ui.echogram_widget = echo
        echo.setMaximumHeight(400)

        self.sonar_viewer = SonarViewer(
            self.main_window,
            self.playback,
            self.detector,
            self.tracker,
            self.fish_manager,
        )
        self.ui.splitter.replaceWidget(0, self.sonar_viewer)
        self.ui.sonar_widget = self.sonar_viewer

        self.tool_bar = ParameterList(
            self.playback,
            self.sonar_viewer.image_processor,
            self.sonar_viewer,
            self.fish_manager,
            self.detector,
            self.tracker,
            echo,
        )
        self.ui.horizontalLayout_2.replaceWidget(self.ui.tool_bar, self.tool_bar)
        self.tool_bar.setMaximumWidth(40)
        self.ui.tool_bar = self.tool_bar

        self.fish_list = FishList(self.fish_manager, self.playback, self.sonar_viewer)
        self.sonar_viewer.measure_event.append(self.fish_list.setMeasurementResult)

        self.detector_parameters = DetectorParametersView(
            self.playback, self.detector, self.sonar_viewer.image_processor
        )

        detection_model = DetectionDataModel(self.detector)
        self.detection_list = DetectionList(detection_model)

        self.tracker_parameters = TrackerParametersView(
            self.playback, self.tracker, self.detector, self.fish_manager
        )
        self.save_manager.file_loaded_event.connect(
            self.tracker_parameters.refreshValues
        )

        self.output = OutputViewer()
        self.output.connectToLogObject()
        self.output.updateLogSignal.connect(self.main_window.updateStatusLog)

        self.logToFile = LogToFile()
        LogObject().connect(self.logToFile.writeLine)

        # Tabs for the side panel.
        self.ui.info_widget.removeTab(0)
        self.ui.info_widget.addTab(self.detector_parameters, "")
        self.ui.info_widget.setTabText(
            self.ui.info_widget.indexOf(self.detector_parameters),
            _translate("MainWindow", "Detector"),
        )
        self.ui.info_widget.addTab(self.detection_list, "")
        self.ui.info_widget.setTabText(
            self.ui.info_widget.indexOf(self.detection_list),
            _translate("MainWindow", "Detections"),
        )
        self.ui.info_widget.addTab(self.tracker_parameters, "")
        self.ui.info_widget.setTabText(
            self.ui.info_widget.indexOf(self.tracker_parameters),
            _translate("MainWindow", "Tracker"),
        )
        self.ui.info_widget.addTab(self.fish_list, "")
        self.ui.info_widget.setTabText(
            self.ui.info_widget.indexOf(self.fish_list),
            _translate("MainWindow", "Tracks"),
        )
        self.ui.info_widget.addTab(self.output, "")
        self.ui.info_widget.setTabText(
            self.ui.info_widget.indexOf(self.output), _translate("MainWindow", "Log")
        )

    def setUpFunctions(self):
        self.ui.menu_File.aboutToShow.connect(self.menuFileAboutToShow)

        self.ui.action_Open.setShortcut("Ctrl+O")
        self.ui.action_Open.triggered.connect(self.openFile)

        self.ui.action_Batch.setShortcut("Ctrl+B")
        self.ui.action_Batch.triggered.connect(self.runBatch)

        self.ui.action_FlowDir.setShortcut("Ctrl+D")
        self.ui.action_FlowDir.triggered.connect(self.changeFlowDirection)

        self.ui.action_UserPref.setShortcut("Ctrl+U")
        self.ui.action_UserPref.triggered.connect(self.openUserPreferences)

        if fh.getTestFilePath() is not None:
            self.ui.action_OpenTest = QtWidgets.QAction(self.main_window)
            self.ui.action_OpenTest.setObjectName("action_OpenTest")
            self.ui.menu_File.addAction(self.ui.action_OpenTest)
            self.ui.action_OpenTest.setShortcut("Ctrl+T")
            self.ui.action_OpenTest.triggered.connect(self.openTestFile)
            self.ui.action_OpenTest.setText(
                QtCore.QCoreApplication.translate("MainWindow", "&Open test file")
            )

        self.ui.action_save_as = QtWidgets.QAction(self.main_window)
        self.ui.action_save_as.setObjectName("action_save_as")
        self.ui.menu_File.addAction(self.ui.action_save_as)
        self.ui.action_save_as.triggered.connect(self.saveAs)
        self.ui.action_save_as.setText(
            QtCore.QCoreApplication.translate("MainWindow", "&Save as...")
        )
        self.ui.action_save_as.setShortcut("Ctrl+Shift+S")

        self.ui.action_save = QtWidgets.QAction(self.main_window)
        self.ui.action_save.setObjectName("action_save")
        self.ui.menu_File.addAction(self.ui.action_save)
        self.ui.action_save.triggered.connect(self.save)
        self.ui.action_save.setText(
            QtCore.QCoreApplication.translate("MainWindow", "&Save...")
        )
        self.ui.action_save.setShortcut("Ctrl+S")

        self.ui.action_load = QtWidgets.QAction(self.main_window)
        self.ui.action_load.setObjectName("action_load")
        self.ui.menu_File.addAction(self.ui.action_load)
        self.ui.action_load.triggered.connect(self.load)
        self.ui.action_load.setText(
            QtCore.QCoreApplication.translate("MainWindow", "&Load...")
        )

        self.ui.action_export_detections = QtWidgets.QAction(self.main_window)
        self.ui.action_export_detections.setObjectName("action_export_detections")
        self.ui.menu_File.addAction(self.ui.action_export_detections)
        self.ui.action_export_detections.triggered.connect(self.exportDetections)
        self.ui.action_export_detections.setText(
            QtCore.QCoreApplication.translate("MainWindow", "&Export detections...")
        )

        self.ui.action_export_tracks = QtWidgets.QAction(self.main_window)
        self.ui.action_export_tracks.setObjectName("action_export_tracks")
        self.ui.menu_File.addAction(self.ui.action_export_tracks)
        self.ui.action_export_tracks.triggered.connect(self.exportTracks)
        self.ui.action_export_tracks.setText(
            QtCore.QCoreApplication.translate("MainWindow", "&Export tracks...")
        )

        self.ui.action_import_detections = QtWidgets.QAction(self.main_window)
        self.ui.action_import_detections.setObjectName("action_import_detections")
        self.ui.menu_File.addAction(self.ui.action_import_detections)
        self.ui.action_import_detections.triggered.connect(self.importDetections)
        self.ui.action_import_detections.setText(
            QtCore.QCoreApplication.translate("MainWindow", "&Import detections...")
        )

        self.ui.action_import_tracks = QtWidgets.QAction(self.main_window)
        self.ui.action_import_tracks.setObjectName("action_import_tracks")
        self.ui.menu_File.addAction(self.ui.action_import_tracks)
        self.ui.action_import_tracks.triggered.connect(self.importTracks)
        self.ui.action_import_tracks.setText(
            QtCore.QCoreApplication.translate("MainWindow", "&Import tracks...")
        )

        self.ui.action_close_file = QtWidgets.QAction(self.main_window)
        self.ui.action_close_file.setObjectName("action_close_file")
        self.ui.menu_File.addAction(self.ui.action_close_file)
        self.ui.action_close_file.setShortcut("Ctrl+Q")
        self.ui.action_close_file.triggered.connect(self.closeFile)
        self.ui.action_close_file.setText(
            QtCore.QCoreApplication.translate("MainWindow", "&Close file")
        )

    def openFile(self):
        try:
            self.playback.openFile()
        except FileNotFoundError as e:
            if e.filename and e.filename != "":
                LogObject().print(e)

    def openTestFile(self):
        try:
            self.playback.openTestFile()
        except FileNotFoundError as e:
            if e.filename and e.filename != "":
                LogObject().print(e)

    def closeFile(self):
        self.playback.closeFile()

    def saveAs(self):
        path = self.playback.selectSaveFile(None, "FishTracker Files (*.fish)")
        if path != "":
            self.save_manager.saveFile(
                path, fh.getConfValue(fh.ConfKeys.save_as_binary)
            )

    def save(self):
        path = None
        if self.save_manager.fast_save_enabled:
            path = self.save_manager.previous_path
            self.save_manager.saveFile(
                path, fh.getConfValue(fh.ConfKeys.save_as_binary)
            )
        else:
            self.saveAs()

    def load(self):
        path = self.playback.selectLoadFile(None, "FishTracker Files (*.fish)")
        if path != "":
            self.save_manager.loadFile(path)

    def exportDetections(self):
        path = self.playback.selectSaveFile()
        if path != "":
            self.detector.saveDetectionsToFile(path)

    def exportTracks(self):
        path = self.playback.selectSaveFile()
        if path != "":
            self.fish_manager.saveToFile(path)

    def importDetections(self):
        path = self.playback.selectLoadFile()
        if path != "":
            self.detector.loadDetectionsFromFile(path)

    def importTracks(self):
        path = self.playback.selectLoadFile()
        if path != "":
            self.fish_manager.loadFromFile(path)

    def runBatch(self):
        dparams = self.detector.parameters.copy()
        tparams = self.tracker.getAllParameters()
        dialog = BatchDialog(self.playback, dparams, tparams)
        dialog.exec_()

    def openUserPreferences(self):
        dialog = UserPreferencesDialog(self.playback)
        dialog.exec_()

    def changeFlowDirection(self):
        self.fish_manager.toggleUpDownInversion()
        if self.playback.isMappingDone():
            self.playback.refreshFrame()
        else:
            self.sonar_viewer.displayImage(None)

    def menuFileAboutToShow(self):
        file_open = self.playback.sonar is not None
        self.ui.action_save_as.setEnabled(file_open)
        self.ui.action_save.setEnabled(file_open)
        self.ui.action_export_detections.setEnabled(file_open)
        self.ui.action_export_tracks.setEnabled(file_open)
        self.ui.action_import_detections.setEnabled(file_open)
        self.ui.action_import_tracks.setEnabled(file_open)
        self.ui.action_close_file.setEnabled(file_open)


def launch_ui():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    fh.checkConfFile()
    playback_manager = PlaybackManager(app, main_window)
    detector = Detector(playback_manager)
    tracker = Tracker(detector)
    fish_manager = FishManager(playback_manager, tracker)
    save_manager = SaveManager(playback_manager, detector, tracker, fish_manager)

    detector.all_computed_signal.connect(playback_manager.refreshFrame)
    tracker.all_computed_signal.connect(lambda x: playback_manager.refreshFrame)

    playback_manager.mapping_done.connect(
        lambda: playback_manager.runInThread(lambda: detector.initMOG(False))
    )

    playback_manager.frame_available_immediate.append(detector.compute_from_event)

    playback_manager.file_opened.connect(detector.clearDetections)
    playback_manager.file_closed.connect(detector.clearDetections)
    playback_manager.file_opened.connect(tracker.clear)
    playback_manager.file_closed.connect(tracker.clear)

    ui_manager = UIManager(
        main_window, playback_manager, detector, tracker, fish_manager, save_manager
    )
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch_ui()
