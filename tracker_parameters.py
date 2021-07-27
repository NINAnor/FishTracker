﻿from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from tracker import Tracker
from detector_parameters import LabeledSlider

import json
import os.path
import numpy as np

PARAMETERS_PATH = "tracker_parameters.json"

class TrackerParametersView(QWidget):
    def __init__(self, playback_manager, tracker, detector, fish_manager=None):
        super().__init__()
        self.playback_manager = playback_manager
        self.tracker = tracker
        self.detector = detector
        self.fish_manager = fish_manager

        self.initUI()

        self.playback_manager.polars_loaded.connect(self.setButtonsEnabled)
        self.detector.state_changed_event.append(self.setButtonsEnabled)
        self.tracker.state_changed_signal.connect(self.setButtonsEnabled)
        self.tracker.state_changed_signal.connect(self.setButtonTexts)
        self.setButtonsEnabled()

    def setButtonsEnabled(self):
        detector_active = self.detector.bg_subtractor.initializing or self.detector.computing
        all_value = self.tracker.parametersDirty() and self.playback_manager.isPolarsDone() and (not detector_active or self.tracker.tracking)
        self.primary_track_btn.setEnabled(all_value)

    def setButtonTexts(self):
        if self.tracker.tracking:
            self.primary_track_btn.setText("Cancel")
        else:
            self.primary_track_btn.setText("Track All")

    def primaryTrack(self):
        """
        Either starts the first tracking iteration, or cancels the process
        if one is already started.
        """

        if not self.tracker.tracking:
            if self.fish_manager:
                self.fish_manager.clear_old_data = True
            self.playback_manager.runInThread(self.tracker.primaryTrack)
        else:
            if self.detector.bg_subtractor.initializing:
                self.detector.bg_subtractor.stop_initializing = True
            elif self.detector.computing:
                self.detector.stop_computing = True
            else:
                self.tracker.stop_tracking = True

    def secondaryTrack(self):
        """
        Either starts the second (or nth) tracking iteration, or cancels the process
        if one is already started. A FishManager is required for this action.
        """

        if self.fish_manager is None:
            return

        if not self.tracker.tracking:
            self.fish_manager.clear_old_data = False
            used_dets = self.fish_manager.applyFiltersAndGetUsedDetections()
            self.playback_manager.runInThread(lambda: self.tracker.secondaryTrack(used_dets, self.tracker.parameters))
        else:
            if self.detector.bg_subtractor.initializing:
                self.detector.bg_subtractor.stop_initializing = True
            elif self.detector.computing:
                self.detector.stop_computing = True
            else:
                self.tracker.stop_tracking = True

    def initUI(self):
        self.vertical_layout = QVBoxLayout()
        self.vertical_layout.setObjectName("verticalLayout")
        self.vertical_layout.setSpacing(5)
        self.vertical_layout.setContentsMargins(7,7,7,7)

        self.main_label = QLabel(self)
        self.main_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.main_label.setText("Tracker options")
        self.vertical_layout.addWidget(self.main_label)

        self.form_layout = QFormLayout()
        self.max_age_slider = LabeledSlider("Max age", self.form_layout, [self.tracker.setMaxAge], self.tracker.parameters.max_age, 1, 100, self)
        self.min_hits_slider = LabeledSlider("Min hits", self.form_layout, [self.tracker.setMinHits], self.tracker.parameters.min_hits, 1, 10, self)
        self.search_radius_slider = LabeledSlider("Search radius", self.form_layout, [self.tracker.setSearchRadius], self.tracker.parameters.search_radius, 1, 100, self)

        self.vertical_spacer1 = QSpacerItem(0, 5, QSizePolicy.Minimum, QSizePolicy.Maximum)
        self.form_layout.addItem(self.vertical_spacer1)

        self.trim_tails_checkbox = QCheckBox("", self)
        self.trim_tails_checkbox.stateChanged.connect(self.tracker.setTrimTails)
        self.form_layout.addRow("Trim tails", self.trim_tails_checkbox)

        self.vertical_layout.addLayout(self.form_layout)

        self.vertical_spacer2 = QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Maximum)
        self.vertical_layout.addItem(self.vertical_spacer2)

        self.primary_track_btn = QPushButton()
        self.primary_track_btn.setObjectName("primaryTrackButton")
        self.primary_track_btn.setText("Primary Track")
        self.primary_track_btn.setToolTip("Start a process that detects fish and tracks them in all the frames")
        self.primary_track_btn.clicked.connect(self.primaryTrack)
        self.primary_track_btn.setMinimumWidth(150)

        self.secondary_track_btn = QPushButton()
        self.secondary_track_btn.setObjectName("secondaryTrackButton")
        self.secondary_track_btn.setText("Secondary Track")
        self.secondary_track_btn.setToolTip("Start a process that detects fish and tracks them in all the frames")
        self.secondary_track_btn.clicked.connect(self.secondaryTrack)
        self.secondary_track_btn.setMinimumWidth(150)

        self.track_button_layout = QHBoxLayout()
        self.track_button_layout.setObjectName("buttonLayout")
        self.track_button_layout.setSpacing(7)
        self.track_button_layout.setContentsMargins(0,0,0,0)

        self.track_button_layout.addStretch()
        self.track_button_layout.addWidget(self.primary_track_btn)
        self.track_button_layout.addWidget(self.secondary_track_btn)

        self.vertical_layout.addLayout(self.track_button_layout)

        self.vertical_layout.addStretch()

        self.save_btn = QPushButton()
        self.save_btn.setObjectName("saveButton")
        self.save_btn.setText("Save")
        self.save_btn.setToolTip("Save tracker parameters")
        self.save_btn.clicked.connect(self.saveJSON)

        self.load_btn = QPushButton()
        self.load_btn.setObjectName("loadButton")
        self.load_btn.setText("Load")
        self.load_btn.setToolTip("Load tracker parameters")
        self.load_btn.clicked.connect(self.loadJSON)

        self.reset_btn = QPushButton()
        self.reset_btn.setObjectName("resetButton")
        self.reset_btn.setText("Reset")
        self.reset_btn.setToolTip("Reset tracker parameters")
        self.reset_btn.clicked.connect(self.tracker.resetParameters)
        self.reset_btn.clicked.connect(self.refreshValues)

        self.button_layout = QHBoxLayout()
        self.button_layout.setObjectName("buttonLayout")
        self.button_layout.setSpacing(7)
        self.button_layout.setContentsMargins(0,0,0,0)

        self.button_layout.addWidget(self.save_btn)
        self.button_layout.addWidget(self.load_btn)
        self.button_layout.addWidget(self.reset_btn)
        self.button_layout.addStretch()

        self.vertical_layout.addLayout(self.button_layout)

        self.setLayout(self.vertical_layout)
        self.refreshValues()

    def saveJSON(self):
        dict = self.tracker.getParameterDict()
        if dict is None:
            return

        try:
            with open(PARAMETERS_PATH, "w") as f:
                json.dump(dict, f, indent=3)
        except FileNotFoundError as e:
            print(e)

    def loadJSON(self):
        try:
            with open(PARAMETERS_PATH, "r") as f:
                dict = json.load(f)
        except FileNotFoundError as e:
            print("Error: Tracker parameters file not found:", e)
            return
        except json.JSONDecodeError as e:
            print("Error: Invalid tracker parameters file:", e)
            return

        self.tracker.parameters.setParameterDict(dict)
        self.refreshValues()

    def refreshValues(self):
        params = self.tracker.parameters

        self.max_age_slider.setValue(params.max_age)
        self.min_hits_slider.setValue(params.min_hits)
        self.search_radius_slider.setValue(params.search_radius)
        self.trim_tails_checkbox.setChecked(params.trim_tails)

if __name__ == "__main__":
    import sys
    from playback_manager import PlaybackManager
    from detector import Detector
    
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    playback_manager = PlaybackManager(app, main_window)

    detector = Detector(playback_manager)

    tracker = Tracker(detector)
    tracker_params = TrackerParametersView(playback_manager, tracker, detector)

    main_window.setCentralWidget(tracker_params)
    main_window.show()
    sys.exit(app.exec_())
