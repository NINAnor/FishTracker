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

import json

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from detector import Detector
from detector_parameters import DetectorParameters
from file_handler import checkAppDataPath, getFilePathInAppData
from log_object import LogObject
from mog_parameters import MOGParameters

PARAMETERS_PATH = getFilePathInAppData("detector_parameters.json")
parameters_lock = QReadWriteLock()


class LabeledSlider:
    def __init__(
        self,
        label,
        form_layout,
        connected_functions=[],
        default_value=0,
        min_value=0,
        max_value=1,
        parent=None,
        mapping=None,
        reverse_mapping=None,
        formatting="{}",
    ):
        self.mapping = mapping
        self.reverse_mapping = reverse_mapping
        self.formatting = formatting

        self.layout = QHBoxLayout()
        self.layout.setObjectName("layout")
        self.layout.setSpacing(5)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.value = QLabel("1.0", parent)
        self.value.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        self.value.setMinimumWidth(50)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_value)
        self.slider.setMaximum(max_value)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setValue(default_value)

        self.connected_functions = connected_functions
        self.slider.valueChanged.connect(self.valueChanged)
        if self.mapping is not None:
            self.value.setText(
                self.formatting.format(self.mapping(self.slider.value()))
            )
        else:
            self.value.setText(self.formatting.format(self.slider.value()))

        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.value)

        form_layout.addRow(label, self.layout)

    def getValue(self):
        return self.slider.value()

    def setValue(self, value):
        self.slider.blockSignals(True)

        self.value.setText(self.formatting.format(value))
        if self.reverse_mapping:
            self.slider.setValue(int(self.reverse_mapping(value)))
        else:
            self.slider.setValue(int(value))

        self.slider.blockSignals(False)

    def valueChanged(self, value):
        if self.mapping:
            applied_value = self.mapping(value)
        else:
            applied_value = value

        self.value.setText(self.formatting.format(applied_value))

        for f in self.connected_functions:
            f(applied_value)


class FloatValidator(QDoubleValidator):
    def __init__(self, bottom, top, decimals, parent=None):
        super().__init__(bottom, top, decimals, parent)

    def validate(self, s, pos):
        if len(s) == 0 or s == "-":
            return (QValidator.Intermediate, s, pos)

        decimal_point = "."

        try:
            chars_after_point = len(s) - s.index(decimal_point) - 1
        except ValueError:
            return (QValidator.Invalid, s, pos)

        if chars_after_point > self.decimals():
            return (QValidator.Invalid, s, pos)

        try:
            d = float(s)
        except ValueError:
            return (QValidator.Invalid, s, pos)

        if self.bottom() <= d <= self.top():
            return (QValidator.Acceptable, s, pos)
        elif chars_after_point == 0:
            return (QValidator.Intermediate, s, pos)
        else:
            return (QValidator.Invalid, s, pos)


class DetectorParametersView(QWidget):
    def __init__(self, playback_manager, detector, sonar_processor):
        super().__init__()
        self.playback_manager = playback_manager
        self.detector = detector
        self.detector.parameters_changed_signal.connect(self.refreshValues)
        self.bg_subtractor = detector.bg_subtractor
        self.bg_subtractor.parameters_changed_signal.connect(self.refreshValues)
        self.sonar_processor = sonar_processor

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setContentsMargins(7, 7, 7, 7)

        def addLine(label, initial_value, validator, connected, layout):
            line = QLineEdit()
            line.setAlignment(Qt.AlignRight)
            line.setValidator(validator)
            line.setText(str(initial_value))
            for f in connected:
                line.textChanged.connect(f)
            layout.addRow(label, line)
            return line

        refresh_lambda = lambda x: playback_manager.refreshFrame()

        self.init_label = QLabel(self)
        self.init_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.init_label.setText("Initialization")
        self.verticalLayout.addWidget(self.init_label)

        self.form_layout2 = QFormLayout()

        # BG subtraction parameters
        bg_sub = detector.bg_subtractor
        bg_sub_data = bg_sub.mog_parameters.data
        lambda_mog = lambda x: bg_sub.setParameter(
            MOGParameters.ParametersEnum.mog_var_thresh, x
        )
        self.mog_var_threshold_line = addLine(
            "MOG var threshold",
            bg_sub_data.mog_var_thresh,
            QIntValidator(0, 200),
            [lambda_mog, refresh_lambda],
            self.form_layout2,
        )

        lambda_bg_frames = lambda x: bg_sub.setParameter(
            MOGParameters.ParametersEnum.nof_bg_frames, x
        )
        self.nof_bg_frames_line = addLine(
            "Background frames",
            bg_sub_data.nof_bg_frames,
            QIntValidator(10, 10000),
            [lambda_bg_frames, refresh_lambda],
            self.form_layout2,
        )

        lambda_learning_rate = lambda x: bg_sub.setParameter(
            MOGParameters.ParametersEnum.learning_rate, x
        )
        lr_validator = FloatValidator(bottom=0.0, top=1.0, decimals=3)
        lr_validator.setNotation(QDoubleValidator.StandardNotation)
        self.learning_rate_line = addLine(
            "Learning rate",
            bg_sub_data.learning_rate,
            lr_validator,
            [lambda_learning_rate, refresh_lambda],
            self.form_layout2,
        )

        self.verticalLayout.addLayout(self.form_layout2)

        self.verticalSpacer1 = QSpacerItem(
            0, 10, QSizePolicy.Minimum, QSizePolicy.Maximum
        )
        self.verticalLayout.addItem(self.verticalSpacer1)

        self.recalculate_mog_btn = QPushButton()
        self.recalculate_mog_btn.setObjectName("recalculateMOGButton")
        self.recalculate_mog_btn.setText("Apply Values")
        self.recalculate_mog_btn.setToolTip(
            "Initialize the detector with given parameters"
        )
        self.recalculate_mog_btn.clicked.connect(self.recalculateMOG)
        self.recalculate_mog_btn.setMinimumWidth(150)

        self.init_button_layout = QHBoxLayout()
        self.init_button_layout.setObjectName("buttonLayout")
        self.init_button_layout.setSpacing(7)
        self.init_button_layout.setContentsMargins(0, 0, 0, 0)

        self.init_button_layout.addStretch()
        self.init_button_layout.addWidget(self.recalculate_mog_btn)

        self.verticalLayout.addLayout(self.init_button_layout)

        self.verticalSpacer2 = QSpacerItem(
            0, 40, QSizePolicy.Minimum, QSizePolicy.Maximum
        )
        self.verticalLayout.addItem(self.verticalSpacer2)

        self.image_controls_label = QLabel(self)
        self.image_controls_label.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Minimum
        )
        self.image_controls_label.setText("Detector options")
        self.verticalLayout.addWidget(self.image_controls_label)

        self.form_layout = QFormLayout()

        # Detector parameters
        det_param_data = detector.parameters.data
        lambda_detection_size = lambda x: detector.setParameter(
            DetectorParameters.ParametersEnum.detection_size, x
        )
        self.detection_size_line = addLine(
            "Detection size",
            det_param_data.detection_size,
            QIntValidator(0, 200),
            [lambda_detection_size, refresh_lambda],
            self.form_layout,
        )

        lambda_min_fg_pixels = lambda x: detector.setParameter(
            DetectorParameters.ParametersEnum.min_fg_pixels, x
        )
        self.min_fg_pixels_line = addLine(
            "Min foreground pixels",
            det_param_data.min_fg_pixels,
            QIntValidator(0, 200),
            [lambda_min_fg_pixels, refresh_lambda],
            self.form_layout,
        )

        lambda_median_size = lambda x: detector.setParameter(
            DetectorParameters.ParametersEnum.median_size, x
        )
        self.median_size_slider = LabeledSlider(
            "Median size",
            self.form_layout,
            [lambda_median_size, refresh_lambda],
            0,
            0,
            3,
            self,
            lambda x: 2 * x + 3,
            lambda x: (x - 3) / 2,
        )
        self.median_size_slider.setValue(det_param_data.median_size)

        lambda_dbscan_eps = lambda x: detector.setParameter(
            DetectorParameters.ParametersEnum.dbscan_eps, x
        )
        self.dbscan_eps_line = addLine(
            "Clustering epsilon",
            det_param_data.dbscan_eps,
            QIntValidator(0, 200),
            [lambda_dbscan_eps, refresh_lambda],
            self.form_layout,
        )

        lambda_dbscan_min_samples = lambda x: detector.setParameter(
            DetectorParameters.ParametersEnum.dbscan_min_samples, x
        )
        self.dbscan_min_samples_line = addLine(
            "Clustering min samples",
            det_param_data.dbscan_min_samples,
            QIntValidator(0, 200),
            [lambda_dbscan_min_samples, refresh_lambda],
            self.form_layout,
        )

        self.verticalLayout.addLayout(self.form_layout)

        self.verticalSpacer3 = QSpacerItem(
            0, 10, QSizePolicy.Minimum, QSizePolicy.Maximum
        )
        self.verticalLayout.addItem(self.verticalSpacer3)

        self.calculate_all_btn = QPushButton()
        self.calculate_all_btn.setObjectName("calculateAllButton")
        self.calculate_all_btn.setText("Calculate All")
        self.calculate_all_btn.setToolTip(
            "Start a process that initializes the detector and "
            "detects fish in all the frames"
        )
        self.calculate_all_btn.clicked.connect(self.calculateAll)
        self.calculate_all_btn.setMinimumWidth(150)

        self.calc_button_layout = QHBoxLayout()
        self.calc_button_layout.setObjectName("buttonLayout")
        self.calc_button_layout.setSpacing(7)
        self.calc_button_layout.setContentsMargins(0, 0, 0, 0)

        self.calc_button_layout.addStretch()
        self.calc_button_layout.addWidget(self.calculate_all_btn)

        self.verticalLayout.addLayout(self.calc_button_layout)

        self.verticalLayout.addStretch()

        self.save_btn = QPushButton()
        self.save_btn.setObjectName("saveButton")
        self.save_btn.setText("Save")
        self.save_btn.setToolTip("Save detector parameters")
        self.save_btn.clicked.connect(self.saveJSON)

        self.load_btn = QPushButton()
        self.load_btn.setObjectName("loadButton")
        self.load_btn.setText("Load")
        self.load_btn.setToolTip("Load detector parameters")
        self.load_btn.clicked.connect(self.loadJSON)

        self.reset_btn = QPushButton()
        self.reset_btn.setObjectName("resetButton")
        self.reset_btn.setToolTip("Reset detector parameters")
        self.reset_btn.setText("Reset")
        self.reset_btn.clicked.connect(self.detector.resetParameters)

        self.button_layout = QHBoxLayout()
        self.button_layout.setObjectName("buttonLayout")
        self.button_layout.setSpacing(7)
        self.button_layout.setContentsMargins(0, 0, 0, 0)

        self.button_layout.addWidget(self.save_btn)
        self.button_layout.addWidget(self.load_btn)
        self.button_layout.addWidget(self.reset_btn)
        self.button_layout.addStretch()

        self.verticalLayout.addLayout(self.button_layout)

        self.setLayout(self.verticalLayout)

        # self.loadJSON()

        self.playback_manager.polars_loaded.connect(self.setButtonsEnabled)
        self.detector.state_changed_signal.connect(self.setButtonsEnabled)
        self.detector.parameters_changed_signal.connect(self.setButtonsEnabled)
        self.detector.state_changed_signal.connect(self.setButtonTexts)
        self.setButtonsEnabled()

    def saveJSON(self):
        param_dict = self.detector.getParameterDict()

        if param_dict is None:
            return

        try:
            checkAppDataPath()
            locker = QWriteLocker(parameters_lock)
            with open(PARAMETERS_PATH, "w") as f:
                json.dump(param_dict, f, indent=3)
        except FileNotFoundError as e:
            LogObject().print(e)

    def loadJSON(self):
        try:
            locker = QReadLocker(parameters_lock)
            with open(PARAMETERS_PATH) as f:
                dict = json.load(f)
        except FileNotFoundError as e:
            LogObject().print("Error: Detector parameters file not found:", e)
            return
        except json.JSONDecodeError as e:
            LogObject().print("Error: Invalid detector parameters file:", e)
            return

        self.detector.setParameterDict(dict)

    def refreshValues(self):
        det_data = self.detector.parameters.data
        mog_data = self.bg_subtractor.mog_parameters.data

        self.detection_size_line.setText(str(det_data.detection_size))
        self.min_fg_pixels_line.setText(str(det_data.min_fg_pixels))
        self.median_size_slider.setValue(det_data.median_size)
        self.dbscan_eps_line.setText(str(det_data.dbscan_eps))
        self.dbscan_min_samples_line.setText(str(det_data.dbscan_min_samples))

        self.nof_bg_frames_line.setText(str(mog_data.nof_bg_frames))
        self.learning_rate_line.setText(str(mog_data.learning_rate))
        self.mog_var_threshold_line.setText(str(mog_data.mog_var_thresh))

    def recalculateMOG(self):
        if not self.bg_subtractor.initializing:
            self.playback_manager.runInThread(self.detector.initMOG)
        else:
            self.bg_subtractor.stop_initializing = True

    def calculateAll(self):
        if not self.detector.computing:
            self.playback_manager.runInThread(self.detector.computeAll)
        else:
            self.detector.stop_computing = True

    def setButtonTexts(self):
        if self.detector.computing:
            self.calculate_all_btn.setText("Cancel")
        else:
            self.calculate_all_btn.setText("Calculate All")

        if self.bg_subtractor.initializing:
            self.recalculate_mog_btn.setText("Cancel")
        else:
            self.recalculate_mog_btn.setText("Apply Values")

    def setButtonsEnabled(self):
        mog_value = (
            self.playback_manager.isMappingDone()
            and self.bg_subtractor.parametersDirty()
        )
        self.recalculate_mog_btn.setEnabled(mog_value)

        all_value = (
            self.playback_manager.isPolarsDone()
            and self.detector.allCalculationAvailable()
        )
        self.calculate_all_btn.setEnabled(all_value)


if __name__ == "__main__":
    import sys

    from image_manipulation import ImageProcessor
    from playback_manager import PlaybackManager

    app = QApplication(sys.argv)
    main_window = QMainWindow()
    playback_manager = PlaybackManager(app, main_window)
    # playback_manager.openTestFile()

    sonar_processor = ImageProcessor()

    detector = Detector(playback_manager)
    detector_parameters = DetectorParametersView(
        playback_manager, detector, sonar_processor
    )

    main_window.setCentralWidget(detector_parameters)
    main_window.show()
    sys.exit(app.exec_())
