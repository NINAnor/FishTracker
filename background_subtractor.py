"""
This file is part of Fish Tracker.
Copyright 2021, VTT Technical research centre of Finland Ltd.
Developed by: Otto Korkalo and Mikael Uimonen.

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

from math import floor

import cv2
from PyQt5 import QtCore

from log_object import LogObject
from mog_parameters import MOGParameters


class BackgroundSubtractor(QtCore.QObject):
    """
    Implements background subtraction for Detector / SonarView and Echogram.
    """

    # When background subtractor parameters changes.
    parameters_changed_signal = QtCore.pyqtSignal()

    # When background subtractor state changes.
    state_changed_signal = QtCore.pyqtSignal()

    def __init__(self, image_provider):
        super().__init__()

        self.image_provider = image_provider
        self.image_height = 0
        self.image_width = 0

        self.fgbg_mog = None

        # [trigger] Terminate initializing process.
        self.stop_initializing = False

        # [flag] Whether MOG is initializing
        self.initializing = False

        # [flag] Whether MOG has been initialized
        self.mog_ready = False

        self.mog_parameters = None
        self.applied_mog_parameters = None
        self.resetParameters()

    def setParameter(self, key, value):
        if self.mog_parameters is not None:
            self.mog_parameters.setKeyValuePair(key, value)
        else:
            LogObject().print2(
                f"MOG Parameters not found. Cannot set key '{key}' to value '{value}'."
            )

    def setParameters(self, parameters: MOGParameters):
        if self.mog_parameters is not None:
            self.mog_parameters.values_changed_signal.disconnect(
                self.parameters_changed_signal
            )

        self.mog_parameters = parameters
        self.mog_parameters.values_changed_signal.connect(
            self.parameters_changed_signal
        )
        self.parameters_changed_signal.emit()

    def resetParameters(self):
        self.setParameters(MOGParameters())

    def initMOG(self):
        if hasattr(self.image_provider, "pausePolarLoading"):
            self.image_provider.pausePolarLoading(True)

        self.mog_ready = False
        self.initializing = True
        self.stop_initializing = False
        self.compute_on_event = True
        self.state_changed_signal.emit()

        self.fgbg_mog = cv2.createBackgroundSubtractorMOG2()
        self.fgbg_mog.setNMixtures(self.mog_parameters.data.mixture_count)
        self.fgbg_mog.setVarThreshold(self.mog_parameters.data.mog_var_thresh)
        self.fgbg_mog.setShadowValue(0)

        nof_frames = self.image_provider.getFrameCount()
        nof_bg_frames = min(nof_frames, self.mog_parameters.data.nof_bg_frames)

        # Create background model from fixed number of frames.
        # Count step based on number of frames
        step = nof_frames / nof_bg_frames

        for i in range(nof_bg_frames):
            ind = floor(i * step)

            if self.stop_initializing:
                LogObject().print2("Stopped initializing (BG subtraction) at", ind)
                self.stop_initializing = False
                self.mog_ready = False
                self.initializing = False
                self.applied_mog_parameters = None
                self.state_changed_signal.emit()
                return

            image_o = self.image_provider.getFrame(ind)
            self.fgbg_mog.apply(
                image_o, learningRate=self.mog_parameters.data.learning_rate
            )

        self.image_height = image_o.shape[0]
        try:
            self.image_width = image_o.shape[1]
        except IndexError:
            self.image_width = 1

        self.mog_ready = True
        self.initializing = False
        self.applied_mog_parameters = self.mog_parameters.copy()

        self.state_changed_signal.emit()

        if hasattr(self.image_provider, "pausePolarLoading"):
            self.image_provider.pausePolarLoading(False)

        if hasattr(self.image_provider, "refreshFrame"):
            self.image_provider.refreshFrame()

    def subtractBG(self, image):
        # Get foreground mask, without updating the  model (learningRate = 0)
        try:
            fg_mask_mog = self.fgbg_mog.apply(image, learningRate=0)
            return fg_mask_mog

        except AttributeError as e:
            LogObject().print2("BG subtractor not initialized", e)
            return None

    def subtractBGFiltered(self, image, median_size):
        fg_mask_mog = self.fgbg_mog.apply(image, learningRate=0)
        fg_mask_filt = cv2.medianBlur(fg_mask_mog, median_size)
        return fg_mask_filt

    def applyParameters(self):
        self.applied_mog_parameters = self.mog_parameters.copy()

    def parametersDirty(self):
        return self.mog_parameters != self.applied_mog_parameters

    def abortComputing(self):
        self.applied_mog_parameters = None
