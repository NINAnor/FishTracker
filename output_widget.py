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
from queue import Queue

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import file_handler as fh
from log_object import LogObject

LOG_PATH = fh.getFilePathInAppData("log.txt")
log_lock = QReadWriteLock()


class WriteStream:
    """
    The new Stream Object which replaces the default stream associated with sys.stdout
    This object puts the received data in a queue
    """

    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        if text not in ["", " ", "\n"]:
            self.queue.put(text)

    def flush(self):
        pass


class StreamReceiver(QObject):
    """
    A QObject (to be run in a QThread) which sits waiting for data to come through a queue.
    It blocks until data is available, and once it gets something from the queue, it sends
    it to the "MainThread" by emitting a Qt Signal
    """

    signal = pyqtSignal(str)

    def __init__(self, queue, *args, **kwargs):
        QObject.__init__(self, *args, **kwargs)
        self.queue = queue

    @pyqtSlot()
    def run(self):
        while True:
            text = self.queue.get()
            self.signal.emit(text)


class OutputViewer(QWidget):
    updateLogSignal = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.text_edit)
        self.layout.addWidget(self.clear_button)
        self.queue = None

        self.verbosity = fh.getConfValue(fh.ConfKeys.log_verbosity)
        self.time_stamp = fh.getConfValue(fh.ConfKeys.log_timestamp)
        self.latestLine = ""

    def connectToLogObject(self, format=None):
        """
        Connect text_edit field to LogObject signal. A formatting function, which takes a string as input
        and returns the modified string, can be provided for custom formatting.

        Note: The conf file has an option to include time stamp to the displayed log.
        """
        log = LogObject()
        if format:
            log.connect(lambda s, v, ts: self.appendText(format(s), v, ts))
        else:
            log.connect(self.appendText)
        log.disconnectDefault()

    def addTimeStamp(self, str, time_stamp):
        """
        Adds a time stamp to the provided string.
        """
        return f"{str} [{time_stamp}]"

    def redirectStdOut(self):
        """
        Redirects standard output stream (from e.g. print()) to StreamReceiver
        and connects text_edit to the receiver.
        """
        # Create Queue and redirect sys.stdout to this queue
        self.queue = Queue()
        sys.stdout = WriteStream(self.queue)

        # Create thread that will listen on the other end of the queue, and send the text to the textedit in our application
        self.thread = QThread()
        self.receiver = StreamReceiver(self.queue)
        self.receiver.signal.connect(self.appendText)
        self.receiver.moveToThread(self.thread)
        self.thread.started.connect(self.receiver.run)
        self.thread.start()

    @pyqtSlot(str, int, str)
    def appendText(self, text, verbosity, time_stamp):
        if self.verbosity >= verbosity:
            self.text_edit.moveCursor(QTextCursor.End)
            if self.time_stamp:
                self.text_edit.insertPlainText(f"{text} [{time_stamp}]\n")
            else:
                self.text_edit.insertPlainText(f"{text}\n")

            lines = self.text_edit.toPlainText().splitlines(False)
            if len(lines) > 0:
                self.updateLogSignal.emit(lines[-1])

    def clear(self):
        self.text_edit.clear()


class LogToFile:
    def __init__(self, verbosity=2, clear=True):
        self.verbosity = verbosity
        if clear:
            self.write("", 2, mode="w")

    def write(self, string, verbosity, mode="a"):
        if self.verbosity >= verbosity:
            locker = QWriteLocker(log_lock)
            with open(LOG_PATH, mode) as file:
                file.write(f"{string}")

    def writeLine(self, line, verbosity, time_stamp, mode="a"):
        if self.verbosity >= verbosity:
            locker = QWriteLocker(log_lock)
            with open(LOG_PATH, mode) as file:
                file.write(f"{line} [{time_stamp}]\n")


if __name__ == "__main__":
    # An example QObject (to be run in a QThread) which outputs information with print
    class LongRunningThing(QObject):
        @pyqtSlot()
        def run(self):
            for i in range(1000):
                print(i)

    # An Example application QWidget containing the textedit to redirect stdout to
    class MyApp(QWidget):
        def __init__(self, *args, **kwargs):
            QWidget.__init__(self, *args, **kwargs)

            self.thread = None
            self.layout = QVBoxLayout(self)
            self.output_viewer = OutputViewer()
            self.output_viewer.redirectStdOut()
            self.button = QPushButton("start long running thread")
            self.button.clicked.connect(self.start_thread)
            self.layout.addWidget(self.output_viewer)
            self.layout.addWidget(self.button)

        @pyqtSlot()
        def start_thread(self):
            if self.thread:
                self.thread.quit()
                self.thread.wait()

            self.thread = QThread()
            self.long_running_thing = LongRunningThing()
            self.long_running_thing.moveToThread(self.thread)
            self.thread.started.connect(self.long_running_thing.run)
            self.thread.start()

    # Create QApplication and QWidget
    qapp = QApplication(sys.argv)
    app = MyApp()
    app.show()
    sys.exit(qapp.exec_())
