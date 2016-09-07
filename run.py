import operator
import os
import sys

import h5py
import numpy as np
import scipy.stats as stats
from PyQt4 import QtCore, QtGui

from ui.main_ui import Ui_MainWindow
from util.spikestats import get_spike_times


class MyForm(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.dialog = QtGui.QMainWindow

        self.filename = ''
        self.threshold = 0

        self.message_num = 0

        QtCore.QObject.connect(self.ui.pushButtonAdd, QtCore.SIGNAL("clicked()"), self.add_to_view)
        QtCore.QObject.connect(self.ui.pushButtonReset, QtCore.SIGNAL("clicked()"), self.clear_view)

        QtCore.QObject.connect(self.ui.pushButton_browse, QtCore.SIGNAL("clicked()"), self.browse)
        QtCore.QObject.connect(self.ui.pushButton_auto_threshold, QtCore.SIGNAL("clicked()"), self.auto_threshold)
        QtCore.QObject.connect(self.ui.doubleSpinBox_threshold, QtCore.SIGNAL("valueChanged(const QString&)"), self.update_thresh)

        QtCore.QObject.connect(self.ui.comboBox_test_num, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.load_traces)
        QtCore.QObject.connect(self.ui.comboBox_trace, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.load_channels)

        self.ui.view.threshLine.sigPositionChangeFinished.connect(self.update_thresh2)
        self.ui.preview.threshLine.sigPositionChangeFinished.connect(self.update_thresh2)

        self.ui.preview.removeLegend()

    def browse(self):
        self.ui.comboBox_test_num.clear()

        QtGui.QFileDialog(self)
        self.filename = QtGui.QFileDialog.getOpenFileName()
        self.ui.lineEdit_file_name.setText(self.filename)

        # If the filename is not blank, attempt to extract test numbers and place them into the combobox
        if self.filename != '':
            if '.hdf5' in self.filename:
                try:
                    h_file = h5py.File(unicode(self.filename), 'r')
                except IOError:
                    self.add_message('Error: I/O Error')
                    self.ui.lineEdit_comments.setEnabled(False)
                    self.ui.lineEdit_comments.setText('')
                    self.ui.comboBox_test_num.setEnabled(False)
                    return

                tests = {}
                for key in h_file.keys():
                    if 'segment' in key:
                        for test in h_file[key].keys():
                            tests[test] = int(test.replace('test_', ''))

                sorted_tests = sorted(tests.items(), key=operator.itemgetter(1))

                for test in sorted_tests:
                    self.ui.comboBox_test_num.addItem(test[0])

                self.ui.lineEdit_comments.setEnabled(True)
                self.ui.comboBox_test_num.setEnabled(True)

                h_file.close()

            else:
                self.add_message('Error: Must select a .hdf5 file.')
                self.ui.lineEdit_comments.setEnabled(False)
                self.ui.lineEdit_comments.setText('')
                self.ui.comboBox_test_num.setEnabled(False)
                return
        else:
            self.add_message('Error: Must select a file to open.')
            self.ui.lineEdit_comments.setEnabled(False)
            self.ui.lineEdit_comments.setText('')
            self.ui.comboBox_test_num.setEnabled(False)
            return

    def auto_threshold(self):
        thresh_fraction = 0.7

        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if self.valid_filename(filename):
            h_file = h5py.File(unicode(self.filename), 'r')
            target_test = self.ui.comboBox_test_num.currentText()
        else:
            return

        # Find target segment
        for segment in h_file.keys():
            for test in h_file[segment].keys():
                if target_test == test:
                    target_seg = segment
                    target_test = test

        trace_data = h_file[target_seg][target_test].value

        if len(trace_data.shape) == 4:
            trace_data = trace_data.squeeze()

        # Still shape of 4
        if len(trace_data.shape) == 4:
            trace_data = trace_data[:, :, 1, :]
            trace_data = trace_data.squeeze()

        if len(trace_data.shape) == 2:
            # Compute threshold from just one rep
            average_max = np.array(np.max(np.abs(trace_data[1, :]))).mean()
            thresh = thresh_fraction * average_max
        else:
            # Compute threshold from average maximum of traces
            max_trace = []
            for n in range(len(trace_data[1, :, 0])):
                max_trace.append(np.max(np.abs(trace_data[1, n, :])))
            average_max = np.array(max_trace).mean()
            thresh = thresh_fraction * average_max

        self.ui.doubleSpinBox_threshold.setValue(thresh)
        self.update_thresh()

        h_file.close()

    def add_to_view(self):
        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if self.valid_filename(filename):
            h_file = h5py.File(unicode(self.filename), 'r')
            target_test = self.ui.comboBox_test_num.currentText()
        else:
            return

        # clear view
        # self.clear_view()

        for key in h_file.keys():
            if 'segment' in key:
                for test in h_file[key].keys():
                    if target_test == test:
                        target_seg = key
                        target_test = test

        # Makes the assumption that all of the traces are of the same type
        stim_info = eval(h_file[target_seg][target_test].attrs['stim'])
        self.ui.label_stim_type.setText(stim_info[int(self.ui.comboBox_trace.currentText().replace('trace_', '')) - 1]['components'][0]['stim_type'])

        fs = h_file[target_seg].attrs['samplerate_ad']

        target_trace = []
        target_rep = []
        target_chan = []

        # Get the values from the combo boxes
        self.ui.comboBox_test_num.currentText()
        if self.ui.comboBox_trace.currentText() != '':
            target_trace = int(self.ui.comboBox_trace.currentText().replace('trace_', '')) - 1
        if self.ui.comboBox_channel.currentText() != '':
            target_chan = int(self.ui.comboBox_channel.currentText().replace('channel_', '')) - 1

        test_data = h_file[target_seg][target_test].value

        # Get the presentation data depending on if there is a channel field or not
        if len(test_data.shape) == 4:
            presentation = test_data[target_trace, target_rep, target_chan, :]
        elif len(test_data.shape) == 3:
            presentation = test_data[target_trace, target_rep, :]

        len_presentation = len(presentation)

        # Get the length of the window and length of presentation depending on if all is selected or not
        if len_presentation != 0:
            window = len(presentation) / float(fs)
        else:
            if len(test_data.shape) == 4:
                window = len(test_data[0, 0, 0, :]) / float(fs)
                len_presentation = len(test_data[0, 0, 0, :])
            elif len(test_data.shape) == 3:
                window = len(test_data[0, 0, :]) / float(fs)
                len_presentation = len(test_data[0, 0, :])

        xlist = np.linspace(0, float(window), len_presentation)
        ylist = presentation

        # Set window size
        if len(presentation) > 0:
            ymin = min(presentation)
            ymax = max(presentation)
        else:
            ymin = 0
            ymax = 0
            if len(test_data.shape) == 3:
                rep_len = test_data.shape[1]
                for i in range(rep_len):
                    if min(test_data[target_trace, i, :]) < ymin:
                        ymin = min(test_data[target_trace, i, :])
                    if max(test_data[target_trace, i, :]) > ymax:
                        ymax = max(test_data[target_trace, i, :])
            else:
                rep_len = test_data.shape[1]
                for i in range(rep_len):
                    if min(test_data[target_trace, i, target_chan, :]) < ymin:
                        ymin = min(test_data[target_trace, i, target_chan, :])
                    if max(test_data[target_trace, i, target_chan, :]) > ymax:
                        ymax = max(test_data[target_trace, i, target_chan, :])

        self.ui.view.setXRange(0, window, 0)
        self.ui.view.setYRange(ymin, ymax, 0.1)

        # self.ui.view.tracePlot.clear()
        # Fix xlist to be the length of presentation
        if len(test_data.shape) == 3:
            self.ui.view.addTraceAverage(xlist, test_data[target_trace, :, :],
                                         target_test + ' trace_' + str(target_trace + 1) + ' chan_' + str(
                                             target_chan + 1))
        else:
            self.ui.view.addTraceAverage(xlist, test_data[target_trace, :, target_chan, :],
                                         target_test + ' trace_' + str(target_trace + 1) + ' chan_' + str(
                                             target_chan + 1))

        h_file.close()

    def generate_preview(self):
        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if self.valid_filename(filename):
            h_file = h5py.File(unicode(self.filename), 'r')
            target_test = self.ui.comboBox_test_num.currentText()
        else:
            return

        # clear view
        self.clear_preview()

        for key in h_file.keys():
            if 'segment' in key:
                for test in h_file[key].keys():
                    if target_test == test:
                        target_seg = key
                        target_test = test

        # Makes the assumption that all of the traces are of the same type
        stim_info = eval(h_file[target_seg][target_test].attrs['stim'])
        self.ui.label_stim_type.setText(stim_info[int(self.ui.comboBox_trace.currentText().replace('trace_', '')) - 1]['components'][0]['stim_type'])

        fs = h_file[target_seg].attrs['samplerate_ad']

        target_trace = []
        target_rep = []
        target_chan = []

        # Get the values from the combo boxes
        self.ui.comboBox_test_num.currentText()
        if self.ui.comboBox_trace.currentText() != '':
            target_trace = int(self.ui.comboBox_trace.currentText().replace('trace_', '')) - 1
        if self.ui.comboBox_channel.currentText() != '':
            target_chan = int(self.ui.comboBox_channel.currentText().replace('channel_', '')) - 1

        test_data = h_file[target_seg][target_test].value

        # Get the presentation data depending on if there is a channel field or not
        if len(test_data.shape) == 4:
            presentation = test_data[target_trace, target_rep, target_chan, :]
        elif len(test_data.shape) == 3:
            presentation = test_data[target_trace, target_rep, :]

        len_presentation = len(presentation)

        # Get the length of the window and length of presentation depending on if all is selected or not
        if len_presentation != 0:
            window = len(presentation) / float(fs)
        else:
            if len(test_data.shape) == 4:
                window = len(test_data[0, 0, 0, :]) / float(fs)
                len_presentation = len(test_data[0, 0, 0, :])
            elif len(test_data.shape) == 3:
                window = len(test_data[0, 0, :]) / float(fs)
                len_presentation = len(test_data[0, 0, :])

        xlist = np.linspace(0, float(window), len_presentation)
        ylist = presentation

        # Set window size
        if len(presentation) > 0:
            ymin = min(presentation)
            ymax = max(presentation)
        else:
            ymin = 0
            ymax = 0
            if len(test_data.shape) == 3:
                rep_len = test_data.shape[1]
                for i in range(rep_len):
                    if min(test_data[target_trace, i, :]) < ymin:
                        ymin = min(test_data[target_trace, i, :])
                    if max(test_data[target_trace, i, :]) > ymax:
                        ymax = max(test_data[target_trace, i, :])
            else:
                rep_len = test_data.shape[1]
                for i in range(rep_len):
                    if min(test_data[target_trace, i, target_chan, :]) < ymin:
                        ymin = min(test_data[target_trace, i, target_chan, :])
                    if max(test_data[target_trace, i, target_chan, :]) > ymax:
                        ymax = max(test_data[target_trace, i, target_chan, :])

        self.ui.preview.setXRange(0, window, 0)
        self.ui.preview.setYRange(ymin, ymax, 0.1)

        # self.ui.preview.tracePlot.clear()
        # Fix xlist to be the length of presentation
        if len(test_data.shape) == 3:
            self.ui.preview.addTraceAverage(xlist, test_data[target_trace, :, :], target_test + ' trace_' + str(target_trace+1) + ' chan_' + str(target_chan+1))
        else:
            self.ui.preview.addTraceAverage(xlist, test_data[target_trace, :, target_chan, :], target_test + ' trace_' + str(target_trace) + ' chan_' + str(target_chan))

        h_file.close()

    def valid_filename(self, filename):
        # Validate filename
        if filename != '':
            if '.hdf5' in filename:
                try:
                    temp_file = h5py.File(unicode(self.filename), 'r')
                    temp_file.close()
                except IOError:
                    self.add_message('Error: I/O Error')
                    return False
            else:
                self.add_message('Error: Must select a .hdf5 file.')
                return False
        else:
            self.add_message('Error: Must select a file to open.')
            return False

        return True

    def load_traces(self):
        self.ui.comboBox_trace.clear()

        # Validate filename
        if self.valid_filename(self.filename):
            h_file = h5py.File(unicode(self.filename), 'r')
            target_test = self.ui.comboBox_test_num.currentText()
        else:
            self.ui.comboBox_trace.setEnabled(False)
            return

        if self.ui.comboBox_test_num.currentText() == 'All' or self.ui.comboBox_test_num.currentText() == '':
            self.ui.comboBox_trace.setEnabled(False)
            self.ui.comboBox_trace.clear()
            h_file.close()
            return
        else:
            self.ui.comboBox_trace.setEnabled(True)

        for key in h_file.keys():
            if 'segment' in key:
                for test in h_file[key].keys():
                    if target_test == test:
                        target_seg = key
                        target_test = test

        traces = h_file[target_seg][target_test].value.shape[0]

        for i in range(traces):
            self.ui.comboBox_trace.addItem('trace_' + str(i+1))

        self.ui.comboBox_trace.setEnabled(True)

        comment = h_file[target_seg].attrs['comment']
        self.ui.lineEdit_comments.setText(comment)

        h_file.close()

    def load_channels(self):
        self.ui.comboBox_channel.clear()

        # Validate filename
        if self.valid_filename(self.filename):
            h_file = h5py.File(unicode(self.filename), 'r')
            target_test = self.ui.comboBox_test_num.currentText()
        else:
            self.ui.comboBox_trace.setEnabled(False)
            return

        if self.ui.comboBox_trace.currentText() == '' or self.ui.comboBox_channel.count() < 2:
            self.ui.comboBox_channel.setEnabled(False)
            self.ui.comboBox_channel.clear()
        else:
            self.ui.comboBox_channel.setEnabled(True)

        if self.ui.comboBox_test_num.count() == 0:
            h_file.close()
            return

        for key in h_file.keys():
            if 'segment' in key:
                for test in h_file[key].keys():
                    if target_test == test:
                        target_seg = key
                        target_test = test

        if len(h_file[target_seg][target_test].value.shape) > 3:
            channels = h_file[target_seg][target_test].value.shape[2]
        else:
            channels = 1

        if channels == 1:
            self.ui.comboBox_channel.addItem('channel_1')
        else:
            for i in range(channels):
                self.ui.comboBox_channel.addItem('channel_' + str(i+1))

        if self.ui.comboBox_trace.currentText() != '' and self.ui.comboBox_channel != '':
            self.generate_preview()

        h_file.close()

    def update_thresh(self):
        self.ui.view.setThreshold(self.ui.doubleSpinBox_threshold.value())
        self.ui.view.update_thresh()
        self.ui.preview.setThreshold(self.ui.doubleSpinBox_threshold.value())
        self.ui.preview.update_thresh()

    def update_thresh2(self):
        self.ui.doubleSpinBox_threshold.setValue(self.ui.view.getThreshold())
        self.ui.doubleSpinBox_threshold.setValue(self.ui.preview.getThreshold())

    def count_spikes(self):
        pass


    def GetFreqsAttns(self, freqTuningHisto):  # Frequency Tuning Curve method
        """ Helper method for ShowSTH() to organize the frequencies in ascending order separated for each intensity.
        :param freqTuningHisto: dict of pandas.DataFrames with spike data
        :type freqTuningHisto: str
        :returns: ordered list of frequencies (DataFrame keys())
                  numpy array of frequencies
                  numpy array of intensities
        """
        freqs = np.array([])
        attns = np.array([])
        for histoKey in list(freqTuningHisto):
            if histoKey != 'None_None':
                freq = histoKey.split('_')[0]
                freqs = np.hstack([freqs, float(freq) / 1000])
                attn = histoKey.split('_')[1]
                attns = np.hstack([attns, float(attn)])
        attnCount = stats.itemfreq(attns)
        freqs = np.unique(freqs)
        attns = np.unique(attns)
        if np.max(attnCount[:, 1]) != np.min(attnCount[:, 1]):
            abortedAttnIdx = np.where(attnCount[:, 1] != np.max(attnCount[:, 1]))
            attns = np.delete(attns, abortedAttnIdx)
        orderedKeys = []
        for attn in attns:
            freqList = []
            for freq in freqs:
                key = str(int(freq * 1000)) + '_' + str(int(attn))
                freqList.append(key)
            orderedKeys.append(freqList)
        return orderedKeys, freqs, attns

    def add_message(self, message):
        self.message_num += 1
        self.ui.textEdit.append('[' + str(self.message_num) + ']: ' + message + '\n')

    def clear_view(self):
        self.ui.view.clearTraces()
        self.ui.view.clearMouse()
        self.ui.view.clearFocus()
        self.ui.view.clearMask()
        self.ui.view.clearData(axeskey='response')
        self.ui.view.tracePlot.clear()
        self.ui.view.rasterPlot.clear()
        self.ui.view.stimPlot.clear()
        self.ui.view.trace_stash = []

    def clear_preview(self):
        self.ui.preview.clearTraces()
        self.ui.preview.clearMouse()
        self.ui.preview.clearFocus()
        self.ui.preview.clearMask()
        self.ui.preview.clearData(axeskey='response')
        self.ui.preview.tracePlot.clear()
        self.ui.preview.rasterPlot.clear()
        self.ui.preview.stimPlot.clear()
        self.ui.preview.trace_stash = []


def check_output_folders():
    if not os.path.exists('output'):
        os.makedirs('output')

    # if not os.path.exists('output' + os.sep + 'rasters'):
    #     os.makedirs('output' + os.sep + 'rasters')
    #
    # if not os.path.exists('output' + os.sep + 'histograms'):
    #     os.makedirs('output' + os.sep + 'histograms')

    if not os.path.exists('output' + os.sep + 'tuning_curves'):
        os.makedirs('output' + os.sep + 'tuning_curves')


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myApp = MyForm()
    myApp.setWindowIcon(QtGui.QIcon('images/horsey.ico'))
    myApp.show()
    sys.exit(app.exec_())
