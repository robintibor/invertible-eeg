import numpy as np
import matplotlib.pyplot as plt
import seaborn

tuh_names = ['FP2', 'FP1', 'F4', 'F3', 'C4', 'C3', 'P4', 'P3', 'O2', 'O1', 'F8',
             'F7', 'T4', 'T3', 'T6', 'T5', 'A2', 'A1', 'FZ', 'CZ', 'PZ']
hgd_names = ['Fp2', 'Fp1', 'F4', 'F3', 'C4', 'C3', 'P4', 'P3', 'O2', 'O1', 'F8',
'F7', 'T8', 'T7', 'P8', 'P7','M2', 'M1', 'Fz', 'Cz', 'Pz']


#https://www.google.com/imgres?imgurl=https%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fthumb%2F7%2F70%2F21_electrodes_of_International_10-20_system_for_EEG.svg%2F1200px-21_electrodes_of_International_10-20_system_for_EEG.svg.png&imgrefurl=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2F10%25E2%2580%259320_system_(EEG)&tbnid=eF8gqawUiKx9TM&vet=12ahUKEwiY8ODK3bvqAhUGtaQKHWTuDhkQMygIegUIARC0AQ..i&docid=_l2v_nz1_IHQ1M&w=1200&h=1073&q=FP1%20FP2%20Cz%20EEG&client=firefox-b-d&ved=2ahUKEwiY8ODK3bvqAhUGtaQKHWTuDhkQMygIegUIARC0AQ
tight_tuh_positions = [
    ['', '', 'FP1', '', 'FP2', '', ''],
    ['', 'F7', 'F3',  'FZ',  'F4', 'F8','',],
    ['A1', 'T3', 'C3', 'CZ', 'C4', 'T4', 'A2'],
    ['', 'T5', 'P3', 'Pz', 'P4', 'T6', ''],
    ['', '', 'O1', '', 'O2', '', '']]

tight_hgd_21_positions = [
    ['', '', 'FP1', '', 'FP2', '', ''],
    ['', 'F7', 'F3',  'FZ',  'F4', 'F8','',],
    ['M1', 'T7', 'C3', 'CZ', 'C4', 'T8', 'M2'],
    ['', 'P7', 'P3', 'Pz', 'P4', 'P8', ''],
    ['', '', 'O1', '', 'O2', '', '']]


def get_sensor_pos(sensor_name, sensor_map=tight_tuh_positions):
    sensor_pos = np.where(
        np.char.lower(np.char.array(sensor_map)) == sensor_name.lower())
    # unpack them: they are 1-dimensional arrays before
    assert len(sensor_pos[0]) == 1, (
        "there should be a position for the sensor "
        "{:s}".format(sensor_name))
    return sensor_pos[0][0], sensor_pos[1][0]


def plot_head_signals_tight(signals, sensor_names=None, figsize=(12, 7),
                            plot_args=None, hspace=0.35,
                            sensor_map=tight_tuh_positions,
                            tsplot=False, sharex=True, sharey=True):
    assert sensor_names is None or len(signals) == len(sensor_names), ("need "
                                                                       "sensor names for all sensor matrices")
    assert sensor_names is not None
    if plot_args is None:
        plot_args = dict()
    figure = plt.figure(figsize=figsize)
    sensor_positions = [get_sensor_pos(name, sensor_map) for name in
                        sensor_names]
    sensor_positions = np.array(sensor_positions)  # sensors x 2(row and col)
    maxima = np.max(sensor_positions, axis=0)
    minima = np.min(sensor_positions, axis=0)
    max_row = maxima[0]
    max_col = maxima[1]
    min_row = minima[0]
    min_col = minima[1]
    rows = max_row - min_row + 1
    cols = max_col - min_col + 1
    first_ax = None
    for i in range(0, len(signals)):
        sensor_name = sensor_names[i]
        sensor_pos = sensor_positions[i]
        assert np.all(sensor_pos == get_sensor_pos(sensor_name, sensor_map))
        # Transform to flat sensor pos
        row = sensor_pos[0]
        col = sensor_pos[1]
        subplot_ind = (
                                  row - min_row) * cols + col - min_col + 1  # +1 as matlab uses based indexing
        if first_ax is None:
            ax = figure.add_subplot(rows, cols, subplot_ind)
            first_ax = ax
        elif sharex is True and sharey is True:
            ax = figure.add_subplot(rows, cols, subplot_ind, sharey=first_ax,
                                    sharex=first_ax)
        elif sharex is True and sharey is False:
            ax = figure.add_subplot(rows, cols, subplot_ind,
                                    sharex=first_ax)
        elif sharex is False and sharey is True:
            ax = figure.add_subplot(rows, cols, subplot_ind, sharey=first_ax)
        else:
            ax = figure.add_subplot(rows, cols, subplot_ind)

        signal = signals[i]
        if tsplot is False:
            ax.plot(signal, **plot_args)
        else:
            seaborn.tsplot(signal.T, ax=ax, **plot_args)
        ax.set_title(sensor_name)
        ax.set_yticks([])
        if len(signal) == 600:
            ax.set_xticks([150, 300, 450])
            ax.set_xticklabels([])
        else:
            ax.set_xticks([])

        ax.xaxis.grid(True)
        # make line at zero
        ax.axhline(y=0, ls=':', color="grey")
        figure.subplots_adjust(hspace=hspace)
    return figure
