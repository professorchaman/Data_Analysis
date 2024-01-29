import numpy as np
from scipy.signal import find_peaks

def find_and_plot_peaks(ax, x_val, y_val, savgol_filt_data, do_peak_finding='y', w_size=10, p_prom=2, thresh=2):
    """
    Find and plot peaks in the data if do_peak_finding is 'y'.

    Parameters:
    - ax: Matplotlib Axes object where the peaks will be plotted.
    - x_val: Array of x values.
    - y_val: Array of y values.
    - savgol_filt_data: Array of Savitzky-Golay filtered data.
    - do_peak_finding: Whether to perform peak finding ('y' or 'n').
    - w_size: Width parameter for peak finding.
    - p_prom: Prominence parameter for peak finding.
    - thresh: Threshold for peak detection.

    Returns:
    - None
    """

    peak_indices = find_peaks(y_val, width=w_size, prominence=p_prom)[0]
    peak_x = x_val[peak_indices]
    peak_y = savgol_filt_data[peak_indices]

    if all(peak_y > thresh): ax.scatter(peak_x, peak_y, marker='D', color='k')
