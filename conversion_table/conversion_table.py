from helpers.helpers import nm2ram, ram2nm, nm2eV, eV2nm

# Define a dictionary to map do_conversion values to corresponding actions
conversion_table = {
    'raman': {
        'x_val': lambda x_data, laser_source: nm2ram(x_data, laser_source),
        'x_label': 'Raman shift(cm$^{-1}$)',
        'y_val': lambda data, x_data: data * x_data / 10**7
    },
    'energy': {
        'x_val': lambda x_data, _: 1239.84193 / x_data,
        'x_label': 'Energy(eV)',
        'y_val': lambda data, x_data: (1239.84193 / (x_data**2)) * data
    },
    'wavelength': {
        'x_val': lambda x_data, _: x_data,
        'x_label': 'Wavelength (nm)',
        'y_val': lambda data, x_data: data  # No change to y_val
    }
}
