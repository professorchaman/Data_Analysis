import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Function to parse the .peaks file into a pandas DataFrame
def parse_peaks_file(filename):
    data = pd.DataFrame(columns=['PeakType', 'Center', "Center error", 'Height', "Height error", 'HWHM', "HWHM error", "Area"])

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            if line.startswith('%'):
                # print(line)
                parts = line.strip().split('\t')
                area = float(parts[3])
                peak_type = parts[0].strip().split(" ")[-1]
                parameters = [float(part) for part in parts[-1].split()[::3]]
                print(parameters)
                std_errors = [float(part) for part in parts[-1].split()[2::3]]
                data = data.append({'PeakType': peak_type, 'Center': parameters[1],
                                'Height': parameters[0], 'HWHM': parameters[2], "Center error": std_errors[1], "Height error": std_errors[0], "HWHM error" : std_errors[2], "Area": area}, ignore_index=True)

    return data

# Add functions for gaussian and lorentzian peaks that can take the parameters as input
def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(-np.log(2) * ((x - mean) / standard_deviation) ** 2)

def lorentzian(x, mean, amplitude, gamma):
    return amplitude / (1 + ((x - mean) / gamma) ** 2)
2
# Function to plot the curves based on the peak type and the parameters
def plot_parameters(data):
    x = np.linspace(500, 900, 2000)
    y_total = np.zeros_like(x)
    for i, row in data.iterrows():
        if row['PeakType'] == 'Gaussian':
            y = gaussian(x, row['Center'], row['Height'], row['HWHM'])
            y_total += y
        elif row['PeakType'] == 'Lorentzian':
            y = lorentzian(x, row['Center'], row['Height'], row['HWHM'])
            y_total += y
        plt.plot(x, y, label=f"{row['PeakType']} {i + 1}")
    plt.plot(x, y_total, label='Total', linestyle='--')

    plt.legend()
    plt.show()

# Example usage
filename = r'../../Projects/NW-IMPACT/High_Pressure/Experiment_16/Final data/NV/Peak_data.peaks'
data = parse_peaks_file(filename)
plot_parameters(data)