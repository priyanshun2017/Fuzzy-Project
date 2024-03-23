import numpy as np
import matplotlib.pyplot as plt

# Define the ranges for each input variable
temperature_range = (1939, 40000)
luminosity_range = (0.000085, 340000)
radius_range = (0.0084, 1783)
absolute_magnitude_range = (-11.92, 20.06)


# Define membership functions for each input variable
def temperature_membership(temperature):
    # Define triangular membership functions
    low = np.maximum(0, (3000 - temperature) / (3000 - temperature_range[0]))
    medium = np.maximum(0, 1 - np.abs(temperature - 15000) / 6000)
    high = np.maximum(0, (temperature - 15000) / (temperature_range[1] - 15000))
    return low, medium, high


def luminosity_membership(luminosity):
    # Define trapezoidal membership functions
    low = np.maximum(0, (1000 - luminosity) / (1000 - luminosity_range[0]))
    medium = np.maximum(0, 1 - np.abs(luminosity - 50000) / 25000)
    high = np.maximum(0, (luminosity - 50000) / (luminosity_range[1] - 50000))
    return low, medium, high


# Define similar functions for radius and absolute magnitude

# Example usage
temperature_value = 10000
luminosity_value = 100000
radius_value = 100
absolute_magnitude_value = 10

temperature_low, temperature_medium, temperature_high = temperature_membership(temperature_value)
luminosity_low, luminosity_medium, luminosity_high = luminosity_membership(luminosity_value)

# Plot membership functions
temperature_values = np.linspace(temperature_range[0], temperature_range[1], 1000)
luminosity_values = np.linspace(luminosity_range[0], luminosity_range[1], 1000)

plt.figure(figsize=(10, 5))
plt.plot(temperature_values, temperature_membership(temperature_values)[0], label='Low')
plt.plot(temperature_values, temperature_membership(temperature_values)[1], label='Medium')
plt.plot(temperature_values, temperature_membership(temperature_values)[2], label='High')
plt.xlabel('Temperature (K)')
plt.ylabel('Membership Value')
plt.title('Temperature Membership Functions')
plt.legend()
plt.show()
