import numpy as np


def luminosity_membership(luminosity):
    # Triangular membership function for luminosity
    low = np.maximum(0, 1 - abs((luminosity - 0.001) / 0.0005))
    medium = np.maximum(0, 1 - abs((luminosity - 0.01) / 0.005))
    high = np.maximum(0, (luminosity - 0.01) / 0.005)
    return low, medium, high


def radius_membership(radius):
    # Triangular membership function for radius
    small = np.maximum(0, 1 - abs((radius - 0.1) / 0.05))
    medium = np.maximum(0, 1 - abs((radius - 0.5) / 0.2))
    large = np.maximum(0, (radius - 0.5) / 0.2)
    return small, medium, large


def magnitude_membership(magnitude):
    # Triangular membership function for absolute magnitude
    low = np.maximum(0, 1 - abs((magnitude - 10) / 5))
    medium = np.maximum(0, 1 - abs((magnitude - 15) / 5))
    high = np.maximum(0, (magnitude - 15) / 5)
    return low, medium, high


def temperature_membership(temperature):
    # Triangular membership function for temperature
    low = np.maximum(0, 1 - abs((temperature - 3000) / 1000))
    medium = np.maximum(0, 1 - abs((temperature - 7000) / 2000))
    high = np.maximum(0, (temperature - 7000) / 2000)
    return low, medium, high


def fuzzy_rules(temperature, luminosity, radius, magnitude):
    # Fuzzy rules (simple rule: if temperature is high OR luminosity is high, then star type is bright)
    temperature_low, temperature_medium, temperature_high = temperature_membership(temperature)
    luminosity_low, luminosity_medium, luminosity_high = luminosity_membership(luminosity)
    bright = np.maximum(temperature_high, luminosity_high)
    return bright


def fuzzy_inference(rules):
    return rules


def defuzzification(output):
    # Ensure output is an array
    if not isinstance(output, np.ndarray):
        output = np.array([output])
    # Simple centroid defuzzification method
    sum_output = np.sum(output)
    if np.isnan(sum_output) or sum_output == 0:
        return np.nan
    else:
        return np.sum(output * np.array([0, 1, 2])) / sum_output



class FuzzySystem:
    def __init__(self):
        pass


def main():
    fuzzy_system = FuzzySystem()
    # Example input data
    temperature = 3068
    luminosity = 0.001
    radius = 0.17
    magnitude = 16.12
    # Fuzzy rules
    rules = fuzzy_rules(temperature, luminosity, radius, magnitude)
    # Fuzzy inference
    output = fuzzy_inference(rules)
    # Defuzzification
    crisp_output = defuzzification(output)
    print("Predicted Star type:", crisp_output)


if __name__ == "__main__":
    main()
