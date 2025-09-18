import numpy as np

# Create a NumPy array
numbers = np.array([5.5, 10.5, 15.5, 20.5])

# Compute statistics
mean = np.mean(numbers)
min_val = np.min(numbers)
max_val = np.max(numbers)
std_val = np.std(numbers)

# print result
print(f"Numbers: {numbers}")
print(f"Mean: {mean}")
print(f"Min: {min_val}")
print(f"Max: {max_val}")
print(f"standard_deviation: {std_val}")