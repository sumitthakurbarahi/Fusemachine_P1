import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters for the looped curve
radius = np.random.uniform(40, 50)  # Random radius between 0.5 and 1.5
num_points = 20
theta = np.linspace(0, 2*np.pi, num_points)

# Random factor for the sin function
factor = np.random.uniform(1.1, 1.5)  # Random factor between 1 and 3

# Parametric equations for the looped curve
x = radius * np.cos(theta)
y = radius * np.sin(factor * theta)



# Create a DataFrame with the x and y coordinates
df = pd.DataFrame({'x': x, 'y': y})

# Write the DataFrame to a CSV file
df.to_csv('curve.csv', index=False)

# Read the CSV file
df = pd.read_csv('curve.csv')

# Plot the looped curve
plt.figure(figsize=(6, 6))
plt.plot(df['x'], df['y'], 'b-')
plt.axis('equal')
plt.title('Looped Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
