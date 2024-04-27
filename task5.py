#this is now a test file
import numpy as np
import matplotlib.pyplot as plt

# Create some data for the plot
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Create a figure
fig, ax = plt.subplots()

# Display the initial frame
im = ax.imshow(Z, cmap='viridis', interpolation='nearest', animated=True)

# Function to update the image for each frame
def update(frame):
    new_Z = np.sin(X + 0.1 * frame) * np.cos(Y + 0.1 * frame)  # Modify the data slightly
    im.set_array(new_Z)
    return [im]

# Create the animation
ani = plt.FuncAnimation(fig, update, frames=100, blit=True)

# Show the animation
plt.show()
