import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import bpy
import csv

# Generate a simple walking motion sequence
def generate_walk_sequence():
    steps = 100
    sequence = np.zeros((1, steps, 2))
    for i in range(steps):
        sequence[0, i, 0] = np.sin(i / 10.0)  # X position
        sequence[0, i, 1] = np.cos(i / 10.0)  # Y position
    return sequence

# Create a LSTM model for generating sequences
model = Sequential()
model.add(LSTM(50, input_shape=(None, 2), return_sequences=True))
model.add(Dense(2, activation='linear'))
model.compile(optimizer='adam', loss='mse')

# Generate and fit the walking sequence
walk_sequence = generate_walk_sequence()
model.fit(walk_sequence, walk_sequence, epochs=500)

# Save the generated sequence to a CSV file
def save_to_csv(sequence, filename):
    sequence = sequence[0]  # Remove the batch dimension
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(sequence)

save_to_csv(walk_sequence, 'walk_sequence.csv')

# Load CSV data and apply animation to a Blender object
def apply_animation(object_name, animation_data):
    obj = bpy.data.objects[object_name]
    for frame, (x, y) in enumerate(animation_data):
        obj.location = (x, y, 0)
        obj.keyframe_insert(data_path="location", frame=frame)

# Load animation data from CSV
animation_data = []
with open('walk_sequence.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        animation_data.append(list(map(float, row)))

# Apply animation to the Blender object
apply_animation('Cube', animation_data)  # Replace 'Cube' with your object name

print("Animation applied successfully to Blender object.")
