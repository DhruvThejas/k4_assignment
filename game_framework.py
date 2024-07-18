import mss
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Set TensorFlow logging to error only to reduce clutter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Function to capture game frame
def capture_frame(monitor):
    with mss.mss() as sct:
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

# Function to preprocess frame
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (84, 84))
    normalized_frame = resized_frame / 255.0
    return normalized_frame

# Function to extract features
def extract_features(frame):
    frame_8bit = (frame * 255).astype(np.uint8)
    edges = cv2.Canny(frame_8bit, threshold1=100, threshold2=200)
    return edges

# Function to get game state
def get_game_state(frame):
    preprocessed_frame = preprocess_frame(frame)
    features = extract_features(preprocessed_frame)
    return np.expand_dims(features, axis=(0, -1))

# Function to build DQN model
def build_dqn(input_shape, action_space):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (8, 8), strides=4, activation='relu'))
    model.add(layers.Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(action_space, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025), loss='mse')
    return model

# Function to train DQN model
def train_dqn(model, game_state, action, reward, next_game_state, done, gamma=0.99):
    target = reward
    if not done:
        target = reward + gamma * np.amax(model.predict(next_game_state)[0])
    target_f = model.predict(game_state)
    target_f[0][action] = target
    model.fit(game_state, target_f, epochs=1, verbose=0)

# Main function to run the pipeline
def main():
    monitor = mss.mss().monitors[1]
    input_shape = (84, 84, 1)
    action_space = 4
    dqn_model = build_dqn(input_shape, action_space)

    # Example loop to capture frames and train the model
    for episode in range(100):  # Replace with your game loop condition
        frame = capture_frame(monitor)
        game_state = get_game_state(frame)

        # Dummy values for action, reward, next_game_state, and done
        action = np.random.randint(0, action_space)
        reward = 1  # Replace with your reward calculation
        next_frame = capture_frame(monitor)
        next_game_state = get_game_state(next_frame)
        done = False  # Replace with your done condition

        try:
            train_dqn(dqn_model, game_state, action, reward, next_game_state, done)
        except Exception as e:
            print(f"Error in training: {e}")

        if done:
            break

if __name__ == "__main__":
    main()
