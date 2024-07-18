# k4_assignment
Q: Can Artificial Intelligence (AI) play games (like HTML5 Games similar to this - 
https://k4.games/)? If yes, how can you use concepts of computer vision to prove this 
and tool you need to use. 
Yes 
We can use Reinforcement learning (where agent gets rewarded by the environment the 
more closely it acts like the actual model) PPO/ TRPO policies can be used. RL is 
particularly effective for games because it can explore various strategies and learn the 
optimal way to play.  
Examples: Go and Chess 
Supervised and Unsupervised learning and its algorithms like SVMs ,decision trees or 
PCA , GANs can also be used depending on the game structure . Like for recognizing an 
object or predicting the next move we may use supervised learning and for games the 
involves clustering or grouping we may use unsupervised learning . 
For using Computer vision(CV): CV is used to interpret game visuals and provide input 
to the AI framework.  
HTML5 Game Frameworks: Libraries like Phaser and Three.js can be integrated with 
machine learning models to create intelligent game agents. 
Gym: An open-source toolkit for developing and comparing reinforcement learning 
algorithms. 
OpenCV: For image processing and feature extraction 
TensorFlow and PyTorch: Popular machine learning libraries that can be used to build 
and train AI models for game playing. 
Steps include: 
Capture Game Frames: Use screen capture tools to grab frames of the game.(if mss 
module is used) 
def capture_frame(monitor): 
with mss.mss() as sct: 
frame = np.array(sct.grab(monitor)) 
frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) 
return frame 
Preprocess Frames: Convert frames to grayscale, resize, or normalize them to a 
suitable format for the AI model. 
def preprocess_frame(frame): 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    resized_frame = cv2.resize(gray_frame, (84, 84)) 
    normalized_frame = resized_frame / 255.0 
    return np.expand_dims(normalized_frame, axis=2) 
 
 Feature Extraction: Use techniques like edge detection, object detection, and 
segmentation to extract meaningful features from the game frames. 
def extract_features(frame): 
    edges = cv2.Canny(frame, threshold1=100, threshold2=200) 
    return edges 
 
State Representation: Represent the game state based on extracted features, which 
the AI will use to make decisions. 
def get_game_state(frame): 
    preprocessed_frame = preprocess_frame(frame) 
    features = extract_features(preprocessed_frame) 
    return np.expand_dims(features, axis=0) 
 
Model Training: Train a reinforcement learning model, such as DQN (Deep Q-Network) 
or PPO (Proximal Policy Optimization), using the game states and actions. 
def build_dqn(input_shape, action_space): 
    model = models.Sequential() 
    model.add(layers.Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=input_shape)) 
    model.add(layers.Conv2D(64, (4, 4), strides=2, activation='relu')) 
    model.add(layers.Conv2D(64, (3, 3), strides=1, activation='relu')) 
    model.add(layers.Flatten()) 
    model.add(layers.Dense(512, activation='relu')) 
//we may use a different activation function and no. of cnn layers may also be different for different problem 
statements// 
    model.add(layers.Dense(action_space, activation='linear')) 
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00025), loss='mse') 
    return model 
 
def train_dqn(model, game_state, action, reward, next_game_state, done, gamma=0.99): 
    target = reward 
    if not done: 
        target = reward + gamma * np.amax(model.predict(next_game_state)[0]) 
    target_f = model.predict(game_state) 
    target_f[0][action] = target 
    model.fit(game_state, target_f, epochs=1, verbose=0) 
 
 
 
Q: Is AI animation is possible? If yes, what kind of AI/ML tools can be used for making 
videos (like https://www.youtube.com/watch?v=ajKIsf4ncu0 ). Also, let us know how 
can we develop some basic tools for the same. 
Yes 
 1.Generative Adversarial Networks (GANs): GANimation( To alter emotions generally in 
the animation) and DeepMotion (motion capture)  
2.Reinforcement Learning (RL):OpenAI's Dactyl (animated characters) and DeepMind's 
Control Suite is also used  
3.Neural Networks: RNNs (for sequential movement)and CNNs(image and video 
generation) 
4.Computer Vision and Graphics Tools: 
Blender/ Unity3D- AI platforms for animation 
5.Speech: DeepSpeech 
tensorflow keras opencv numpy are used for the framework  
Example for an animation for walk  
def generate_walk_sequence(): 
    steps = 100 
    sequence = np.zeros((1, steps, 2)) 
    for i in range(steps): 
        sequence[0, i, 0] = np.sin(i / 10.0)  # X position 
        sequence[0, i, 1] = np.cos(i / 10.0)  # Y position 
    return sequence 
