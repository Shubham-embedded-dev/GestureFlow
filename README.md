# GestureFlow

This is an AI-powered Gesture Control System that allows users to interact with their computer without using a mouse or keyboard.  The system uses Computer Vision and MediaPipe to track hand movements in real-time and convert them into commands.


# It has two main modes:
  
  1. Command Mode: Where specific finger gestures act as shortcuts to launch applications .
  2. Mouse Mode: Which uses smooth tracking for cursor control and includes a 'hold-to-click' feature for hands-free navigation.

   I’ve also implemented a Privacy Mode and Fist-based commands for tasks like system control and navigation. It’s designed to make human-       computer interaction more intuitive and accessible."
  

 # WORKING : TECHNICAL SIDE 
 
  Under the hood, the project is built using Python and MediaPipe's Hand Landmarker API. The core logic involves processing the webcam feed     to detect hand landmarks. I’ve implemented custom logic to handle finger counting and gesture recognition, including an Exponential Moving    Average (EMA) filter for smooth cursor movement and stability buffers to eliminate gesture flickering. It maps physical hand movements into   digital commands using the pyautogui library.
  

# Key Features:

   1. Dynamic Gesture Recognition: Seamlessly distinguishes between complex hand signs to trigger system-level tasks and application shortcuts.
   2. Smooth Cursor Control: Implements an Exponential Moving Average (EMA) algorithm to ensure fluid, jitter-free mouse movement.
   3. Fist-Based Priority Actions: A dedicated 'Kill-Switch' mechanism using fist gestures for instant back-navigation and system privacy/lock mode.
   4. Intuitive UX: Real-time visual feedback and on-screen status overlays, making the system highly interactive and user-friendly.


# Technical Highlights:

   1. Stability Optimization: Incorporates frame-skipping and stability buffers to eliminate gesture flickering and conflict between similar gestures .
   2. State-Machine Logic: Uses a robust state-based architecture to handle multi-mode operations (Command vs. Mouse) without command overlap.
