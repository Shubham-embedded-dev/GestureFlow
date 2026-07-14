import cv2
import mediapipe as mp
import pyautogui
import time
import os
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional
from dataclasses import dataclass

# --- PATH HANDLING FOR EXE ---
def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Configuration
MODEL_PATH: str = get_resource_path('hand_landmarker.task')
SCREEN_WIDTH: int = 1920
SCREEN_HEIGHT: int = 1080
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
SMOOTHING_FACTOR: float = 4.0
FIST_HOLD_THRESHOLD: float = 1.0
FIST_LOCK_THRESHOLD: float = 2.5
MODE_TOGGLE_DELAY: float = 1.5
CLICK_HOLD_THRESHOLD: float = 1.2
CLICK_COOLDOWN: float = 2.0

pyautogui.FAILSAFE = False

@dataclass
class Point:
    x: float
    y: float

@dataclass
class HandState:
    finger_count: int
    is_fist: bool
    index_tip: Point

class HandGestureController:
    def __init__(self, model_path: str):
        self.detector = self._create_hand_detector(model_path)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open webcam.")
        self.mode: str = "COMMAND"
        self.fist_start_time: Optional[float] = None
        self.click_start_time: Optional[float] = None
        self.last_action_time: float = 0
        self.prev_mouse_x: float = 0
        self.prev_mouse_y: float = 0

    def _create_hand_detector(self, model_path: str) -> vision.HandLandmarker:
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1
        )
        return vision.HandLandmarker.create_from_options(options)

    def _is_finger_extended(self, landmarks, tip_idx: int, pip_idx: int) -> bool:
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    def _is_thumb_extended(self, landmarks) -> bool:
        return abs(landmarks[4].x - landmarks[5].x) > 0.08

    def _is_thumb_tucked(self, landmarks) -> bool:
        return abs(landmarks[4].x - landmarks[5].x) < 0.05

    def _are_finger_tips_below_bases(self, landmarks) -> bool:
        return all(landmarks[i].y > landmarks[i - 2].y for i in [8, 12, 16, 20])

    def get_hand_state(self, landmarks) -> HandState:
        finger_count = sum([self._is_finger_extended(landmarks, i, i-2) for i in [8, 12, 16, 20]])
        if self._is_thumb_extended(landmarks): finger_count += 1
        is_fist = self._are_finger_tips_below_bases(landmarks) and self._is_thumb_tucked(landmarks)
        return HandState(finger_count=finger_count, is_fist=is_fist, index_tip=Point(landmarks[8].x, landmarks[8].y))

    def _draw_landmarks(self, frame: cv2.Mat, landmarks) -> None:
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * FRAME_WIDTH), int(lm.y * FRAME_HEIGHT)), 6, (0, 255, 0), -1)

    def _handle_fist_action(self, frame: cv2.Mat, is_fist: bool) -> None:
        current_time = time.time()
        if is_fist:
            if self.fist_start_time is None: self.fist_start_time = current_time
            elapsed = current_time - self.fist_start_time
            if elapsed > FIST_LOCK_THRESHOLD:
                self.mode = "LOCKED"
        else:
            self.fist_start_time = None

    def _handle_mouse_mode(self, index_tip: Point) -> None:
        current_time = time.time()
        target_x, target_y = int(index_tip.x * SCREEN_WIDTH), int(index_tip.y * SCREEN_HEIGHT)
        self.prev_mouse_x += (target_x - self.prev_mouse_x) / SMOOTHING_FACTOR
        self.prev_mouse_y += (target_y - self.prev_mouse_y) / SMOOTHING_FACTOR
        pyautogui.moveTo(self.prev_mouse_x, self.prev_mouse_y)
        if self.click_start_time is None: self.click_start_time = current_time
        if current_time - self.click_start_time > CLICK_HOLD_THRESHOLD:
            pyautogui.click()
            self.click_start_time = current_time + CLICK_COOLDOWN

    def _handle_command_mode(self, finger_count: int) -> None:
        current_time = time.time()
        if 1 <= finger_count <= 4 and (current_time - self.last_action_time) > MODE_TOGGLE_DELAY:
            pyautogui.hotkey('win', str(finger_count))
            self.last_action_time = current_time

    def _process_hand_gesture(self, frame: cv2.Mat, landmarks) -> None:
        hand_state = self.get_hand_state(landmarks)
        self._draw_landmarks(frame, landmarks)
        
        self._handle_fist_action(frame, hand_state.is_fist)
        
        # UNLOCK LOGIC
        if hand_state.finger_count == 5 and self.mode == "LOCKED":
            self.mode = "COMMAND"

        if self.mode == "LOCKED":
            cv2.putText(frame, "PRIVACY LOCKED!", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            return

        if not hand_state.is_fist:
            if hand_state.finger_count == 5 and (time.time() - self.last_action_time) > MODE_TOGGLE_DELAY:
                self.mode = "MOUSE" if self.mode == "COMMAND" else "COMMAND"
                self.last_action_time = time.time()
            if self.mode == "MOUSE": self._handle_mouse_mode(hand_state.index_tip)
            else: self._handle_command_mode(hand_state.finger_count)

    def run(self) -> None:
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # Draw status
            color = (0, 0, 255) if self.mode == "LOCKED" else (0, 255, 0)
            cv2.putText(frame, f"MODE: {self.mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame), int(time.time() * 1000))
            if results.hand_landmarks: self._process_hand_gesture(frame, results.hand_landmarks[0])
            
            cv2.imshow("Gesture Controller", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'): break
        self.cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    HandGestureController(MODEL_PATH).run()