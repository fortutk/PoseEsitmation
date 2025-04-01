import cv2
import mediapipe as mp
import torch
import numpy as np
import os
from BaseClassifier import PhaseLSTM

class PoseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1 if self.upBody else 2,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
    
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(
                img, 
                self.results.pose_landmarks, 
                self.mpPose.POSE_CONNECTIONS,
                self.mpDraw.DrawingSpec(color=(0,255,0)),  # Custom colors
                self.mpDraw.DrawingSpec(color=(255,0,0))
            )
        return img
    
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks and self.results.pose_world_landmarks:
            for id, (lm, world_lm) in enumerate(zip(
                self.results.pose_landmarks.landmark,
                self.results.pose_world_landmarks.landmark
            )):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy, lm.x, lm.y, lm.z, world_lm.x, world_lm.y, world_lm.z])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return lmList

class SquatPhaseDetector:
    def __init__(self, model_path, seq_length=30):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PhaseLSTM(input_size=7).to(self.device)
        self.seq_length = seq_length
        self.sequence = []
        self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            self.model.load_state_dict(torch.load(model_path, weights_only=True))  # Security fix
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()
    
    def extract_features(self, landmarks):
        """Simplified feature extraction using key joints"""
        key_indices = [11, 12, 23, 24, 25, 26, 0]  # Shoulders, hips, knees, nose
        return np.array([
            landmarks[id][3:6] for id in key_indices  # Normalized x,y,z
        ]).flatten()[:7]  # Ensure 7 features
    
    def predict_phase(self, landmarks):
        if not landmarks:
            return None, None
            
        features = self.extract_features(landmarks)
        self.sequence.append(features)
        
        if len(self.sequence) > self.seq_length:
            self.sequence = self.sequence[-self.seq_length:]
            
        if len(self.sequence) == self.seq_length:
            with torch.no_grad():
                prob = self.model(
                    torch.tensor([self.sequence], dtype=torch.float32).to(self.device)
                ).item()
            return "UP" if prob > 0.5 else "DOWN", prob
        return None, None

def process_video(video_path, model_path):
    # Verify paths
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    detector = PoseDetector()
    phase_detector = SquatPhaseDetector(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    print(f"Processing: {video_path}")
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
            
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        
        phase, prob = phase_detector.predict_phase(lmList)
        
        if phase:
            cv2.putText(img, f"{phase} phase ({prob:.2f})", (50, 50), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
        
        cv2.imshow("Squat Analysis", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Configurable paths
    VIDEO_FOLDER = "DL_Videos"  # Try "DL_Vidoes" if this fails
    VIDEO_FILE = "Recording 2025-03-26 145136.mp4"
    MODEL_FILE = "best_model.pth"
    
    # Try multiple path variations
    video_path = os.path.join(VIDEO_FOLDER, VIDEO_FILE)
    if not os.path.exists(video_path):
        video_path = os.path.join("DL_Vidoes", VIDEO_FILE)  # Alternate spelling
    
    process_video(video_path, MODEL_FILE)