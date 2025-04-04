import cv2
import mediapipe as mp
import torch
import numpy as np
import os
import argparse
from BaseClassifier import PhaseLSTM
from collections import deque

# Configuration constants
CONFIG = {
    'phase_threshold_up': 0.7,  # Threshold for "UP"
    'phase_threshold_down': 0.3,  # Threshold for "DOWN"
    'phase_threshold_stable': 0.5,  # Threshold for "STABLE" (mid-squat)
    'min_confidence': 0.5,
    'seq_length': 30,
    'debug': True  # Enable detailed logging
}

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.landmark_history = deque(maxlen=5)

    def process_frame(self, image):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        world_landmarks = None
        if results.pose_world_landmarks:
            world_landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_world_landmarks.landmark]
            self.landmark_history.append(world_landmarks)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
            )
        return image, world_landmarks

class SquatClassifier:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PhaseLSTM(input_size=4).to(self.device)
        self.sequence = deque(maxlen=CONFIG['seq_length'])
        self.load_model(model_path)
        
        # Normalization stats (should match training)
        self.angle_means = np.array([142.58, 157.02, 52.76, 32.77, 142.58])
        self.angle_stds = np.array([28.15, 23.07, 26.82, 24.26, 28.15])
        
        # State tracking
        self.frame_count = 0
        self.prediction_history = []
        self.last_valid_angles = None
        self.last_valid_landmarks = None

    def load_model(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()

    def validate_landmarks(self, landmarks):
        """Check for valid landmark coordinates"""
        if not landmarks or len(landmarks) < 33:
            if CONFIG['debug']:
                print("Invalid landmarks - insufficient points detected")
            return False
            
        # Check for implausible coordinates
        for lm in landmarks[:33]:  # Only check first 33 standard landmarks
            if any(abs(coord) > 10 for coord in lm):  # Unrealistic coordinates
                if CONFIG['debug']:
                    print(f"Implausible landmark coordinates detected: {lm}")
                return False
                
        # Additional checks for key joints visibility
        key_joints = [11, 12, 23, 24, 25, 26, 27, 28]  # Shoulders, hips, knees, ankles
        for joint in key_joints:
            if joint >= len(landmarks):
                if CONFIG['debug']:
                    print(f"Missing key joint {joint} in landmarks")
                return False
        return True

    def calculate_angles(self, landmarks):
        """Calculate 5 key joint angles with enhanced validation"""
        if not self.validate_landmarks(landmarks):
            return None
            
        joints = {
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_shoulder': 11, 'right_shoulder': 12
        }
        
        def angle(a, b, c):
            """Robust 3D angle calculation with validation"""
            ba = np.array(a) - np.array(b)
            bc = np.array(c) - np.array(b)
            
            with np.errstate(all='raise'):
                try:
                    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    return np.arccos(np.clip(cosine, -1, 1))
                except:
                    return None
        
        try:
            angles = []
            # Left knee angle
            l_knee = angle(landmarks[joints['left_hip']],
                          landmarks[joints['left_knee']],
                          landmarks[joints['left_ankle']])
            if l_knee is None: raise ValueError("Invalid left knee angle")
            
            # Right knee angle
            r_knee = angle(landmarks[joints['right_hip']],
                          landmarks[joints['right_knee']],
                          landmarks[joints['right_ankle']])
            if r_knee is None: raise ValueError("Invalid right knee angle")
            
            # Left hip angle
            l_hip = angle(landmarks[joints['left_shoulder']],
                         landmarks[joints['left_hip']],
                         landmarks[joints['left_knee']])
            if l_hip is None: raise ValueError("Invalid left hip angle")
            
            # Right hip angle
            r_hip = angle(landmarks[joints['right_shoulder']],
                         landmarks[joints['right_hip']],
                         landmarks[joints['right_knee']])
            if r_hip is None: raise ValueError("Invalid right hip angle")
            
            # Hip alignment angle
            hip_align = angle(landmarks[joints['left_knee']],
                            landmarks[joints['left_hip']],
                            landmarks[joints['right_hip']])
            if hip_align is None: raise ValueError("Invalid hip alignment")
            
            angles = [l_knee, r_knee, l_hip, r_hip, hip_align]
            self.last_valid_angles = angles
            self.last_valid_landmarks = landmarks
            
            if CONFIG['debug']:
                angles_deg = [np.degrees(a) for a in angles]
                print("Calculated angles (degrees):")
                print(f"Left knee: {angles_deg[0]:.1f}°")
                print(f"Right knee: {angles_deg[1]:.1f}°")
                print(f"Left hip: {angles_deg[2]:.1f}°")
                print(f"Right hip: {angles_deg[3]:.1f}°")
                print(f"Hip alignment: {angles_deg[4]:.1f}°")
            
            return angles
            
        except Exception as e:
            if CONFIG['debug']:
                print(f"Angle calculation error: {e}")
            return self.last_valid_angles

    def calculate_velocity(self, current_angles, prev_angles):
        """Calculate angular velocity between frames"""
        if not prev_angles or not current_angles:
            return None
        return np.array(current_angles) - np.array(prev_angles)

    def normalize_angles(self, angles):
        """Safe normalization with validation"""
        if angles is None:
            return None
        normalized = (np.array(angles) - self.angle_means) / self.angle_stds
        
        if CONFIG['debug']:
            print(f"Normalized angles: {normalized}")
            
        # Check for extreme values that might indicate problems
        if any(abs(n) > 5 for n in normalized):
            print(f"Warning: Extreme normalized angle value detected: {normalized}")
            return None
            
        return normalized

    def predict(self, landmarks):
        self.frame_count += 1
        
        angles = self.calculate_angles(landmarks)
        if angles is None:
            if CONFIG['debug']:
                print(f"Frame {self.frame_count}: Using last valid angles")
            angles = self.last_valid_angles
            if angles is None:
                return None, None
        
        # Calculate and log velocity
        if self.last_valid_angles:
            velocity = self.calculate_velocity(angles, self.last_valid_angles)
            if CONFIG['debug'] and velocity is not None:
                velocity_deg = np.degrees(velocity)
                # If velocity is a single value (scalar), print it directly.
                if velocity_deg.size == 1:
                    print(f"Angular velocity: {velocity_deg[0]:.1f}°/frame")
                else:
                    # If it's an array, print each element separately.
                    print(f"Angular velocity (degrees/frame): {', '.join(f'{v:.1f}' for v in velocity_deg)}")
        
        norm_angles = self.normalize_angles(angles)
        if norm_angles is None:
            return None, None
            
        self.sequence.append(norm_angles)
        
        if CONFIG['debug'] and len(self.sequence) >= 5:
            print("\nSequence buffer (last 5 normalized angle sets):")
            for i, angles in enumerate(list(self.sequence)[-5:]):
                print(f"Frame {self.frame_count - len(self.sequence) + i + 1}: {angles}")
        
        if len(self.sequence) == CONFIG['seq_length']:
            with torch.no_grad():
                input_tensor = torch.tensor([list(self.sequence)], dtype=torch.float32).to(self.device)
                prob_up = self.model(input_tensor).item()
                
                if CONFIG['debug']:
                    print(f"Raw model output probability for UP: {prob_up:.4f}")
                
                # Define the logic for "UP", "DOWN", and "STABLE"
                if prob_up > CONFIG['phase_threshold_up']:
                    phase = "UP"
                    prob = prob_up
                elif prob_up < CONFIG['phase_threshold_down']:
                    phase = "DOWN"
                    prob = prob_up
                else:
                    phase = "STABLE"
                    prob = 1 - abs(prob_up - 0.5)  # Higher probability for being stable
                
                self.prediction_history.append((phase, prob))
                
                return phase, prob
                
        return None, None


def visualize_angles(frame, angles, width, height):
    """Visualize angles on the output frame"""
    if angles:
        angles_deg = [np.degrees(a) for a in angles]
        y_pos = height - 150
        
        cv2.putText(frame, "Joint Angles:", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        angle_labels = [
            f"L Knee: {angles_deg[0]:.1f}°",
            f"R Knee: {angles_deg[1]:.1f}°",
            f"L Hip: {angles_deg[2]:.1f}°", 
            f"R Hip: {angles_deg[3]:.1f}°",
            f"Hip Align: {angles_deg[4]:.1f}°"
        ]
        
        for i, label in enumerate(angle_labels):
            cv2.putText(frame, label, (20, y_pos + 30*(i+1)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def process_video(video_path, model_path):
    print(f"\nStarting video processing: {video_path}")
    detector = PoseDetector()
    classifier = SquatClassifier(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    print(f"Video info: {width}x{height} @ {fps:.1f} fps")
    print(f"Sequence length: {CONFIG['seq_length']} frames (~{CONFIG['seq_length']/fps:.1f}s)")
    
    cv2.namedWindow("Squat Phase Detection", cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, landmarks = detector.process_frame(frame)
        phase, confidence = classifier.predict(landmarks)
        
        # Display phase prediction
        if phase:
            if phase == "UP":
                color = (0, 255, 0)
            elif phase == "DOWN":
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)  # Stable (Yellow)
            
            cv2.putText(processed_frame, f"{phase} ({confidence:.2f})", 
                       (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Display system status
        status_text = [
            f"Buffer: {len(classifier.sequence)}/{CONFIG['seq_length']}",
            f"Frames: {classifier.frame_count}"
        ]
        
        for i, text in enumerate(status_text):
            cv2.putText(processed_frame, text, (20, height-30*i-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Visualize angles
        visualize_angles(processed_frame, classifier.last_valid_angles, width, height)
        
        cv2.imshow("Squat Phase Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print comprehensive summary
    print("\n==== Processing Complete ====")
    print(f"Total frames processed: {classifier.frame_count}")
    print(f"Valid frames with angles: {len(classifier.sequence)}")
    
    if classifier.prediction_history:
        ups = sum(1 for p in classifier.prediction_history if p[0] == "UP")
        downs = sum(1 for p in classifier.prediction_history if p[0] == "DOWN")
        stables = sum(1 for p in classifier.prediction_history if p[0] == "STABLE")
        total = len(classifier.prediction_history)
        
        print(f"\nPredictions made: {total}")
        print(f"UP phases: {ups} ({ups/total:.1%})")
        print(f"DOWN phases: {downs} ({downs/total:.1%})")
        print(f"STABLE phases: {stables} ({stables/total:.1%})")
        
        avg_confidence = sum(p[1] for p in classifier.prediction_history) / total
        print(f"Average confidence: {avg_confidence:.2f}")
        
        # Confidence distribution
        print("\nConfidence Distribution:")
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i in range(len(bins)-1):
            count = sum(1 for p in classifier.prediction_history 
                       if bins[i] <= p[1] < bins[i+1])
            print(f"{bins[i]:.1f}-{bins[i+1]:.1f}: {count} predictions")
    else:
        print("\nNo predictions were made (sequence buffer never filled)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Squat Phase Detection")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="best_model.pth", help="Path to trained model")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    CONFIG['debug'] = args.debug
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
        exit(1)
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        exit(1)
    
    process_video(args.video, args.model)