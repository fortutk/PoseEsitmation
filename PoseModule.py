import cv2
import mediapipe as mp
import time 
import os 
import csv
import argparse
import numpy as np
from CSVComb import CSV_Combiner

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
            model_complexity=1 if self.upBody else 0,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
    
    def findPose(self, img, rotate_angle=0):
        """Process image and return rotated image if needed"""
        if rotate_angle == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotate_angle == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif rotate_angle == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        return img
        
    def findPosition(self, img):
        """Return landmarks only for the person closest to center"""
        lmList = []
        if not self.results.pose_landmarks:
            return lmList
            
        # Find the person closest to center
        h, w = img.shape[:2]
        center = (w//2, h//2)
        
        # Calculate distance to center for nose landmark (index 0)
        nose = self.results.pose_landmarks.landmark[0]
        cx, cy = int(nose.x * w), int(nose.y * h)
        
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            if hasattr(self.results, 'pose_world_landmarks'):
                world_lm = self.results.pose_world_landmarks.landmark[id]
                lmList.append([id, cx, cy, lm.x, lm.y, lm.z, world_lm.x, world_lm.y, world_lm.z])
            else:
                lmList.append([id, cx, cy, lm.x, lm.y, lm.z, 0, 0, 0])
                
        return lmList

def process_video(video_path: str, rotate_angle=0):
    """Process video without displaying it for faster processing"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Couldn't read video stream from file '{video_path}'")
        return None
    
    # Get video properties for progress reporting
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # default assumption if fps not available
    
    detector = PoseDetector()
    
    # Prepare CSV file
    base_filename = os.path.basename(video_path)
    filename_without_ext = os.path.splitext(base_filename)[0]
    os.makedirs("Squat_PoseCSVs", exist_ok=True)
    csv_file = f"Squat_PoseCSVs/pose_data_{filename_without_ext.lower()}.csv"
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "ID", "Pixel_X", "Pixel_Y", "Norm_X", "Norm_Y", "Norm_Z", 
                        "World_X", "World_Y", "World_Z"])
        
        frame_num = 0
        
        while True:
            success, img = cap.read()
            if not success:
                break
                
            # Process frame
            img = detector.findPose(img, rotate_angle=rotate_angle)
            lmList = detector.findPosition(img)
            
            # Write to CSV
            timestamp = frame_num / fps  # more accurate than time.time() for video timestamps
            for lm in lmList:
                writer.writerow([timestamp] + lm)
            
            frame_num += 1
            if frame_num % 100 == 0:
                print(f"Processing {video_path}: {frame_num}/{frame_count} frames")
    
    cap.release()
    return csv_file

def determine_rotation_angle(reference_video_path):
    """Determine optimal rotation angle from a reference video"""
    cap = cv2.VideoCapture(reference_video_path)
    if not cap.isOpened():
        return 0
    
    # Get first frame
    success, img = cap.read()
    if not success:
        return 0
    
    # Check orientation - simple heuristic based on aspect ratio
    h, w = img.shape[:2]
    if h > w * 1.5:  # portrait
        return 90
    return 0

def main():
    parser = argparse.ArgumentParser(description="Pose estimation with MediaPipe and CSV logging.")
    parser.add_argument("video_path", type=str, help="Path to the input video file or folder.")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], 
                       help="Force rotation angle (0, 90, 180, 270 degrees)")
    args = parser.parse_args()
    
    video_path = args.video_path
    rotation_angle = args.rotate
    
    # Determine rotation angle if not specified
    if rotation_angle is None and os.path.isdir(video_path):
        # Find first video in directory to determine rotation
        for filename in os.listdir(video_path):
            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                rotation_angle = determine_rotation_angle(os.path.join(video_path, filename))
                break
    
    start_time = time.time()
    
    if os.path.isdir(video_path):
        for filename in os.listdir(video_path):
            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                full_path = os.path.join(video_path, filename)
                process_video(full_path, rotate_angle=rotation_angle or 0)
                print(f"Processed {filename}")
    else:
        process_video(video_path, rotate_angle=rotation_angle or 0)
    
    print(f"All videos processed in {time.time() - start_time:.2f} seconds")
    print("Combining CSVs...")
    
    CSV_Combiner("Squat_PoseCSVs/Ryans", "Squat_PoseDataRyan.csv")
    print("Pose CSVs combined and split into train and test sets.")

if __name__ == "__main__":
    main()