import cv2
import mediapipe as mp
import time
import os
import csv
import math
import argparse
import numpy as np
from CSVComb import CSV_Combiner

# Load class names for YOLO (COCO dataset)
def load_coco_names(filepath="YOLO/coco.names"):
    with open(filepath, "r") as f:
        return [line.strip() for line in f.readlines()]

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
    
    def findPose(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        # Draw skeleton
        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img):
        lmList = []
        if not self.results.pose_landmarks:
            return lmList
        
        h, w = img.shape[:2]
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            if hasattr(self.results, 'pose_world_landmarks'):
                world_lm = self.results.pose_world_landmarks.landmark[id]
                lmList.append([id, cx, cy, lm.x, lm.y, lm.z, world_lm.x, world_lm.y, world_lm.z])
            else:
                lmList.append([id, cx, cy, lm.x, lm.y, lm.z, 0, 0, 0])
                
        return lmList

def load_yolo_model():
    yolo_net = cv2.dnn.readNet("YOLO/yolov3.weights", "YOLO/yolov3.cfg")
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers().flatten()]
    classes = load_coco_names()
    return yolo_net, output_layers, classes

def process_video(video_path: str, rotate_angle=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Couldn't read video stream from file '{video_path}'")
        return None
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ↓ Add these lines here to reduce resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(original_width * 0.5))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(original_height * 0.5))
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    detector = PoseDetector()
    yolo_net, output_layers, class_names = load_yolo_model()
    
    base_filename = os.path.basename(video_path)
    filename_without_ext = os.path.splitext(base_filename)[0]
    os.makedirs("Squat_PoseCSVs", exist_ok=True)
    csv_file = f"Squat_PoseCSVs/pose_data_{filename_without_ext.lower()}.csv"
    
    # Open the CSV file once for writing
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "ID", "Pixel_X", "Pixel_Y", "Norm_X", "Norm_Y", "Norm_Z", 
                        "World_X", "World_Y", "World_Z"])
        
        frame_num = 0
        csv_buffer = []  # Buffer to store the data rows
        
        while True:
            success, img = cap.read()
            if not success:
                break

            # Rotate the entire frame BEFORE any further processing
            if rotate_angle != 0:
                img = rotate_image(img, rotate_angle)
            
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            yolo_net.setInput(blob)
            outputs = yolo_net.forward(output_layers)
            
            boxes, confidences, indexes, class_ids = [], [], [], []
            height, width = img.shape[:2]

            # Define the central region (for example, 50% of the width and height)
            center_x_min = int(width * 0.25)
            center_x_max = int(width * 0.75)
            center_y_min = int(height * 0.25)
            center_y_max = int(height * 0.75)

            for out in outputs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and class_names[class_id] == "person":
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w // 2
                        y = center_y - h // 2

                        # Only consider bounding boxes in the central region
                        if center_x_min <= center_x <= center_x_max and center_y_min <= center_y <= center_y_max:
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
            
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            if len(indexes) > 0:
                index = indexes[0] if isinstance(indexes[0], (int, np.integer)) else indexes[0][0]
                box = boxes[index]
                x, y, w, h = box
                
                # Ensure the bounding box is valid
                if x < 0 or y < 0 or x + w > width or y + h > height:
                    print(f"Invalid bounding box at frame {frame_num}, skipping frame.")
                    continue
                
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cropped_img = img[y:y+h, x:x+w]
                
                # Check if cropped_img is empty
                if cropped_img is None or cropped_img.size == 0:
                    print(f"Empty cropped image at frame {frame_num}, skipping frame.")
                    continue
                
                cropped_with_pose = detector.findPose(cropped_img.copy())
                lmList = detector.findPosition(cropped_img)  # Still use the original for positions
                img[y:y+h, x:x+w] = cropped_with_pose

                # Map the landmarks to the original frame coordinates
                full_frame_lmList = []
                for lm in lmList:
                    lm_id, cx, cy, _, _, _, _, _, _ = lm
                    
                    # Map to full-frame coordinates
                    full_cx = int(cx + x)
                    full_cy = int(cy + y)
                    
                    # Append to the list with the full-frame coordinates
                    full_frame_lmList.append([lm_id, full_cx, full_cy] + lm[3:])
                
                timestamp = frame_num / fps
                for lm in full_frame_lmList:
                    csv_buffer.append([timestamp] + lm)

                # Write to CSV in batches
                if len(csv_buffer) >= 100:  # Flush buffer every 100 rows
                    writer.writerows(csv_buffer)
                    csv_buffer = []  # Reset buffer

            frame_num += 1
            if frame_num % 100 == 0:
                print(f"Processing {video_path}: {frame_num}/{frame_count} frames")

            # resized = cv2.resize(img, (math.floor(img.shape[1] * 0.5), math.floor(img.shape[0] * 0.5)))
            # cv2.imshow("Processed Video", resized)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        # Write any remaining data in the buffer at the end
        if csv_buffer:
            writer.writerows(csv_buffer)

    cap.release()
    cv2.destroyAllWindows()
    return csv_file

def rotate_image(img, angle):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# def determine_rotation_angle(reference_video_path):
#     cap = cv2.VideoCapture(reference_video_path)
#     if not cap.isOpened():
#         return 0
#     success, img = cap.read()
#     if not success:
#         return 0
#     h, w = img.shape[:2]
#     return 90 if h > w * 1.5 else 0

def main():
    parser = argparse.ArgumentParser(description="Pose estimation with MediaPipe and CSV logging.")
    parser.add_argument("video_path", type=str, help="Path to the input video file or folder.")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], 
                       help="Force rotation angle (0, 90, 180, 270 degrees)")
    args = parser.parse_args()
    
    video_path = args.video_path
    rotation_angle = args.rotate
    
    if rotation_angle is None and os.path.isdir(video_path):
        for filename in os.listdir(video_path):
            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                rotation_angle = 0 #determine_rotation_angle(os.path.join(video_path, filename))
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
    CSV_Combiner("Squat_PoseCSVs", "Squat_PoseDataFull.csv")
    print("Pose CSVs combined and split into train and test sets.")

if __name__ == "__main__":
    main()
