import cv2
import mediapipe as mp
import time 
import os 
import csv
import argparse

class poseDetector():
    
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=True, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)
    
    def findPose(self, img, draw=True, rotate=False):
       
        if rotate: 
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # mediapipe uses RGB not GBR 
        self.results = self.pose.process(imgRGB)
        # print(self.results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
        
    
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks and self.results.pose_world_landmarks:
            for id, (lm, world_lm) in enumerate(zip(self.results.pose_landmarks.landmark, self.results.pose_world_landmarks.landmark)):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy, lm.x, lm.y, lm.z, world_lm.x, world_lm.y, world_lm.z])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return lmList

def main():
    parser = argparse.ArgumentParser(description="Pose estimation with MediaPipe and CSV logging.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    args = parser.parse_args()
    
    video_path = args.video_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Couldn't read video stream from file '{video_path}'")
        return
    pTime = 0
    detector = poseDetector()

    base_filename = os.path.basename(video_path)
    filename_without_ext = os.path.splitext(base_filename)[0]
    csv_file = f"pose_data_{filename_without_ext}.csv"
    file = open(csv_file, mode='w', newline='')
    writer = csv.writer(file)
    writer.writerow(["Time", "ID", "Pixel_X", "Pixel_Y", "Norm_X", "Norm_Y", "Norm_Z", "World_X", "World_Y", "World_Z"])
    
    start_time = time.time()
    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to read frame from video stream")
            break
        img = detector.findPose(img,rotate=True)
        lmList = detector.findPosition(img, draw=False)
        # print(lmList[14])
        # cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0,0,255), cv2.FILLED) example of how to locate and draw one point
        cTime = time.time()        
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        cv2.imshow("Image", img)
        timestamp = time.time() - start_time
        for lm in lmList:
            writer.writerow([timestamp] + lm)
        file.flush() 
        cv2.waitKey(1)


if __name__ == "__main__":
    main()