
import cv2
import time
import PoseModule as pm

video_path = "PoseVids/Bench2.MOV" 

cap = cv2.VideoCapture(video_path)

pTime = 0
detector = pm.poseDetector()
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
    cv2.waitKey(1)