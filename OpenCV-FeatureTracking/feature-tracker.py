import argparse, os, time

# Set environment variables
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

import numpy as np
import cv2

# Fast
fast = cv2.FastFeatureDetector_create(threshold=65, nonmaxSuppression=5)

# Orb
orb = cv2.ORB_create()


def use_gft(frame):
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   corners = cv2.goodFeaturesToTrack(gray,
                                     2000,
                                     0.01,
                                     0.1,
                                     useHarrisDetector=True)
   corners = np.int0(corners)
   for i in corners:
      x, y = i.ravel()
      cv2.circle(frame, (x, y), 3, (0, 255, 0), 1)
   return frame


def feature_tracker(detector, frame):
   if detector.lower() == 'fast':
      kp = fast.detect(frame, None)
   elif detector.lower() == 'orb':
      kp = orb.detect(frame, None)
   else:
      return use_gft(frame)

   frame = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0))
   return frame


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Feature Tracking using OpenCV")
   parser.add_argument("-s",
                       "--source",
                       help="Source Video",
                       type=str,
                       required=True)
   parser.add_argument("-d",
                       "--detector",
                       help="Detector",
                       type=str,
                       default="Good Feature to Track")

   args = parser.parse_args()
   if args.source == '0':
      args.source = 0

   print('[INFO] Running Feature Tracker with :')
   print(f'  * Source   : {"Webcam" if args.source == 0 else args.source}')
   print(f'  * Detector : {args.detector}')

   cap = cv2.VideoCapture(args.source)

   if cap.isOpened():
      cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
      cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

   print('[INFO] Starting GUI, Please type "q" to exit.')
   while cap.isOpened():
      start = time.time()
      ret, frame = cap.read()

      if not ret:
         break

      frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_AREA)

      # Main
      frame = feature_tracker(args.detector, frame)

      # FPS counter
      fps = round(1 / (time.time() - start), 3)

      cv2.putText(
          frame,
          f"FPS : {fps}",
          (10, 30),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (255, 255, 255),
          2,
      )

      cv2.imshow("Webcam" if args.source == 0 else args.source, frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
         break

   cap.release()
   cv2.destroyAllWindows()
   print('[INFO] Done!')
