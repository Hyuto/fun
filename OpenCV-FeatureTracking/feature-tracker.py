import argparse, os, time, logging

# Set environment variables
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

import numpy as np
import cv2

# Global Config
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO)

# Tracker
# Fast with treshold=65 and nunmaxSuppression=5
# and ORB with default configuration
FAST = cv2.FastFeatureDetector_create(threshold=65, nonmaxSuppression=5)
ORB = cv2.ORB_create()


def feature_tracker(detector, frame):
    # Tracking feature
    # Use goodFeaturesToTrack by default if user did
    # not specified detector and wrongly type detector
    if detector.lower() == 'fast':
        kp = FAST.detect(frame, None)
    elif detector.lower() == 'orb':
        kp = ORB.detect(frame, None)
    else:
        # Change to grayscale, find corners and
        # add circle to every corner detected
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # to gray
        # goodFeaturesToTrack with maxCorner=2000, qualityLevel=0.001,
        # and minDistance=0.1 using HarrisDetector
        corners = cv2.goodFeaturesToTrack(
            gray, 2000, 0.01, 0.1, useHarrisDetector=True)  # detect corner
        for i in np.int0(corners):  # Loop trough every corner
            x, y = i.ravel()
            cv2.circle(frame, (x, y), 3, (0, 255, 0), 1)  # draw circle

        return frame

    frame = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0))
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature Tracking using OpenCV")
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

    # If source==0 then we use Webcam as source
    if args.source == '0':
        args.source = 0

    logging.info('Running Feature Tracker with :')
    print(f' * Source   : {"Webcam" if args.source == 0 else args.source}')
    print(f' * Detector : {args.detector}')

    # Open Video
    cap = cv2.VideoCapture(args.source)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    logging.info('Starting GUI, Please type "q" to exit.')

    while cap.isOpened():  # Loop trough the video
        fps, start = '-', time.time()
        ret, frame = cap.read()  # read

        # break when video has no more frame
        if not ret:
            break

        frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_AREA)

        # Main
        frame = feature_tracker(args.detector, frame)

        # FPS counter
        try:
            fps = round(1 / (time.time() - start), 3)
        except ZeroDivisionError:  # if the time difference ~ 0
            logging.warning('Zero divison when calculating fps')

        # print FPS
        cv2.putText(
            frame,
            f"FPS : {fps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Show frame
        cv2.imshow("Webcam" if args.source == 0 else args.source, frame)

        # Handle key q pressed to quit session
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info('Exited at')
            break

    # Realease & destroy windows
    cap.release()
    cv2.destroyAllWindows()

    logging.info('Done!')
