# Feature Tracking - OpenCV

<p align="center">
  <img src="./sample.gif" alt="preview" width=640 height=360 
  style="border-radius: 10px;" />
</p>

Make feature tracker script with OpenCV.

## Run on local

Clone this repo, setup `venv`, and install requirements for this project.

Run tracker on **example video**

```bash
python feature-tracker.py -s day.mp4
```

or live using your **webcam**

```bash
python feature-tracker.py -s 0
```

**Change Detector**

The detector by default is set to be `Good Feature to Track` module from `OpenCV`, 
but you can use `FastFeatureDetector` or `ORB` by configuring `-d` args.

```bash
python feature-tracker.py -s day.mp4 -d fast # using FastFeatureDetector
python feature-tracker.py -s day.mp4 -d orb  # using ORB
```
