Most part of this code is from https://github.com/lmeulen/PeopleCounter. This code has been modified to create an application to detect crowd and social distancing.


Count people in video (webcam or file) with YOLO

Download pre-trained YOLO v3 model from Darknet (https://pjreddie.com/darknet/yolo/) 
and place in yolo-coco directory. Required files:
- coco.names
- yolov3.cfg
- yolov3.weights

App can be configured with ini file.

Usage:

python detector.py -c configfile

Example inifiles are provided
