I have implemented four face detection models here namely-
1) yolov3
2) resnet
3) dlib hog
4) dlib cnn

follow the instructions based on the detector to be used.

<b>for yolo v3-</b>
Download weights from the link in the following repository and save in folder yolo - https://github.com/sthanhng/yoloface

<b>for resnet-</b>
ping me for the weights file and save it in thee folder resnet

<b>for dlib cnn/hog-</b>
for more info- check out https://github.com/ageitgey/face_recognition
for ubuntu-
```
pip install face_recogntion
```
for windows-
```
pip install dlib>19.7
pip install face_recognition
```

install the required libraries-
```
pip install -r requirements.txt
```

Some interesting findings-
check the prediction on Abhinav2.png with resnet and yolo. it will definitely surprise you.

Instructions to run-
1) modify config.conf file as following-
model = 'your prefered detector'
source_type = 'webcam or video or image'
source_path = 'camera id or video/image path'
output_dir = 'directory to save detected video/image'

2) run python face_detection.py
