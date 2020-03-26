I find the following repository really good to start working on yolov3 face detection-
```https://github.com/sthanhng/yoloface
```
Download weights from the link given in above repository and save in folder yolo

pip install -r requirements.txt

check the prediction on Abhinav2.png with resnet and yolo. it will definitely surprise you.

dlib_hog and dlib_cnn are easy to work in ubuntu (pip install face_recognition) but for windows you will need to install dlib to make it work
for more info- check out https://github.com/ageitgey/face_recognition

download dlib.whl file (19.16.0)
pip instal dlib
pip install face_recognition