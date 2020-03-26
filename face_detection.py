#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream
import time
import configparser
import os
import face_recognition


# Get the names of the output layers
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def draw_predict(frame, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

    text = '{:.2f}'.format(conf)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, label_size[1])
    cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_WHITE, 1)


def yolo_process(frame, image_size, conf_threshold, nms_threshold):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1/255, image_size,
                                 [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))

    # fetch captured image dimensions
    (frame_height, frame_width) = frame.shape[:2]
    
    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []
    final_boxes = []
    
    # looping through grid cells
    for out in outs:
        # looping through detectors
        for detection in out:
            # fetch classes probability
            scores = detection[5:]
            # fetch class with maximum probability
            class_id = np.argmax(scores)
            # fetch maximum probability
            confidence = scores[class_id]
            # filter prediction based on threshold value
            if confidence > conf_threshold:
                # fetch validated bounding boxes
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                # add confidences and bounding boxes in list
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)
    
    # fetch legitimate face bounding boxes
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        draw_predict(frame, confidences[i], left, top, left + width,
                     top + height)
    return final_boxes


def resnet_process(frame, image_size, conf_threshold, nms_threshold):
    blob = cv2.dnn.blobFromImage(frame, 1.0,
                                    image_size, (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    (h, w) = frame.shape[:2]
    final_boxes = []
    confidences = []
    boxes = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the 'confidence' is
        # greater than the minimum confidence
        if confidence > conf_threshold:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])        
            box = box.astype("int")
            # startX, startY, endX, endY
            confidences.append(float(confidence))
            boxes.append([box[0], box[1], box[2]-box[0], box[3]-box[1]])
    
    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    try:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            final_boxes.append(np.array([left, top, left + width,
                         top + height]))
            draw_predict(frame, confidences[i], left, top, left + width,
                     top + height)
    except:
        for i, box in enumerate(boxes):
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            final_boxes.append(np.array([left, top, left + width,
                         top + height]))
            draw_predict(frame, confidences[i], left, top, left + width,
                     top + height)
    return final_boxes


if __name__=='__main__':
    # load configuration file
    config = configparser.ConfigParser()
    config.read('config.conf')
    
    # read input parameters
    # path for the input image/video/webcam
    src = eval(config['inputs']['source_path'])

    # perform detections on image/video
    src_type = eval(config['inputs']['source_type'])

    # load the model name
    model_type = eval(config['inputs']['model'])

    # load the output directory path
    output_dir = eval(config['outputs']['output_dir'])
    
    if model_type == 'yolov3':
        # read model configuration & weights
        model_cfg = eval(config['yolo_model_params']['model_config'])
        model_weights = eval(config['yolo_model_params']['model_weights'])

        # read model parameters
        image_size = eval(config['yolo_model_params']['target_size'])
        conf_threshold = eval(config['yolo_model_params']['conf_threshold'])
        nms_threshold = eval(config['yolo_model_params']['nms_threshold'])

        # load the network
        net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    elif model_type == 'resnet':
        # read model configuration & weights
        model_cfg = eval(config['resnet_model_params']['model_config'])
        model_weights = eval(config['resnet_model_params']['model_weights'])

        # read model parameters
        image_size = eval(config['resnet_model_params']['target_size'])
        conf_threshold = eval(config['resnet_model_params']['conf_threshold'])
        nms_threshold = eval(config['resnet_model_params']['nms_threshold'])

        # load the network
        net = cv2.dnn.readNetFromCaffe(model_cfg, model_weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Default colors
        COLOR_BLUE = (255, 0, 0)
        COLOR_GREEN = (0, 255, 0)
        COLOR_RED = (0, 0, 255)
        COLOR_WHITE = (255, 255, 255)
        COLOR_YELLOW = (0, 255, 255)
        
        if src_type=='webcam':
            cap = cv2.VideoCapture(src)
            time.sleep(2.0)

            fps = FPS().start()

            while True:

                ret, frame = cap.read()
                if ret!=True:
                    print('Not able to load image from webcam')
                    break

                if model_type=='yolov3':
                    faces = yolo_process(frame, image_size, conf_threshold, nms_threshold)
                elif model_type=='resnet':
                    faces = resnet_process(frame, image_size, conf_threshold, nms_threshold)
                elif model_type=='dlib_hog':
                    faces = np.array(face_recognition.face_locations(frame, model="hog"))
                    for i, value in enumerate(faces):
                        faces[i][0], faces[i][1], faces[i][2], faces[i][3] = faces[i][3], faces[i][0], faces[i][1], faces[i][2]  
                        # draw bounding boxes
                        left, top, right, bottom = faces[i][0], faces[i][1], faces[i][2], faces[i][3]
                        cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)
                elif model_type=='dlib_cnn':
                    faces = np.array(face_recognition.face_locations(frame, model="cnn"))
                    for i, value in enumerate(faces):
                        faces[i][0], faces[i][1], faces[i][2], faces[i][3] = faces[i][3], faces[i][0], faces[i][1], faces[i][2]  
                        # draw bounding boxes
                        left, top, right, bottom = faces[i][0], faces[i][1], faces[i][2], faces[i][3]
                        cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

                # initialize the set of information we'll displaying on the frame
                info = [
                    ('number of faces detected', '{}'.format(len(faces)))
                ]

                for (i, (txt, val)) in enumerate(info):
                    text = '{}: {}'.format(txt, val)
                    cv2.putText(frame, text, (10, (i * 20) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

                cv2.imshow('faces', frame)

                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):
                    break

                # update the FPS counter
                fps.update()

            # stop the timer and display FPS information
            fps.stop()
            print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

            cap.release()
            cv2.destroyAllWindows()

        if src_type=='image':
            frame = cv2.imread(src)

            if model_type=='yolov3':
                faces = yolo_process(frame, image_size, conf_threshold, nms_threshold)
            elif model_type=='resnet':
                faces = resnet_process(frame, image_size, conf_threshold, nms_threshold)
            elif model_type=='dlib_hog':
                faces = np.array(face_recognition.face_locations(frame, model="hog"))
                for i, value in enumerate(faces):
                    faces[i][0], faces[i][1], faces[i][2], faces[i][3] = faces[i][3], faces[i][0], faces[i][1], faces[i][2]  
                    # draw bounding boxes
                    left, top, right, bottom = faces[i][0], faces[i][1], faces[i][2], faces[i][3]
                    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)
            elif model_type=='dlib_cnn':
                faces = np.array(face_recognition.face_locations(frame, model="cnn"))
                for i, value in enumerate(faces):
                    faces[i][0], faces[i][1], faces[i][2], faces[i][3] = faces[i][3], faces[i][0], faces[i][1], faces[i][2]  
                    # draw bounding boxes
                    left, top, right, bottom = faces[i][0], faces[i][1], faces[i][2], faces[i][3]
                    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

            # initialize the set of information we'll displaying on the frame
            info = [
                ('number of faces detected', '{}'.format(len(faces)))
            ]

            for (i, (txt, val)) in enumerate(info):
                text = '{}: {}'.format(txt, val)
                cv2.putText(frame, text, (10, (i * 20) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

            cv2.imshow('faces', cv2.resize(frame, image_size))
            cv2.waitKey(1)
            time.sleep(2.0)
            cv2.destroyAllWindows()

            cv2.imwrite(os.path.join(output_dir,"prediction.jpg"), frame)



