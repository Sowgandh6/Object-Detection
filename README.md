# Object-Detection
Deep Learning for Automated Threat Detection for Airport X-ray Baggage Screening

X-ray security screening plays a crucial role for ensuring public safety and preventing potential threats. The process of manually searching and visually identifying prohibited objects in X-ray images is time-consuming and prone to human errors. To fill this void and improve the efficiency of this process, We investigate object detection algorithms based on deep learning to find and categorize prohibited items in X-ray pictures. Specifically, we aim to detect the presence of Guns, Knives, Wrenches, Pliers, and Scissors in the images using Object Detection models. We evaluate the performance of basic Convolutional Neural Networks (CNN), Faster Region-based Convolutional Neural Networks (Faster-RCNN), You Only Look Once (YOLO), (YOLOv5 & YOLOv7) on the Sixray dataset. Our study provides insights into the effectiveness of Object Detection models for detecting prohibited objects in X-ray images and highlights the importance of using deep learning techniques for improving security inspection processes. This is done as a part of the course work in master's degree. We conclude that YOLOv7 model achieved highest mAP value of 94% outperforming the remaining models.
Each folder includes the implementations for that specific model.

Performance of YOLOV5 model:
![results](https://github.com/Sowgandh6/Object-Detection/assets/74649012/941485de-621f-4e96-986d-a40ad4d009c1)

Performance of YOLOV7 model:
![results](https://github.com/Sowgandh6/Object-Detection/assets/74649012/017ad3ae-8b17-4513-9007-24818fd6d6b4)

Precision and Recall values of YOLOv5 and YOLOv7 models:
![6](https://github.com/Sowgandh6/Object-Detection/assets/74649012/a17cd5fd-7c37-49ac-a64b-8502a098a705)

Detection of threat objects using YOLOv5:
![1](https://github.com/Sowgandh6/Object-Detection/assets/74649012/727c8697-f513-4123-b26e-c37f7861f325)

Detection of threat objects using YOLOv7:
![1](https://github.com/Sowgandh6/Object-Detection/assets/74649012/c091bb2e-c927-4de3-ac06-6ef498539e13)

## Acknowledgments

Some parts of this project uses the YOLOv5 and YOLOv7 original implementation by Ultralytics LLC. We would like to thank the authors for their contributions to the computer vision community.
