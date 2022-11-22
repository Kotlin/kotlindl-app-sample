##  KotlinDL Android inference demo application [![official JetBrains project](http://jb.gg/badges/incubator.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)

[![Slack channel](https://img.shields.io/badge/chat-slack-green.svg?logo=slack)](https://kotlinlang.slack.com/messages/kotlindl/)

||||
| ---------- | ----------- | ----------- |
| <img src="./readme_materials/sheeps.png" alt="sheeps" width="200"/>      |    <img src="./readme_materials/pose.jpg" alt="pose" width="200"/>    | <img src="./readme_materials/face.jpg" alt="face" width="200"/> |


This repo demonstrates how to use KotlinDL for neural network inference on Android devices.
It contains a simple Android app that uses KotlinDL to demonstrate the inference of a bunch of pre-trained models for different computer vision tasks.

The list of demonstrated models includes:
* MobileNetV1 and EfficientNetV4Lite for image classification
* SSDMobileNetV1 and EfficientDetLite0 for object detection
* MoveNet for human pose estimation
* UltraFace320 for Face detection
* Fan2d106Face for Face Alignment

This application is based on CameraX Android API and uses the latest KotlinDL version.
The actual model inference is performed by the [Onnx Runtime](https://github.com/microsoft/onnxruntime).

This example is based on [ort_image_classification example](https://github.com/guoyu-wang/ort_image_classification_android)
