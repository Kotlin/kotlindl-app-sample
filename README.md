An Android image classification app using [Onnx Runtime Mobile](https://github.com/microsoft/onnxruntime) and MobileNet V1

This example is loosely based on [Google CodeLabs - Getting Started with CameraX](https://codelabs.developers.google.com/codelabs/camerax-getting-started)

# Build Instructions
## Requirements
- Android SDK 29+
- Android NDK r21+

Download the MobileNet V1 model, label file and prebuilt Onnx Runtime arm64 AAR package [here](https://1drv.ms/u/s!Auaxv_56eyubgQbphWRzoO_ykl2e?e=VVJMGt)

Copy MobileNet V1 model and the label file to `app/src/main/res/raw/`

Copy the `onnxruntime-release-1.7.0.aar` to `app/libs`

[Optional]Build the Onnx Runtime for Android arm64 (see [Build Instruction](https://www.onnxruntime.ai/docs/how-to/build.html#android))


<img width=40% src="images/screenshot_1.jpg" alt="App Screenshot" />
