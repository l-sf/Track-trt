

# Track-trt

![Language](https://img.shields.io/badge/language-c++-brightgreen) ![Language](https://img.shields.io/badge/CUDA-12.1-brightgreen) ![Language](https://img.shields.io/badge/TensorRT-8.6.1.6-brightgreen) ![Language](https://img.shields.io/badge/OpenCV-4.5.5-brightgreen) ![Language](https://img.shields.io/badge/ubuntu-20.04-brightorigin)



## Introduction

基于 TensorRT 的 C++ 高性能 单目标跟踪 推理，支持单目标跟踪算法 OSTrack、LightTrack。

其中 OSTrack 为ViT模型，适用于服务端计算设备，LightTrack 为NAS搜索出来的轻量CNN架构，适用于边缘端计算设备。请按需使用。

更多 TensorRT 部署模型，请移步仓库 [github](https://github.com/l-sf/Linfer) 



## Project Build and Run

1. install cuda/tensorrt/opencv

   [cuda/tensorrt/opencv](https://github.com/l-sf/Notes/blob/main/notes/Ubuntu20.04_install_tutorials.md#%E4%BA%94cuda--cudnn--tensorrt-install)

2. compile engine

   1. 下载onnx模型 [google driver](https://drive.google.com/drive/folders/16ZqDaxlWm1aDXQsjsxLS7yFL0YqzHbxT?usp=sharing) 或者 跟踪教程自己导出

   2. ```bash
      cd Track-trt/workspace
      bash compile_engine.sh
      ```

3. build 

   ```bash
   # 修改CMakeLists.txt中 cuda/tensorrt/opencv 为自己的路径
   cd Track-trt
   mkdir build && cd build
   cmake .. && make -j4
   ```

4. run

   视频文件输入：

   ```bash
   cd Track-trt/workspace
   ./pro 0 "bag.avi"
   ```

   摄像头输入：

   ```bash
   cd Track-trt/workspace
   ./pro 1 0
   ```

   图片序列输入：

   ```bash
   cd Track-trt/workspace
   ./pro 2 "Woman/img/%04d.jpg"
   ```



## Speed Test

在 Jetson Orin Nano 8G 上进行测试，包括整个流程（即预处理+推理+后处理）

|   Method   | Precision | Resolution | Average Latency |
| :--------: | :-------: | :--------: |:---------------:|
| LightTrack |   fp16    |  256x256   |      11ms       |
|  OSTrack   |   fp16    |  256x256   |      27ms       |



## onnx导出

[LightTrack](./lighttrack/README.md) 

[OSTrack](./ostrack/README.md) 



## Reference

[tensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro.git) 

