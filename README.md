

# Track-trt

![Language](https://img.shields.io/badge/language-c++-brightgreen) ![Language](https://img.shields.io/badge/CUDA-12.1-brightgreen) ![Language](https://img.shields.io/badge/TensorRT-8.6.1.6-brightgreen) ![Language](https://img.shields.io/badge/OpenCV-4.5.5-brightgreen) ![Language](https://img.shields.io/badge/ubuntu-20.04-brightorigin)

## Introduction

基于 TensorRT 的 C++ 高性能 单目标跟踪 推理。

支持单目标跟踪算法 OSTrack、LightTrack，分别适用于服务端和边缘端的计算设备，请按需使用。



## Project Build and Run

1. install cuda/tensorrt/opencv

   [reference](https://github.com/l-sf/Notes/blob/main/notes/Ubuntu20.04_install_tutorials.md#%E4%BA%94cuda--cudnn--tensorrt-install) 

2. compile engine

   1. 下载onnx模型 [google driver](https://drive.google.com/drive/folders/16ZqDaxlWm1aDXQsjsxLS7yFL0YqzHbxT?usp=sharing) 

   2. ```bash
      cd Linfer/workspace
      # 修改其中的onnx路径
      bash compile_engine.sh
      ```

3. build 

   ```bash
   # 修改CMakeLists.txt中cuda/tensorrt/opencv为自己的路径
   cd Linfer
   mkdir build && cd build
   cmake .. && make -j4
   ```

4. run

   ```bash
   cd Linfer/workspace
   ./pro
   ```



## Speed Test

在 Jetson Orin Nano 8G 上进行测试，测试包括整个流程（即预处理+推理+后处理）

|   Model    | Precision | Resolution | FPS(bs=1) | FPS(bs=4) |
| :--------: | :-------: | :--------: | :-------: | :-------: |
|  yolov5_s  |   fp16    |  640x640   |   96.06   |   100.9   |
|  yolox_s   |   fp16    |  640x640   |   79.64   |   85.00   |
|   yolov7   |   int8    |  640x640   |   49.55   |   50.42   |
|  yolov8_n  |   fp16    |  640x640   |  121.94   |  130.16   |
|  yolov8_s  |   fp16    |  640x640   |   81.40   |   84.74   |
|  yolov8_l  |   fp16    |  640x640   |    13     |    tbd    |
| rtdetr_r50 |   fp16    |  640x640   |    12     |    tbd    |



## Reference

- [tensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro.git) 
- [infer](https://github.com/shouxieai/infer.git) 
- [Video：详解TensorRT的C++/Python高性能部署，实战应用到项目](https://www.bilibili.com/video/BV1Xw411f7FW/?share_source=copy_web&vd_source=4bb05d1ac6ff39b7680900de14419dca) 

