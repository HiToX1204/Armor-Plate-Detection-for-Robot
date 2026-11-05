# hongxihe 校内赛标志物识别

这是个人作业目录，使用传统 OpenCV 图像处理方法完成视频中的校内赛
标志物识别。程序会逐帧读取视频，检测标志物中心由四个白色不完整
L 形角块组成的方框，并在输出视频中绘制目标四边形和四个角点。

## 环境

- Linux
- GCC 11.4.0
- C++17
- CMake 3.22.1
- OpenCV 4.5.4

Ubuntu/Debian 可以使用以下命令安装依赖：

```bash
sudo apt update
sudo apt install build-essential cmake libopencv-dev
```

## 构建

从仓库根目录执行：

```bash
cmake -S src/hongxihe -B build/hongxihe \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/hongxihe --parallel
```

运行程序：

```bash
./build/hongxihe/marker_recognition \
  data/raw/marker_video.avi \
  src/hongxihe/evidence/hongxihe_marker_result.mp4
```

程序参数依次为输入视频和输出视频。输出目录不存在时会自动创建。
提交用的结果视频位于 `src/hongxihe/evidence/`，时长不超过 30 秒。
运行程序时也可以使用其他本地输出路径；构建目录和临时结果不应提交。

当前处理流程：

```text
灰度化 → 白色区域阈值分割 → 形态学处理 → 轮廓提取
→ L 形角块筛选 → 四个角块合并 → 目标四角点生成 → 输出标注视频
```
