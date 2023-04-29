# NeuralNetworkDeployment

本项目主要用来探索工业级（区别于学界只用Python构建Idea即可）深度学习模型的部署，为了追求卓越的性能，使用C/C++语言构建。支持Mac、Linux、Windows系统。

## 实施方案

目前，主要探索的方案有两个：

- [x] ONNX Runtime + OpenCV
- [ ] Torch + OpenCV

由于TensorFlow API过于混乱以及未向下兼容，暂不考虑部署使用。

## 安装与使用

使用cmake构建，请检查系统中是否正确安装cmake。

```shell
mkdir build
cd build
cmake ..
make
```