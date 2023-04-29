#pragma once

#include <iostream>
#include <array>
#include <algorithm>
#include <unistd.h>
#include "onnx/onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"

#define C_EXPORT extern "C"


C_EXPORT void createEnv(const void *path);
C_EXPORT std::size_t prediction(char *img_bytes, const int img_size);
