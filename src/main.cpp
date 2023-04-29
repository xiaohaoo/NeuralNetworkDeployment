//
//  main.cpp
//  NeuralNetworkDeployment
//
//  Created by xiaohao on 2022/3/4.
//

#include "main.h"

using namespace std;

static const char *model_path;

void createEnv(const void *path) {
    model_path = (char *) path;
}

std::size_t prediction(char *img_bytes, const int img_size) {
    clock_t start_time = clock();

    static Ort::AllocatorWithDefaultOptions allocatorWithDefaultOptions;
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX");
    static Ort::Session session = Ort::Session(env, model_path, Ort::SessionOptions());
    static vector<int64_t> modal_input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    static vector<int64_t> modal_output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    static Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    static const char *modal_input_names[] = {session.GetInputName(0, allocatorWithDefaultOptions)};
    static const char *modal_output_names[] = {session.GetOutputName(0, allocatorWithDefaultOptions)};

    cv::Mat input_image = cv::imdecode(cv::Mat(1, img_size, CV_8UC1, img_bytes, 0), cv::ImreadModes::IMREAD_COLOR);

    cv::resize(input_image, input_image, cv::Size((int) modal_input_shape.at(1), (int) modal_input_shape.at(2)), cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(input_image, input_image, cv::ColorConversionCodes::COLOR_BGR2RGB);
    input_image.convertTo(input_image, CV_32F, 1.0 / 255.0, -1);

    vector<int64_t> input_tensor_dims = {1, input_image.rows, input_image.cols, input_image.channels()};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image.ptr<float>(), input_image.total() * input_image.channels(), input_tensor_dims.data(), input_tensor_dims.size());

    vector<Ort::Value> output_value = session.Run(Ort::RunOptions(), modal_input_names, &input_tensor, input_tensor_dims.at(0), modal_output_names, input_tensor_dims.at(0));
    auto *out_arr = output_value[0].GetTensorMutableData<float>();

    printf("-> 模型预测用时：%.3f秒\n", (double) (clock() - start_time) / CLOCKS_PER_SEC);
    return max_element(out_arr, out_arr + modal_output_shape.at(1)) - out_arr;
}

