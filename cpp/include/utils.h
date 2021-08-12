#pragma once

#include <torch/torch.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

namespace utils {
cv::Mat TensortoCV(torch::Tensor x);
torch::Tensor CVtoTensor(cv::Mat x, int KRows, int kCols);

}  // namespace utils
