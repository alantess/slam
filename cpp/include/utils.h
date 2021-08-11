#pragma once


#include <iostream>
#include<opencv2/opencv.hpp>
#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace utils {
  cv::Mat TensortoCV(torch::Tensor x);
  torch::Tensor CVtoTensor(cv::Mat x);

}
