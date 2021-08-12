#include "utils.h"

cv::Mat utils::TensortoCV(torch::Tensor x) {
  x = x.permute({1, 2, 0});
  x = x.mul(0.5).add(0.5).mul(255).clamp(0, 255).to(torch::kByte);
  x = x.contiguous();
  int height = x.size(0);
  int width = x.size(1);
  cv::Mat output(cv::Size{width, height}, CV_8UC3);
  std::memcpy((void*)output.data, x.data_ptr(), sizeof(torch::kU8) * x.numel());

  return output.clone();
}

torch::Tensor utils::CVtoTensor(cv::Mat x, int kRows, int kCols ) {
  cv::resize(x, x, cv::Size{kRows, kCols}, 0, 0, cv::INTER_LINEAR);
  cv::cvtColor(x, x, cv::COLOR_BGR2RGB);
  auto x_tensor = torch::from_blob(x.data, {kRows, kCols, 3}, torch::kByte);
  x_tensor = x_tensor.permute({2, 0, 1}).toType(torch::kFloat).div_(255);
  return x_tensor;
}
