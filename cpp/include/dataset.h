#pragma once
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <torch/torch.h>
#include <unordered_map>

namespace fs = std::filesystem;
class KittiDataset : public torch::data::Dataset<KittiDataset> {
public:
  enum Mode { kTrain, kVal };
  explicit KittiDataset(const std::string &root, Mode mode = Mode::kTrain);

  // Returns an example at the index
  torch::data::Example<> get(size_t index) override;

  // Gets the size of the dataset
  torch::optional<size_t> size() const override;

  // Checks to see which folders to use  between train/val
  bool is_train() const noexcept;

  const torch::Tensor &images() const;
  const torch::Tensor &targets() const;

private:
  std::unordered_map<std::string, std::vector<std::string>> ref_files;
  torch::Tensor images_;
  torch::Tensor targets_;
  Mode mode_;
  int n;
};

cv::Mat convert_to_cv(torch::Tensor img);
