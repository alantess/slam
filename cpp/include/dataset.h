#pragma once

#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
namespace fs = std::filesystem;

struct KittiSet : torch::data::datasets::Dataset<KittiSet> {
  public:
  // The mode in which the dataset is loaded
  enum Mode { kTrain, kVal };

  explicit KittiSet(const std::string& root, Mode mode = Mode::kTrain);

  // Returns the `Example` at the given `index`.
  torch::data::Example<> get(size_t index) override;

  // Returns the size of the dataset.
  torch::optional<size_t> size() const override;

  private:
  std::vector<std::jthread> workers;
  torch::Tensor images_;
  torch::Tensor targets_;
  torch::Tensor depth_;
  torch::Tensor Rt_;
  torch::Tensor K_;

  Mode mode_;
};
