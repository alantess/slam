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
#include <random>
#include <thread>

#include "utils.h"
namespace fs = std::filesystem;

struct KittiSet {
  public:
  // The mode in which the dataset is loaded
  enum Mode { kTrain, kVal };

  KittiSet(const std::string& root,
           torch::data::transforms::Normalize<>& preprocess,
           Mode mode = Mode::kTrain);

  // Returns the `Example` at the given `index`.
  /* template <typename T = torch::Tensor> */
  void get(size_t index, std::vector<torch::Tensor>& imgs,
           std::vector<torch::Tensor>& depths, std::vector<torch::Tensor>& cams,
           std::vector<torch::Tensor>& poses);

  // Returns the size of the dataset.
  size_t size() const;

  private:
  std::vector<std::map<std::string, std::string>> data;
  std::vector<torch::Tensor> x;
  torch::data::transforms::Normalize<> transforms;
  Mode mode_;
};
// Iterates through a given dataset
struct DataLoader {
  public:
  DataLoader(KittiSet& dataset_, size_t batch_size_, bool shuffle_,
             size_t num_workers_, bool pin_memory_, bool drop_last_);
  size_t get_max_count();
  bool operator()(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                             torch::Tensor>& data);
  void reset();

  private:
  KittiSet dataset;
  std::mt19937_64 mt;
  bool drop_last;
  size_t batch_size;
  bool shuffle;
  bool pin_memory;
  size_t num_workers;
  size_t size;
  size_t count;
  std::vector<size_t> idx;
  size_t max_count;
};

