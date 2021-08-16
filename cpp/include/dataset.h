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
  std::vector<std::map<std::string, std::string>> data;
  std::vector<torch::Tensor> x;

  Mode mode_;
};
// Custom Data Loader
struct DataLoader {
  public:
  DataLoader() = default;
  DataLoader(KittiSet& dataset_, size_t batch_size_, bool shuffle_,
             size_t num_workers_, bool pin_memory_, bool drop_last_);
  size_t get_count_max();
  template <typename T = torch::Tensor>
  bool operator()(std::tuple<T, T, T, T>& data);
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
  size_t max_count;
};

