#include "dataset.h"

#include "utils.h"

constexpr auto kTrain = "train.txt";
constexpr auto kVal = "val.txt";

std::vector<std::string> folder_iter(const std::string &root,
                                     std::string file) {
  std::ifstream txtFile(file);
  std::vector<std::string> folders;
  std::string lines;
  // Read text file
  if (txtFile.is_open()) {
    while (getline(txtFile, lines)) {
      auto folder_dir = root + lines;
      folders.push_back(folder_dir);
    }
    txtFile.close();
  }
  return folders;
}

std::vector<float> txtToTensor(std::string file) {
  std::ifstream input_stream(file);
  std::vector<float> input;
  input.insert(input.begin(), std::istream_iterator<float>(input_stream),
               std::istream_iterator<float>());
  return input;
}

std::pair<torch::Tensor, torch::Tensor> read_data(const std::string &root,
                                                  bool train) {
  std::string img_ext(".jpg");
  std::string depth_ext(".png");
  std::vector<cv::Mat> imgs;
  std::vector<cv::Mat> depths;

  auto file = train ? root + kTrain : root + kVal;
  auto folders = folder_iter(root, file);
  // Iterate through each folder;
  for (auto &f : folders) {
    for (const auto &p : fs::directory_iterator(f)) {
      if (p.path().extension() == img_ext) {
        cv::Mat cv_image = cv::imread(p.path());
        imgs.push_back(cv_image);
      }
      if (p.path().extension() == depth_ext) {
        cv::Mat cv_depth = cv::imread(p.path());
        depths.push_back(cv_depth);
      }
      if (p.path().filename() == "cam.txt") {
        auto input = txtToTensor(p.path());
        auto a = torch::from_blob(input.data(), {3, 3});
      }
      if (p.path().filename() == "poses.txt") {
        auto input = txtToTensor(p.path());
        auto len = (int)input.size() / 12;
        auto a = torch::from_blob(input.data(), {len, 3, 4});
        std::cout << a;
      }
    }
  }

  auto depth_samples = (int)depths.size();
  auto img_samples = (int)imgs.size();
  auto depths_t = torch::empty({depth_samples, 3, 300, 300}, torch::kFloat);
  auto images = torch::empty({img_samples, 3, 300, 300}, torch::kFloat);
  return {images, depths_t};
}

KittiSet::KittiSet(const std::string &root, Mode mode) : mode_(mode) {
  auto [images, depth] = read_data(root, mode);
}

torch::data::Example<> KittiSet ::get(size_t index) {
  return {images_[index], depth_[index]};
}

torch::optional<size_t> KittiSet::size() const { return images_.size(0); }
