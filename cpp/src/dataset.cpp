#include "dataset.h"

#include "utils.h"

constexpr auto kTrain = "train.txt";
constexpr auto kVal = "val.txt";
constexpr auto valSize = 2266;
constexpr auto trainSize = 13510;
constexpr auto num_val_folders = 9;
constexpr auto num_train_folders = 67;
unsigned int n_threads = std::thread::hardware_concurrency() / 2;

std::mutex my_mutex;
using namespace utils;

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
void get_files(std::vector<std::string> folders,
               std::vector<torch::Tensor> &poses,
               std::vector<torch::Tensor> &imgs,
               std::vector<torch::Tensor> &depths,
               std::vector<torch::Tensor> &cams) {
  // Slows down code dramatically
  // Negates segmenation faults
  std::lock_guard<std::mutex> g(my_mutex);
  int v_count = 0;
  std::string img_ext(".jpg");
  std::string depth_ext(".png");

  for (auto &f : folders) {
    for (const auto &p : fs::directory_iterator(f)) {
      if (p.path().extension() == img_ext) {
        cv::Mat cv_image = cv::imread(p.path());
        auto tensor_img = CVtoTensor(cv_image,30,30);
        imgs.push_back(tensor_img);
      } else if (p.path().extension() == depth_ext) {
        cv::Mat cv_depth = cv::imread(p.path());
        auto tensor_depth = CVtoTensor(cv_depth,30,30);
        depths.push_back(tensor_depth);
      } else if (p.path().filename() == "cam.txt") {
        auto input = txtToTensor(p.path());
        auto out = torch::from_blob(input.data(), {3, 3});
        cams.push_back(out);
      } else if (p.path().filename() == "poses.txt") {
        auto input = txtToTensor(p.path());
        auto len = (int)input.size() / 12;
        auto out = torch::from_blob(input.data(), {len, 3, 4});
        poses.push_back(out);
      }
    }
  }

}

std::pair<torch::Tensor, torch::Tensor> read_data(const std::string &root,
                                                  bool train) {
  std::vector<torch::Tensor> poses;
  std::vector<torch::Tensor> cams;

  std::vector<torch::Tensor> imgs;
  std::vector<std::jthread> workers;
  std::vector<torch::Tensor> depths;

  auto file = train ? root + kTrain : root + kVal;
  auto num_samples = train ? trainSize : valSize;
  auto folders = folder_iter(root, file);
  int prev = 0;
  int inc = folders.size() / n_threads;
  int j = inc;

  auto depths_t = torch::empty({num_samples, 3, 300, 300}, torch::kFloat);
  auto images = torch::empty({num_samples, 3, 300, 300}, torch::kFloat);

  depths.reserve(num_samples);
  imgs.reserve(num_samples);
  poses.reserve((int)folders.size());
  cams.reserve((int)folders.size());

  while (j <= n_threads) {
    std::vector<std::string> f_alloc(folders.begin() + prev,
                                     folders.begin() + j);

    workers.emplace_back(std::jthread(get_files, f_alloc, std::ref(poses),
                                      std::ref(imgs), std::ref(depths),
                                      std::ref(cams)));

    prev = j;
    j += inc;
  }
  for (auto &w : workers) {
    /* if(w.joinable()) */ 
    /*   w.join(); */
    w.detach();
  }

  imgs.shrink_to_fit();
  depths.shrink_to_fit();
  poses.shrink_to_fit();
  cams.shrink_to_fit();

  return {images, depths_t};
}

KittiSet::KittiSet(const std::string &root, Mode mode) : mode_(mode) {
  auto [images, depth] = read_data(root, mode);
}

torch::data::Example<> KittiSet ::get(size_t index) {
  return {images_[index], depth_[index]};
}

torch::optional<size_t> KittiSet::size() const { return images_.size(0); }
