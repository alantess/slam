#include "dataset.h"

#include "utils.h"

constexpr auto kTrain = "train.txt";
constexpr auto kVal = "val.txt";
constexpr auto valSize = 2266;
constexpr auto num_val_folders = 9;
constexpr auto num_train_folders = 67;
std::mutex my_mutex;

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
               std::vector<torch::Tensor> &poses, std::vector<cv::Mat> &imgs,
               std::vector<cv::Mat> &depths, torch::Tensor &cams) {
  // Slows down code dramatically
  /* std::lock_guard<std::mutex> g(my_mutex); */
  int v_count = 0;
  std::string img_ext(".jpg");
  std::string depth_ext(".png");

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
        cams[v_count] = torch::from_blob(input.data(), {3, 3});
        v_count++;
        // Need to append to a tensor
      }
      if (p.path().filename() == "poses.txt") {
        auto input = txtToTensor(p.path());
        auto len = (int)input.size() / 12;
        auto out = torch::from_blob(input.data(), {len, 3, 4});
        poses.push_back(out);

        // Need to append to a tensor
      }
    }
  }
}

std::pair<torch::Tensor, torch::Tensor> read_data(const std::string &root,
                                                  bool train) {
  torch::Tensor cams = torch::empty({num_val_folders, 3, 3});
  std::vector<torch::Tensor> poses;

  std::vector<cv::Mat> imgs;
  std::vector<std::jthread> processes;
  std::vector<cv::Mat> depths;

  unsigned int n_threads = std::thread::hardware_concurrency();
  auto file = train ? root + kTrain : root + kVal;
  auto folders = folder_iter(root, file);
  int prev = 0;
  int inc = folders.size() / n_threads;
  int j = inc;

  while (j <= n_threads) {
    std::vector<std::string> f_alloc(folders.begin() + prev,
                                     folders.begin() + j);

    processes.push_back(std::jthread(get_files, f_alloc, std::ref(poses),
                                     std::ref(imgs), std::ref(depths),
                                     std::ref(cams)));
    prev = j;
    j += inc;
  }
  for (auto &p : processes) {
    if (p.joinable()) p.join();
  }

  auto depth_samples = (int)depths.size();
  auto img_samples = (int)imgs.size();
  std::cout << img_samples << "\n" << depth_samples;
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
