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
// Change vector from matrix to string
// Append the paths intead of teh matrix
std::vector<float> txtToTensor(std::string file) {
  std::ifstream input_stream(file);
  std::vector<float> input;
  input.insert(input.begin(), std::istream_iterator<float>(input_stream),
               std::istream_iterator<float>());
  return input;
}
void get_files(std::vector<std::string> folders, std::vector<std::string> &imgs,
               std::vector<std::string> &depths,
               std::vector<std::string> &poses,
               std::vector<std::string> &cams) {
  int v_count = 0;
  std::string img_ext(".jpg");
  std::string depth_ext(".png");

  for (auto &f : folders) {
    for (const auto &p : fs::directory_iterator(f)) {
      if (p.path().extension() == img_ext) {
        imgs.push_back(p.path());
      } else if (p.path().extension() == depth_ext) {
        depths.push_back(p.path());
      } else if (p.path().filename() == "cam.txt") {
        cams.push_back(p.path());
      } else if (p.path().filename() == "poses.txt") {
        poses.push_back(p.path());
      }
    }
  }
}

template <typename T = std::vector<std::string>>
std::tuple<T, T, T, T> read_data(const std::string &root, bool train) {
  std::vector<std::string> poses;
  std::vector<std::string> cams;
  std::vector<std::string> depths;
  std::vector<std::string> imgs;

  auto file = train ? root + kTrain : root + kVal;
  auto num_samples = train ? trainSize : valSize;
  auto folders = folder_iter(root, file);

  depths.reserve(num_samples);
  imgs.reserve(num_samples);
  poses.reserve((int)folders.size());
  cams.reserve((int)folders.size());

  get_files(folders, imgs, depths, poses, cams);

  imgs.shrink_to_fit();
  depths.shrink_to_fit();
  poses.shrink_to_fit();
  cams.shrink_to_fit();

  return {imgs, depths, poses, cams};
}

// Constructor
KittiSet::KittiSet(const std::string &root, Mode mode) : mode_(mode) {
  // Setup multithreading

  // Gather Data
  auto [images, depth, poses, cam] = read_data(root, mode == Mode::kTrain);
}

torch::data::Example<> KittiSet ::get(size_t index) {
  return {images_[index], depth_[index]};
}

torch::optional<size_t> KittiSet::size() const { return images_.size(0); }
