#include "dataset.h"

#include "utils.h"

constexpr auto kTrain = "train.txt";
constexpr auto kVal = "val.txt";
constexpr auto valSize = 2266;
constexpr auto trainSize = 13510;
constexpr auto WIDTH = 256;
constexpr auto HEIGHT = 256;
constexpr auto num_val_folders = 9;
constexpr auto num_train_folders = 67;
unsigned int n_threads = std::thread::hardware_concurrency() / 2;

std::mutex my_mutex;
using namespace utils;
// Retrieves folders
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

template <typename S = std::vector<std::string>>
std::vector<std::map<std ::string, std::string>> get_files(S folders, S &imgs,
                                                           S &depths, S &poses,
                                                           S &cams) {
  std::map<std ::string, std::string> m;
  std::string img_ext(".jpg");
  std::string depth_ext(".png");
  std::vector<std::map<std ::string, std::string>> data;

  for (auto &f : folders) {
    int count = 0;
    auto cam_file = f + "/cam.txt";
    cams.push_back(cam_file);
    auto pose_file = f + "/poses.txt";
    poses.push_back(pose_file);
    for (const auto &p : fs::directory_iterator(f)) {
      if (p.path().extension() == img_ext) {
        imgs.push_back(p.path());
        count++;
      } else if (p.path().extension() == depth_ext) {
        depths.push_back(p.path());
      }
    }
    for (int i = 0; i < count; i++) {
      m["cam"] = cam_file;
      m["pose"] = pose_file;
      m["depth"] = depths[i];
      m["image"] = imgs[i];
      data.push_back(m);
      m.clear();
    }
  }

  return data;
}
template <typename T = std::vector<std::string>>
std::vector<std::map<std ::string, std::string>> read_data(
    const std::string &root, bool train) {
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

  auto data = get_files(folders, imgs, depths, poses, cams);

  imgs.shrink_to_fit();
  depths.shrink_to_fit();
  poses.shrink_to_fit();
  cams.shrink_to_fit();

  return data;
}

// Dataset
KittiSet::KittiSet(const std::string &root,
                   torch::data::transforms::Normalize<> &preprocess, Mode mode)
    : mode_(mode), transforms(preprocess) {
  // Gather Data
  auto samples = read_data(root, mode == Mode::kTrain);
  data = samples;
}

std::map<std::string, torch::Tensor> KittiSet::get(size_t index) {
  vals.clear();
  auto sample = data[index];
  auto cv_img = cv::imread(sample["image"]);
  vals["image"] = CVtoTensor(cv_img, WIDTH, HEIGHT);
  auto cv_depth = cv::imread(sample["depth"]);
  vals["depth"] = CVtoTensor(cv_depth, WIDTH, HEIGHT);
  auto cam_data = txtToTensor(sample["cam"]);
  vals["cam"] = torch::from_blob(cam_data.data(), {3, 3});
  auto pose_data = txtToTensor(sample["pose"]);
  auto len = (int)pose_data.size() / 12;
  vals["pose"] = torch::from_blob(pose_data.data(), {len, 3, 4});

  return vals;
}

size_t KittiSet::size() const { return (int)data.size(); }

// Data Loader
DataLoader::DataLoader(KittiSet &dataset_, size_t batch_size_, bool shuffle_,
                       size_t num_workers_, bool pin_memory_, bool drop_last_)
    : dataset(dataset_),
      batch_size(batch_size_),
      shuffle(shuffle_),
      num_workers(num_workers_),
      pin_memory(pin_memory_),
      drop_last(drop_last_) {
  DataLoader::reset();
  size = DataLoader::get_max_count();
  idx.reserve(size);
  for (size_t i = 0; i < size; i++) idx.at(i) = i;

  if (drop_last) {
    max_count = std::floor((float)size / (float)batch_size);
    if ((max_count == 0) && (size > 0)) {
      max_count = 1;
    }
  } else {
    max_count = std::ceil((float)size / (float)batch_size);
  }
  mt.seed(std::rand());
}

/* template <typename T = torch::Tensor> */
/* bool DataLoader::operator()(std::tuple<T, T, T, T> &data) {} */

size_t DataLoader::get_max_count() { return dataset.size(); }
void DataLoader::reset() {
  count = 0;
  return;
}
