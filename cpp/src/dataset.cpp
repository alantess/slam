#include "dataset.h"

#define HEIGHT 256
#define WIDTH 512

std::unordered_map<std::string, std::vector<std::string>>
get_folders(std::string root, std::string directory) {
  std::unordered_map<std::string, std::vector<std::string>> mappings;
  // Extension corresponding base image and depth
  std::string depth_ext(".png");
  std::string img_ext(".jpg");
  // Holds paths to folders and files
  std::vector<std::string> folder_names, depth_files, img_files;
  // Appends the correct folders
  std::ifstream myFiles(directory);
  std::copy(std::istream_iterator<std::string>(myFiles),
            std::istream_iterator<std::string>(),
            std::back_inserter(folder_names));

  // Iterate through each folder and append each file
  for (int i = 0; i < folder_names.size(); i++) {
    folder_names[i] = root + folder_names[i];
    // Traverse all the files  through each folder
    for (auto &p : fs::directory_iterator(folder_names[i])) {
      if (p.path().extension() == depth_ext) {
        depth_files.push_back(p.path());
      } else if (p.path().extension() == img_ext) {
        img_files.push_back(p.path());
      }
    }
  }
  mappings["images"] = img_files;
  mappings["depth"] = depth_files;

  return mappings;
};
// Converts the image into a tensor
torch::Tensor img_to_tensor(std::string path) {
  cv::Mat img = cv::imread(path, 0);
  cv::Mat dst;
  if (img.empty()) {
    std::cout << "Could not read the image: " << path << std::endl;
  }
  cv::resize(img, img, cv::Size(WIDTH, HEIGHT));
  img.convertTo(img, CV_32FC3, 1.0f / 255.0f);
  cv::normalize(img, dst, 0, 1, cv::NORM_MINMAX);

  torch::Tensor tensor_img =
      torch::from_blob(dst.data, {1, dst.rows, dst.cols, 1}, torch::kFloat32);
  tensor_img = tensor_img.permute({0, 3, 1, 2});

  return tensor_img;
}

KittiDataset::KittiDataset(const std::string &root, Mode mode) : mode_(mode) {
  // check if file exist
  if (!fs::exists(root)) {
    std::cout << "\nInvalid Directory\n";
    exit(EXIT_FAILURE);
  }

  std::string file_ext;
  if (mode_ == KittiDataset::Mode::kTrain) {
    file_ext = root + "train.txt";
  } else if (mode_ == KittiDataset::Mode::kVal) {
    file_ext = root + "val.txt";
  } else {
    std::cout << "INVALID MODE\n";
  }

  auto files = get_folders(root, file_ext);
  assert(files["images"].size() == files["depth"].size());
  int n = files["images"].size(); // Size of dataset

  images_ = torch::empty({n, 1, HEIGHT, WIDTH});
  targets_ = torch::empty({n, 1, HEIGHT, WIDTH});
}

torch::data::Example<> KittiDataset::get(size_t index) {
  auto img = img_to_tensor(ref_files["images"][index]);
  auto depth = img_to_tensor(ref_files["depth"][index]);
  images_[index] = img[0];
  targets_[index] = depth[0];

  return {images_[index], targets_[index]};
}

torch::optional<size_t> KittiDataset::size() const { return n; }

bool KittiDataset::is_train() const noexcept { return mode_ == Mode::kTrain; }

const torch::Tensor &KittiDataset::images() const { return images_; }

const torch::Tensor &KittiDataset::targets() const { return targets_; }

cv::Mat convert_to_cv(torch::Tensor img) {
  img = img[0].permute({1, 2, 0});

  cv::Mat output_mat(cv::Size{WIDTH, HEIGHT}, CV_32FC3, img.data_ptr());
  return output_mat;
}
