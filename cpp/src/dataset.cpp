#include "dataset.h"

#define HEIGHT 256
#define WIDTH 832

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
  cv::Mat img = cv::imread(path);
  if (img.empty()) {
    std::cout << "Could not read the image: " << path << std::endl;
  }

  img.convertTo(img, CV_32FC3, 1.0f / 255.0f);
  /* cv::imshow("frame", img); */

  torch::Tensor tensor_img =
      torch::from_blob(img.data, {1, HEIGHT, WIDTH, 1}, torch::kFloat);
  tensor_img = tensor_img.permute({0, 3, 2, 1});
  std::cout << tensor_img;
  return tensor_img;
}

// Retrieves the files from the dataset;
std::unordered_map<std::string, torch::Tensor> read_data(std::string root,
                                                         std::string file_ext) {

  auto files = get_folders(root, file_ext);
  assert(files["images"].size() == files["depth"].size());
  auto n = files["images"].size(); // Size of dataset

  std::unordered_map<std::string, torch::Tensor> tensor_data;

  auto img = img_to_tensor(files["depth"][0]);

  img = img[0].permute({1, 2, 0});
  /* img = img.mul(255).clamp(0, 255); */
  cv::Mat output_mat(cv::Size{WIDTH, HEIGHT}, CV_8UC1, img.data_ptr());
  cv::imshow("Window", output_mat);
  cv::waitKey(0);

  return tensor_data;
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
  auto data = read_data(root, file_ext);
}

torch::data::Example<> KittiDataset::get(size_t index) {
  return {images_[index], targets_[index]};
}

torch::optional<size_t> KittiDataset::size() const { return images_.size(0); }

bool KittiDataset::is_train() const noexcept { return mode_ == Mode::kTrain; }

const torch::Tensor &KittiDataset::images() const { return images_; }

const torch::Tensor &KittiDataset::targets() const { return targets_; }

cv::Mat convert_to_cv(torch::Tensor img) {
  img = img[0].permute({1, 2, 0});

  cv::Mat output_mat(cv::Size{WIDTH, HEIGHT}, CV_32FC3, img.data_ptr());
  return output_mat;
}
