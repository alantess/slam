#include "dataset.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <torch/torch.h>
#include <torchvision/models/resnet.h>
#include <torchvision/vision.h>

namespace fs = std::filesystem;
int main() {
  std::string line;
  std::string root = "/media/alan/seagate/datasets/kitti/cpp/";
  auto ds = KittiDataset(root);

  return 0;
}
