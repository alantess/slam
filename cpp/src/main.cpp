#include <iostream>

#include "dataset.h"

int main() {
  std::string root = "/media/alan/seagate/datasets/kitti/cpp/";
  auto train_dataset = KittiSet(root)
                           .map(torch::data::transforms::Normalize<>(
                               {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                           .map(torch::data::transforms::Stack<>());
  return 0;
}
