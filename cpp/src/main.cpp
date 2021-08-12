#include <chrono>
#include <iostream>

#include "dataset.h"

int main() {
  auto start_time = std::chrono::high_resolution_clock::now();

  std::string root = "/media/alan/seagate/datasets/kitti/cpp/";
  auto train_dataset = KittiSet(root)
                           .map(torch::data::transforms::Normalize<>(
                               {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                           .map(torch::data::transforms::Stack<>());

  auto current_time = std::chrono::high_resolution_clock::now();
  std::cout << "\nProgram has been running for "
            << std::chrono::duration_cast<std::chrono::seconds>(current_time -
                                                                start_time)
                   .count()
            << " seconds" << std::endl;

  return 0;
}
