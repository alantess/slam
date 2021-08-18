#include <chrono>
#include <iostream>

#include "dataset.h"

int main() {
  auto start_time = std::chrono::high_resolution_clock::now();
  auto BATCH_SIZE = 3;
  auto preproc = torch::data::transforms::Normalize<>({0.5,0.5,0.5}, {0.5,0.5,0.5});

  // Create Data loaders
  std::cout << "\nConstructing datasets...\n";

  std::string root = "/media/alan/seagate/datasets/kitti/cpp/";
  auto train_dataset = KittiSet(root, preproc);


  auto val_dataset = KittiSet(root, preproc, KittiSet::Mode::kVal );

  /* // Train Function */
  /* std::cout << "Starting training....\n"; */
  /* for (auto& batch : *train_loader) { */
  /*   auto img = batch.data; */
  /*   auto target = batch.target; */
  /*   std::cout << img.sizes(); */
  /*   std::cout << target.sizes(); */

  /*   break; */
  /* } */

  auto current_time = std::chrono::high_resolution_clock::now();
  std::cout << "\nExection Time: "
            << std::chrono::duration_cast<std::chrono::seconds>(current_time -
                                                                start_time)
                   .count()
            << " seconds" << std::endl;

  return 0;
}
