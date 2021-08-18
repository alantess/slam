#include <chrono>
#include <iostream>

#include "dataset.h"
#include "utils.h"

int main() {
  auto start_time = std::chrono::high_resolution_clock::now();
  auto BATCH_SIZE = 3;
  bool SHUFFLE = true;
  bool DROP_LAST = true;
  bool PIN_MEM = false;
  size_t WORKERS = 4;
  std::vector<double> mean = {0.406, 0.456, 0.485};
  std::vector<double> std = {0.225, 0.224, 0.229};
  auto preproc = torch::data::transforms::Normalize<>(mean, std);

  // Create Data loaders
  std::cout << "\nConstructing datasets...\n";

  std::string root = "/media/alan/seagate/datasets/kitti/cpp/";
  auto train_dataset = KittiSet(root, preproc);
  auto val_dataset = KittiSet(root, preproc, KittiSet::Mode::kVal);
  DataLoader train_loader(train_dataset, BATCH_SIZE, SHUFFLE, WORKERS, PIN_MEM,
                          DROP_LAST);

  DataLoader val_loader(val_dataset, BATCH_SIZE, SHUFFLE, WORKERS, PIN_MEM,
                        DROP_LAST);
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
