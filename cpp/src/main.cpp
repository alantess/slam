#include <chrono>
#include <iostream>

#include "dataset.h"
#include "utils.h"
int main() {
  auto cuda_avail = torch::cuda::is_available();
  auto device = (cuda_avail ? torch::kCUDA : torch::kCPU);
  auto start_time = std::chrono::high_resolution_clock::now();
  auto BATCH_SIZE = 3;
  bool SHUFFLE = true;
  bool DROP_LAST = true;
  bool PIN_MEM = true;
  size_t EPOCHS = 1;
  size_t WORKERS = 4;
  std::vector<double> mean = {0.406, 0.456, 0.485};
  std::vector<double> std = {0.225, 0.224, 0.229};
  auto preproc = torch::data::transforms::Normalize<>(mean, std);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
      mini_batch;

  // Create Data loaders
  std::cout << "\nConstructing datasets...\n";

  std::string root = "/media/alan/seagate/datasets/kitti/cpp/";
  auto train_dataset = KittiSet(root, preproc);
  auto val_dataset = KittiSet(root, preproc, KittiSet::Mode::kVal);
  DataLoader train_loader(train_dataset, BATCH_SIZE, SHUFFLE, WORKERS, PIN_MEM,
                          DROP_LAST);

  DataLoader val_loader(val_dataset, BATCH_SIZE, SHUFFLE, WORKERS, PIN_MEM,
                        DROP_LAST);

  for (size_t i = 0; i < EPOCHS; i++) {
    while (train_loader(mini_batch)) {
      auto image = std::get<0>(mini_batch).to(device);
      /* auto depths = std::get<1>(mini_batch).to(device); */
      /* auto cams = std::get<2>(mini_batch).to(device); */
      /* auto poses = std::get<3>(mini_batch).to(device); */
      break;

    }
  }

  auto current_time = std::chrono::high_resolution_clock::now();
  std::cout << "\nExection Time: "
            << std::chrono::duration_cast<std::chrono::seconds>(current_time -
                                                                start_time)
                   .count()
            << " seconds" << std::endl;

  return 0;
}
