#include "demo_classification.hpp"

namespace Shadow {

inline bool PairCompare(const std::pair<float, int> &lhs,
                        const std::pair<float, int> &rhs) {
  return lhs.first > rhs.first;
}

inline std::vector<int> Argmax(const std::vector<float> &v, int K) {
  std::vector<std::pair<float, int>> pairs;
  for (int i = 0; i < v.size(); ++i) {
    pairs.push_back(std::make_pair(v[i], i));
  }
  std::partial_sort(pairs.begin(), pairs.begin() + K, pairs.end(), PairCompare);
  std::vector<int> result;
  for (int i = 0; i < K; ++i) {
    result.push_back(pairs[i].second);
  }
  return result;
}

void DemoClassification::Test(const std::string &image_file) {
  im_ini_.Read(image_file);
  VecRectF rois{RectF(0, 0, im_ini_.w_, im_ini_.h_)};
  timer_.start();
  Predict(im_ini_, rois, &scores_);
  std::cout << "Predicted in " << timer_.get_millisecond() << " ms"
            << std::endl;
  for (const auto &score_map : scores_) {
    for (const auto &it : score_map) {
      std::cout << it.first << ": ";
      const auto &score = it.second;
      const auto &max_k = Argmax(it.second, 1);
      for (const auto idx : max_k) {
        std::cout << idx << " (" << score[idx] << ") ";
      }
      std::cout << std::endl;
    }
  }
}

void DemoClassification::BatchTest(const std::string &list_file) {
  const auto &image_list = Util::load_list(list_file);
  int num_im = image_list.size(), count = 0;
  double time_cost = 0;
  Process process(20, num_im, "Processing: ");
  const auto &result_file = Util::find_replace_last(list_file, ".", "-result.");
  std::ofstream file(result_file);
  CHECK(file.is_open()) << "Can't open file " << result_file;
  for (const auto &im_path : image_list) {
    im_ini_.Read(im_path);
    VecRectF rois{RectF(0, 0, im_ini_.w_, im_ini_.h_)};
    timer_.start();
    Predict(im_ini_, rois, &scores_);
    time_cost += timer_.get_millisecond();
    PrintDetections(im_path, scores_, &file);
    process.update(count++, &std::cout);
  }
  file.close();
  std::cout << "Processed in: " << time_cost
            << " ms, each frame: " << time_cost / num_im << " ms" << std::endl;
}

void DemoClassification::PrintDetections(
    const std::string &im_name,
    const std::vector<std::map<std::string, VecFloat>> &scores,
    std::ostream *os) {
  *os << im_name << ":" << std::endl;
  for (const auto &score_map : scores) {
    for (const auto &it : score_map) {
      *os << it.first << ": ";
      const auto &score = it.second;
      const auto &max_k = Argmax(it.second, 1);
      for (const auto idx : max_k) {
        *os << idx << " (" << score[idx] << ") ";
      }
      std::cout << std::endl;
    }
  }
  *os << "-------------------------" << std::endl;
}

}  // namespace Shadow
