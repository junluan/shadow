#include "demo_classify.hpp"

#include "algorithm/classify.hpp"

namespace Shadow {

DemoClassify::DemoClassify(const std::string& method_name) {
  if (method_name == "classify") {
    method_ = std::make_shared<Classify>();
  } else {
    LOG(FATAL) << "Unknown method " << method_name;
  }
}

void DemoClassify::Test(const std::string& image_file) {
  im_ini_.Read(image_file);
  timer_.start();
  method_->Predict(im_ini_, RectF(0, 0, im_ini_.w_, im_ini_.h_), &scores_);
  LOG(INFO) << "Predicted in " << timer_.get_millisecond() << " ms";
  PrintConsole(scores_, 1);
}

void DemoClassify::BatchTest(const std::string& list_file) {
  const auto& image_list = Util::load_list(list_file);
  int num_im = static_cast<int>(image_list.size()), count = 0;
  double time_cost = 0;
  ProcessBar process_bar(20, num_im, "Processing: ");
  const auto& result_file = Util::find_replace_last(list_file, ".", "-result.");
  std::ofstream file(result_file);
  CHECK(file.is_open()) << "Can't open file " << result_file;
  for (const auto& im_path : image_list) {
    im_ini_.Read(im_path);
    timer_.start();
    method_->Predict(im_ini_, RectF(0, 0, im_ini_.w_, im_ini_.h_), &scores_);
    time_cost += timer_.get_millisecond();
    PrintStream(im_path, scores_, 1, &file);
    process_bar.update(count++, &std::cout);
  }
  file.close();
  LOG(INFO) << "Processed in: " << time_cost
            << " ms, each frame: " << time_cost / num_im << " ms";
}

void DemoClassify::PrintConsole(const std::map<std::string, VecFloat>& scores,
                                int top_k, bool split) {
  for (const auto& it : scores) {
    const auto& score = it.second;
    const auto& max_k = Util::top_k(score, top_k);
    std::stringstream ss;
    ss << it.first << ": ";
    for (const auto idx : max_k) {
      ss << idx << " (" << score[idx] << ") ";
    }
    LOG(INFO) << ss.str();
  }
  if (split) {
    LOG(INFO) << "------------------------------";
  }
}

void DemoClassify::PrintStream(const std::string& im_name,
                               const std::map<std::string, VecFloat>& scores,
                               int top_k, std::ostream* os) {
  *os << im_name << ":" << std::endl;
  for (const auto& it : scores) {
    const auto& score = it.second;
    const auto& max_k = Util::top_k(score, top_k);
    *os << it.first << ": ";
    for (const auto idx : max_k) {
      *os << idx << " (" << score[idx] << ") ";
    }
    *os << std::endl;
  }
  *os << "------------------------------" << std::endl;
}

}  // namespace Shadow
