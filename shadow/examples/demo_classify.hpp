#ifndef SHADOW_EXAMPLES_DEMO_CLASSIFY_HPP_
#define SHADOW_EXAMPLES_DEMO_CLASSIFY_HPP_

#include "algorithm/method.hpp"

#include <memory>

namespace Shadow {

class DemoClassify {
 public:
  explicit DemoClassify(const std::string& method_name);
  ~DemoClassify() { LOG(INFO) << profiler_.get_stats_str(); }

  void Setup(const std::string& model_file) { method_->Setup(model_file); }

  void Test(const std::string& image_file);
  void BatchTest(const std::string& list_file);

 private:
  static void PrintConsole(const std::map<std::string, VecFloat>& scores,
                           int top_k, bool split = false);

  static void PrintStream(const std::string& im_name,
                          const std::map<std::string, VecFloat>& scores,
                          int top_k, std::ostream* os);

  Profiler profiler_;
  std::shared_ptr<Method> method_ = nullptr;
  std::map<std::string, VecFloat> scores_;
};

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_DEMO_CLASSIFY_HPP_
