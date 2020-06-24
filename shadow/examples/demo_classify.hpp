#ifndef SHADOW_EXAMPLES_DEMO_CLASSIFY_HPP_
#define SHADOW_EXAMPLES_DEMO_CLASSIFY_HPP_

#include "algorithm/method.hpp"

#include <memory>

namespace Shadow {

class DemoClassify {
 public:
  explicit DemoClassify(const std::string& method_name);

  void Setup(const std::string& model_file) { method_->Setup(model_file); }

  void Test(const std::string& image_file);
  void BatchTest(const std::string& list_file);

 private:
  void PrintConsole(const std::map<std::string, VecFloat>& scores, int top_k,
                    bool split = false);

  void PrintStream(const std::string& im_name,
                   const std::map<std::string, VecFloat>& scores, int top_k,
                   std::ostream* os);

  Timer timer_;
  JImage im_ini_;
  std::shared_ptr<Method> method_ = nullptr;
  std::map<std::string, VecFloat> scores_;
};

}  // namespace Shadow

#endif  // SHADOW_EXAMPLES_DEMO_CLASSIFY_HPP_
