#include <grpc++/grpc++.h>

#include "service.grpc.pb.h"

#include "util/log.hpp"
#include "util/util.hpp"

#include <memory>

namespace Shadow {

class Client {
 public:
  Client(const std::shared_ptr<grpc::Channel>& channel)
      : stub_(shadow::Inference::NewStub(channel)) {}

  void Setup(const std::string& method_name, const std::string& model_file) {
    method_name_ = method_name;
    if (method_name_ == "mtcnn" || method_name_ == "ssd" ||
        method_name_ == "refinedet") {
      is_detection_ = true;
    } else if (method_name_ == "classification") {
      is_detection_ = false;
    } else {
      LOG(FATAL) << "Unknown method " << method_name_;
    }

    shadow::SetupParam setup_param;
    setup_param.set_model_file(model_file);
    setup_param.set_method_name(method_name_);

    grpc::ClientContext context;
    const auto& status = stub_->Setup(&context, setup_param, &response_);
    CHECK(status.ok());
  }

  void Predict(const std::string& file_path) {
    shadow::RequestParam request_param;
    request_param.set_method_name(method_name_);
    request_param.set_file_path(file_path);

    timer_.start();
    grpc::ClientContext context;
    const auto& status = stub_->Predict(&context, request_param, &response_);
    LOG(INFO) << "Predicted in " << timer_.get_millisecond() << " ms";

    if (status.ok()) {
      if (is_detection_) {
        for (const auto& object : response_.objects()) {
          LOG(INFO) << "xmin = " << object.xmin()
                    << ", ymin = " << object.ymin()
                    << ", xmax = " << object.xmax()
                    << ", ymax = " << object.ymax()
                    << ", label = " << object.label()
                    << ", score = " << object.score();
        }
      } else {
        for (const auto& task : response_.tasks()) {
          std::vector<float> values;
          for (auto val : task.values()) {
            values.push_back(val);
          }
          const auto& max_k = Util::top_k(values, 1);
          std::stringstream ss;
          ss << task.name() << ": ";
          for (const auto idx : max_k) {
            ss << idx << " (" << values[idx] << ") ";
          }
          LOG(INFO) << ss.str();
        }
      }
    } else {
      LOG(INFO) << status.error_code() << ": " << status.error_message() << ", "
                << status.error_details();
    }
  }

 private:
  std::unique_ptr<shadow::Inference::Stub> stub_;
  shadow::Response response_;
  bool is_detection_ = true;
  std::string method_name_;
  Timer timer_;
};

}  // namespace Shadow

int main(int argc, char** argv) {
  std::string server_address("localhost:50051");
  std::string model("models/classify/squeezenet_v1.1_merged.shadowmodel");
  std::string test_image("data/static/cat.jpg");

  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());

  Shadow::Client client(channel);

  client.Setup("classification", model);
  client.Predict(test_image);

  return 0;
}
