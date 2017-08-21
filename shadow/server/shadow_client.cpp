#include <grpc++/grpc++.h>

#include "server/server.grpc.pb.h"
#include "util/log.hpp"
#include "util/util.hpp"

namespace Shadow {

namespace Client {

class ShadowClient {
 public:
  ShadowClient(const std::string& method,
               std::shared_ptr<grpc::Channel> channel)
      : stub_(shadow::ShadowService::NewStub(channel)) {
    if (method == "ssd" || method == "yolo") {
      is_detection_ = true;
    } else if (method == "classification") {
      is_detection_ = false;
    } else {
      LOG(FATAL) << "Unknown method " << method;
    }
    request_.set_server_name(method);
  }

  void Process(const std::string& file_path) {
    Timer timer;
    request_.set_file_path(file_path);
    const auto& status = stub_->Process(&context_, request_, &reply_);
    double time_cost = timer.get_millisecond();

    if (status.ok()) {
      std::cout << "Predicted in " << time_cost << " ms" << std::endl;
      if (is_detection_) {
        for (const auto& rect : reply_.objects()) {
          std::cout << "xmin = " << rect.xmin() << ", ymin = " << rect.ymin()
                    << ", xmax = " << rect.xmax() << ", ymax = " << rect.ymax()
                    << ", label = " << rect.label()
                    << ", score = " << rect.score() << std::endl;
        }
      } else {
        for (const auto& sub_task : reply_.tasks()) {
          std::cout << sub_task.task_name() << ": ";
          std::vector<float> scores;
          for (const auto score : sub_task.task_score()) {
            scores.push_back(score);
          }
          const auto& max_k = Util::top_k(scores, 1);
          for (const auto idx : max_k) {
            std::cout << idx << " (" << scores[idx] << ") ";
          }
          std::cout << std::endl;
        }
      }
    } else {
      std::cout << status.error_code() << ": " << status.error_message() << ", "
                << status.error_details() << std::endl;
    }
  }

 private:
  std::unique_ptr<shadow::ShadowService::Stub> stub_;
  grpc::ClientContext context_;
  shadow::ShadowRequest request_;
  shadow::ShadowReply reply_;
  bool is_detection_ = true;
};

}  // namespace Client

}  // namespace Shadow

int main(int argc, char** argv) {
  std::string server_address("localhost:50051");
  std::string image_path("data/static/demo_6.png");

  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());

  Shadow::Client::ShadowClient client("ssd", channel);

  client.Process(image_path);

  return 0;
}
