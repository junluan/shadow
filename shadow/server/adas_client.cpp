#include <grpc++/grpc++.h>

#include "server/server.grpc.pb.h"
#include "util/util.hpp"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

class ADASClient {
 public:
  ADASClient(std::shared_ptr<Channel> channel)
      : stub_(shadow::ADAS::NewStub(channel)) {}

  void Process(const std::string& file_path) {
    Timer timer;
    request_.set_file_path(file_path);
    const auto& status = stub_->Process(&context_, request_, &reply_);
    double time_cost = timer.get_millisecond();

    if (status.ok()) {
      std::cout << "Predicted in " << time_cost << " ms" << std::endl;
      for (const auto& rect : reply_.objects()) {
        std::cout << "xmin = " << rect.xmin() << ", ymin = " << rect.ymin()
                  << ", xmax = " << rect.xmax() << ", ymax = " << rect.ymax()
                  << ", label = " << rect.label()
                  << ", score = " << rect.score() << std::endl;
      }
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
    }
  }

 private:
  std::unique_ptr<shadow::ADAS::Stub> stub_;
  ClientContext context_;
  shadow::ADASRequest request_;
  shadow::ADASReply reply_;
};

int main(int argc, char** argv) {
  std::string server_address("localhost:50051");
  std::string image_path("data/static/demo_6.png");

  ADASClient client(
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials()));
  client.Process(image_path);

  return 0;
}
