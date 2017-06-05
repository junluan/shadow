#include <grpc++/grpc++.h>

#include "examples/demo.hpp"
#include "server/server.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

class ADASService final : public shadow::ADAS::Service {
 public:
  ADASService(const std::string &model, const std::string &method = "ssd") {
    demo_ = new Shadow::Demo(method);
    demo_->Setup(model, 3, 1);
  }
  ~ADASService() { delete demo_; }

  Status Process(ServerContext *context, const shadow::ADASRequest *request,
                 shadow::ADASReply *reply) override {
    const auto &image_path = request->file_path();
    std::cout << "Processing " << image_path << ", ";
    Timer timer;
    demo_->Test(image_path, reply);
    std::cout << "Processed in " << timer.get_millisecond() << " ms"
              << std::endl;
    return Status::OK;
  }

 private:
  Shadow::Demo *demo_;
};

int main(int argc, char **argv) {
  std::string server_address("localhost:50051");
  std::string model("models/ssd/adas/adas_model_finetune_reduce_3.shadowmodel");

  ADASService service(model);

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());

  std::cout << "Server listening on " << server_address << std::endl;

  server->Wait();

  return 0;
}
