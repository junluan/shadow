#if !defined(__linux)
#include <sdkddkver.h>
#endif
#include <grpc++/grpc++.h>

#include "examples/demo.hpp"
#include "server/server.grpc.pb.h"
#include "util/jimage_proc.hpp"

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
  ~ADASService() {
    if (demo_ != nullptr) {
      delete demo_;
      demo_ = nullptr;
    }
  }

  Status Process(ServerContext *context, const shadow::ADASRequest *request,
                 shadow::ADASReply *reply) override {
    Timer timer;
    const auto &image_path = request->file_path();
    std::cout << "Processing " << image_path << ", ";
    im_ini_.Read(image_path);
    VecRectF rois{RectF(0, 0, im_ini_.w_, im_ini_.h_)};
    demo_->Predict(im_ini_, rois, &Bboxes_);
    Shadow::Boxes::Amend(&Bboxes_, rois);
    const auto &boxes = Shadow::Boxes::NMS(Bboxes_, 0.5);
    reply->clear_objects();
    for (const auto &boxF : boxes) {
      const Shadow::BoxI box(boxF);
      int color_r = (box.label * 100) % 255;
      int color_g = (color_r + 100) % 255;
      int color_b = (color_g + 100) % 255;
      Scalar scalar(color_r, color_g, color_b);
      Shadow::JImageProc::Rectangle(&im_ini_, box.RectInt(), scalar);
      auto object = reply->add_objects();
      object->set_xmin(box.xmin);
      object->set_ymin(box.ymin);
      object->set_xmax(box.xmax);
      object->set_ymax(box.ymax);
      object->set_label(box.label);
      object->set_score(box.score);
    }
    const auto &out_file = Util::find_replace_last(image_path, ".", "-result.");
    im_ini_.Write(out_file);
    std::cout << "Processed in " << timer.get_millisecond() << " ms"
              << std::endl;
    return Status::OK;
  }

 private:
  Shadow::Demo *demo_;
  Shadow::JImage im_ini_;
  std::vector<Shadow::VecBoxF> Bboxes_;
};

int main(int argc, char **argv) {
  std::string server_address("localhost:50051");
  std::string model("D:/Workspace/datasets/models/ssd/adas/adas_model_finetune_reduce_3.shadowmodel");

  ADASService service(model);

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());

  std::cout << "Server listening on " << server_address << std::endl;

  server->Wait();

  return 0;
}
