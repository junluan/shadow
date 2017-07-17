#include <grpc++/grpc++.h>

#include "examples/demo_classification.hpp"
#include "examples/demo_detection.hpp"
#include "server/server.grpc.pb.h"
#include "util/jimage_proc.hpp"

namespace Shadow {

namespace Server {

class ShadowServer final : public shadow::ShadowService::Service {
 public:
  ShadowServer(const std::string &model, const VecInt &classes,
               const std::string &method) {
    if (!method.compare("ssd") || !method.compare("yolo")) {
      detection_ = new DemoDetection(method);
      detection_->Setup(model, classes, 1);
      is_detection_ = true;
    } else if (!method.compare("classification")) {
      classification_ = new DemoClassification(method);
      classification_->Setup(model, classes, 1);
      is_detection_ = false;
    } else {
      LOG(FATAL) << "Unknown method " << method;
    }
    method_name_ = method;
  }
  ~ShadowServer() {
    if (detection_ != nullptr) {
      delete detection_;
      detection_ = nullptr;
    }
    if (classification_ != nullptr) {
      delete classification_;
      classification_ = nullptr;
    }
  }

  grpc::Status Process(grpc::ServerContext *context,
                       const shadow::ShadowRequest *request,
                       shadow::ShadowReply *reply) override {
    const auto &request_name = request->server_name();
    if (request_name.compare(method_name_)) {
      std::stringstream ss;
      ss << "Current server method name: " << method_name_
         << ", request method name: " << request_name;
      LOG(WARNING) << ss.str();
      grpc::Status status(grpc::INVALID_ARGUMENT, "Unsupported method",
                          ss.str());
      return status;
    }
    Timer timer;
    const auto &image_path = request->file_path();
    std::cout << "Processing " << image_path << ", ";
    im_ini_.Read(image_path);
    VecRectF rois{RectF(0, 0, im_ini_.w_, im_ini_.h_)};
    if (is_detection_) {
      detection_->Predict(im_ini_, rois, &Bboxes_);
      Boxes::Amend(&Bboxes_, rois);
      const auto &boxes = Boxes::NMS(Bboxes_, 0.5);
      reply->clear_objects();
      for (const auto &boxF : boxes) {
        const BoxI box(boxF);
        int color_r = (box.label * 100) % 255;
        int color_g = (color_r + 100) % 255;
        int color_b = (color_g + 100) % 255;
        Scalar scalar(color_r, color_g, color_b);
        JImageProc::Rectangle(&im_ini_, box.RectInt(), scalar);
        auto object = reply->add_objects();
        object->set_xmin(box.xmin);
        object->set_ymin(box.ymin);
        object->set_xmax(box.xmax);
        object->set_ymax(box.ymax);
        object->set_label(box.label);
        object->set_score(box.score);
      }
      const auto &out_file =
          Util::find_replace_last(image_path, ".", "-result.");
      im_ini_.Write(out_file);
    } else {
      classification_->Predict(im_ini_, rois, &scores_);
      reply->clear_tasks();
      for (const auto &score_map : scores_) {
        for (const auto &it : score_map) {
          auto sub_task = reply->add_tasks();
          sub_task->set_task_name(it.first);
          for (const auto score : it.second) {
            sub_task->add_task_score(score);
          }
        }
      }
    }
    std::cout << "Processed in " << timer.get_millisecond() << " ms"
              << std::endl;
    return grpc::Status::OK;
  }

 private:
  DemoDetection *detection_ = nullptr;
  DemoClassification *classification_ = nullptr;
  std::string method_name_;
  bool is_detection_ = true;
  JImage im_ini_;
  std::vector<VecBoxF> Bboxes_;
  std::vector<std::map<std::string, VecFloat>> scores_;
};

}  // namespace Server

}  // namespace Shadow

int main(int argc, char **argv) {
  std::string server_address("localhost:50051");
  std::string model("models/ssd/adas/adas_model_finetune_reduce_3.shadowmodel");

  Shadow::Server::ShadowServer service(model, {3}, "ssd");

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

  std::cout << "Server listening on " << server_address << std::endl;

  server->Wait();

  return 0;
}
