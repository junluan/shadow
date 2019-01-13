#include <grpc++/grpc++.h>

#include "service.grpc.pb.h"

#include "examples/classify.hpp"
#include "examples/detect_faster_rcnn.hpp"
#include "examples/detect_mtcnn.hpp"
#include "examples/detect_refinedet.hpp"
#include "examples/detect_ssd.hpp"
#include "examples/detect_yolo.hpp"

#include <memory>

namespace Shadow {

class Server final : public shadow::Inference::Service {
 public:
  grpc::Status Setup(grpc::ServerContext *context,
                     const shadow::SetupParam *setup_param,
                     shadow::Response *response) override {
    is_detection_ = true;
    method_name_ = setup_param->method_name();
    if (method_name_ == "classify") {
      method_ = std::make_shared<Classify>();
      is_detection_ = false;
    } else if (method_name_ == "faster") {
      method_ = std::make_shared<DetectFasterRCNN>();
    } else if (method_name_ == "mtcnn") {
      method_ = std::make_shared<DetectMTCNN>();
    } else if (method_name_ == "ssd") {
      method_ = std::make_shared<DetectSSD>();
    } else if (method_name_ == "refinedet") {
      method_ = std::make_shared<DetectRefineDet>();
    } else if (method_name_ == "yolo") {
      method_ = std::make_shared<DetectYOLO>();
    } else {
      LOG(FATAL) << "Unknown method " << method_name_;
    }

    method_->Setup(setup_param->model_file());

    return grpc::Status::OK;
  }

  grpc::Status Predict(grpc::ServerContext *context,
                       const shadow::RequestParam *request_param,
                       shadow::Response *response) override {
    const auto &method_name = request_param->method_name();
    const auto &file_path = request_param->file_path();

    if (method_name != method_name_) {
      std::stringstream ss;
      ss << "Current server method: " << method_name_
         << ", request method: " << method_name;
      LOG(WARNING) << ss.str();
      grpc::Status status(grpc::INVALID_ARGUMENT, "Unsupported method",
                          ss.str());
      return status;
    }

    LOG(INFO) << "Processing " << file_path << " ... ";

    timer_.start();
    im_ini_.Read(file_path);
    RectF roi(0, 0, im_ini_.w_, im_ini_.h_);
    if (is_detection_) {
      VecBoxF boxes;
      std::vector<VecPointF> Gpoints;
      method_->Predict(im_ini_, roi, &boxes, &Gpoints);
      boxes = Boxes::NMS(boxes, 0.5);
      response->clear_objects();
      for (const auto &box : boxes) {
        auto *object = response->add_objects();
        object->set_xmin(box.xmin);
        object->set_ymin(box.ymin);
        object->set_xmax(box.xmax);
        object->set_ymax(box.ymax);
        object->set_label(box.label);
        object->set_score(box.score);
      }
    } else {
      std::map<std::string, VecFloat> scores;
      method_->Predict(im_ini_, roi, &scores);
      response->clear_tasks();
      for (const auto &it : scores) {
        auto *task = response->add_tasks();
        task->set_name(it.first);
        for (auto val : it.second) {
          task->add_values(val);
        }
      }
    }

    LOG(INFO) << "Processed in " << timer_.get_millisecond() << " ms";

    return grpc::Status::OK;
  }

 private:
  bool is_detection_ = true;
  std::string method_name_;
  std::shared_ptr<Method> method_ = nullptr;
  JImage im_ini_;
  Timer timer_;
};

}  // namespace Shadow

int main(int argc, char **argv) {
  std::string server_address("localhost:50051");

  Shadow::Server service;

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::shared_ptr<grpc::Server> server(builder.BuildAndStart());

  std::cout << "Server listening on " << server_address << std::endl;

  server->Wait();

  return 0;
}
