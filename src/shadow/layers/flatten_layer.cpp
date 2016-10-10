#include "shadow/layers/flatten_layer.hpp"

void FlattenLayer::Setup(VecBlob *blobs) {
  Blob<float> *bottom = find_blob_by_name(*blobs, layer_param_.bottom(0));
  if (bottom != nullptr) {
    if (bottom->num() && bottom->num_axes() > 1) {
      bottom_.push_back(bottom);
    } else {
      Fatal(layer_name_ + ": bottom blob(" + layer_param_.bottom(0) +
            Util::format_vector(bottom->shape(), ",", "(", ")") +
            ") dimension mismatch!");
    }
  } else {
    Fatal(layer_name_ + ": bottom blob(" + layer_param_.bottom(0) +
          ") not exist!");
  }

  for (int i = 0; i < layer_param_.top_size(); ++i) {
    Blob<float> *top = new Blob<float>(layer_param_.top(i));
    top_.push_back(top);
    blobs->push_back(top);
  }

  start_axis_ = layer_param_.flatten_param().start_axis();
  end_axis_ = layer_param_.flatten_param().end_axis();
  if (end_axis_ == -1) end_axis_ = bottom->num_axes() - 1;

  if (start_axis_ < 0 || end_axis_ >= bottom->num_axes() ||
      start_axis_ > end_axis_) {
    Fatal("Axes period out of range!");
  }
}

void FlattenLayer::Reshape() {
  const Blob<float> *bottom = bottom_.at(0);
  Blob<float> *top = top_.at(0);

  for (int i = 0; i < start_axis_; ++i) {
    top->add_shape(bottom->shape(i));
  }
  top->add_shape(bottom->count(start_axis_, end_axis_));
  for (int i = end_axis_ + 1; i < bottom->num_axes(); ++i) {
    top->add_shape(bottom->shape(i));
  }

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
      << Util::format_vector(top->shape(), ",", "(", ")");
  DInfo(out.str());
}

void FlattenLayer::Forward() {
  Blob<float> *bottom = bottom_.at(0);
  Blob<float> *top = top_.at(0);

  top->share_data(bottom->mutable_data());
}

void FlattenLayer::Release() {
  bottom_.clear();
  top_.clear();

  // std::cout << "Free DropoutLayer!" << std::endl;
}
