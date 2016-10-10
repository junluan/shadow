#include "shadow/layers/data_layer.hpp"
#include "shadow/util/image.hpp"

void DataLayer::Setup(VecBlob *blobs) {
  Blob<float> *bottom = find_blob_by_name(*blobs, "in_blob");
  if (bottom != nullptr) {
    if (bottom->num() && bottom->num_axes() == 4) {
      bottom_.push_back(bottom);
    } else {
      Fatal(layer_name_ + ": bottom blob(" + "in_blob" +
            Util::format_vector(bottom->shape(), ",", "(", ")") +
            ") dimension mismatch!");
    }
  } else {
    Fatal(layer_name_ + ": bottom blob(" + "in_blob" + ") not exist!");
  }

  for (int i = 0; i < layer_param_.top_size(); ++i) {
    Blob<float> *top = new Blob<float>(layer_param_.top(i));
    top_.push_back(top);
    blobs->push_back(top);
  }

  scale_ = layer_param_.data_param().scale();
  mean_value_ = layer_param_.data_param().mean_value();
}

void DataLayer::Reshape() {
  const Blob<float> *bottom = bottom_.at(0);
  Blob<float> *top = top_.at(0);

  *top->mutable_shape() = bottom->shape();
  top->allocate_data(top->count());

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom->shape(), ",", "(", ")");
  DInfo(out.str());
}

void DataLayer::Forward() {
  const Blob<float> *bottom = bottom_.at(0);
  Blob<float> *top = top_.at(0);

  Image::DataTransform(top->count(), bottom->data(), scale_, mean_value_,
                       top->mutable_data());
}

void DataLayer::Release() {
  bottom_.clear();
  top_.clear();

  // std::cout << "Free DataLayer!" << std::endl;
}
