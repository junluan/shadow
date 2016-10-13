#include "shadow/layers/data_layer.hpp"
#include "shadow/util/image.hpp"

void DataLayer::Setup(VecBlob *blobs) {
  Blob<float> *bottom = get_blob_by_name(*blobs, "in_blob");
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
}

void DataLayer::Reshape() {
  scale_ = layer_param_.data_param().scale();
  mean_value_ = layer_param_.data_param().mean_value();

  top_[0]->set_shape(bottom_[0]->shape());
  top_[0]->allocate_data(top_[0]->count());

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom_[0]->shape(), ",", "(", ")");
  DInfo(out.str());
}

void DataLayer::Forward() {
  Image::DataTransform(bottom_[0]->data(), top_[0]->count(), scale_,
                       mean_value_, top_[0]->mutable_data());
}

void DataLayer::Release() {
  bottom_.clear();
  top_.clear();

  // DInfo("Free DataLayer!");
}
