#include "shadow/layers/data_layer.hpp"
#include "shadow/util/image.hpp"

void DataLayer::Setup(VecBlob *blobs) {
  Blob<float> *bottom = get_blob_by_name(*blobs, "in_blob");
  if (bottom != nullptr) {
    if (bottom->num() && bottom->num_axes() == 4) {
      bottoms_.push_back(bottom);
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
    tops_.push_back(top);
    blobs->push_back(top);
  }

  const shadow::DataParameter &data_param = layer_param_.data_param();

  scale_ = data_param.scale();
  mean_value_ = data_param.mean_value();
}

void DataLayer::Reshape() {
  tops_[0]->reshape(bottoms_[0]->shape());

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")");
  DInfo(out.str());
}

void DataLayer::Forward() {
  Image::DataTransform(bottoms_[0]->data(), tops_[0]->count(), scale_,
                       mean_value_, tops_[0]->mutable_data());
}

void DataLayer::Release() {
  bottoms_.clear();
  tops_.clear();

  // DInfo("Free DataLayer!");
}
