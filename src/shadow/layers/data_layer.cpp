#include "shadow/layers/data_layer.hpp"
#include "shadow/util/image.hpp"

void DataLayer::Setup(VecBlob *blobs) {
  Blob<float> *bottom = find_blob_by_name(*blobs, "in_blob");
  if (bottom == nullptr)
    Fatal("Layer: " + layer_param_.name() + ", bottom " +
          layer_param_.bottom(0) + " not exist!");

  Blob<float> *top = new Blob<float>(layer_param_.top(0));

  if (!(bottom->shape(1) && bottom->shape(2) && bottom->shape(3)))
    Fatal("Channel, height and width must greater than zero.");

  scale_ = layer_param_.data_param().scale();
  mean_value_ = layer_param_.data_param().mean_value();

  *top->mutable_shape() = bottom->shape();
  top->allocate_data(top->count());

  bottom_.push_back(bottom);
  top_.push_back(top);

  blobs->push_back(top);

  std::stringstream out;
  out << layer_param_.name() << ": ("
      << Util::format_vector(bottom->shape(), ",") << ")";
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
