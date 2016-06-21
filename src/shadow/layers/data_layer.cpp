#include "shadow/layers/data_layer.hpp"
#include "shadow/util/image.hpp"

void DataLayer::Setup(VecBlob *blobs) {
  Blob *bottom = find_blob_by_name(*blobs, "in_blob");
  if (bottom == nullptr)
    Fatal("Layer: " + layer_param_.name() + ", bottom " +
          layer_param_.bottom(0) + " not exist!");

  Blob *top = new Blob(layer_param_.top(0));

  if (!(bottom->shape(1) && bottom->shape(2) && bottom->shape(3)))
    Fatal("Channel, height and width must greater than zero.");

  scale_ = layer_param_.data_param().scale();
  mean_value_ = layer_param_.data_param().mean_value();

  int in_c = bottom->shape(1), in_h = bottom->shape(2), in_w = bottom->shape(3);

  *top->mutable_shape() = bottom->shape();
  top->allocate_data(top->count());

  bottom_.push_back(bottom);
  top_.push_back(top);

  blobs->push_back(top);

#if defined(VERBOSE)
  printf("Data Layer: %d x %d x %d input\n", in_c, in_h, in_w);
#endif
}

void DataLayer::Forward() {
  const Blob *bottom = bottom_.at(0);
  Blob *top = top_.at(0);

  Image::DataTransform(top->count(), bottom->data(), scale_, mean_value_,
                       top->mutable_data());
}

void DataLayer::Release() {
  bottom_.clear();
  top_.clear();

  // std::cout << "Free DataLayer!" << std::endl;
}
