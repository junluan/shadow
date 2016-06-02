#include "shadow/layers/data_layer.hpp"
#include "shadow/util/image.hpp"

DataLayer::DataLayer(shadow::LayerParameter layer_param) {
  layer_param_ = layer_param;
  in_blob_ = new Blob<BType>();
  out_blob_ = new Blob<BType>();
}
DataLayer::~DataLayer() { ReleaseLayer(); }

void DataLayer::MakeLayer(Blob<BType> *blob) {
  if (!(blob->shape(1) && blob->shape(2) && blob->shape(3)))
    Fatal("Channel, height and width must greater than zero.");

  scale_ = layer_param_.data_param().scale();
  mean_value_ = layer_param_.data_param().mean_value();

  int batch = blob->shape(0);
  int in_c = blob->shape(1), in_h = blob->shape(2), in_w = blob->shape(3);

  int num = in_c * in_h * in_w;

  *in_blob_->mutable_shape() = blob->shape();
  *out_blob_->mutable_shape() = blob->shape();

  in_blob_->set_num(num);
  out_blob_->set_num(num);
  in_blob_->set_count(batch * num);
  out_blob_->set_count(batch * num);

  in_blob_->allocate_data(in_blob_->count());
  out_blob_->allocate_data(out_blob_->count());

#if defined(VERBOSE)
  printf("Data Layer: %d x %d x %d input\n", in_c, in_h, in_w);
#endif
}

void DataLayer::ForwardLayer(float *in_data) {
  in_blob_->set_data(in_data);
  Image::DataTransform(out_blob_->count(), in_blob_->data(), scale_,
                       mean_value_, out_blob_->mutable_data());
}

void DataLayer::ReleaseLayer() {
  in_blob_->clear();
  out_blob_->clear();
  // std::cout << "Free DataLayer!" << std::endl;
}
