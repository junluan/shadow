#include "shadow/layers/connected_layer.hpp"
#include "shadow/util/activations.hpp"
#include "shadow/util/blas.hpp"

ConnectedLayer::ConnectedLayer(shadow::LayerParameter layer_param) {
  layer_param_ = layer_param;
  in_blob_ = new Blob<BType>();
  out_blob_ = new Blob<BType>();
  weights_ = new Blob<BType>();
  biases_ = new Blob<BType>();
}
ConnectedLayer::~ConnectedLayer() { ReleaseLayer(); }

void ConnectedLayer::MakeLayer(Blob<BType> *blob) {
  if (!(blob->shape(1) && blob->shape(2) && blob->shape(3)))
    Fatal("Channel, height and width must greater than zero.");

  num_output_ = layer_param_.connected_param().num_output();
  activate_ = layer_param_.connected_param().activate();

  int batch = blob->shape(0);

  int in_num = blob->shape(1) * blob->shape(2) * blob->shape(3);
  int out_num = num_output_;

  *in_blob_->mutable_shape() = blob->shape();
  blob->set_shape(1, num_output_);
  blob->set_shape(2, 1);
  blob->set_shape(3, 1);
  *out_blob_->mutable_shape() = blob->shape();

  in_blob_->set_num(in_num);
  out_blob_->set_num(out_num);
  in_blob_->set_count(batch * in_num);
  out_blob_->set_count(batch * out_num);

  out_data_ = new float[out_blob_->count()];
  out_blob_->allocate_data(out_blob_->count());

  weights_->allocate_data(in_num * out_num);
  biases_->allocate_data(out_num);

#if defined(VERBOSE)
  printf("Connected Layer: %d input, %d output\n", in_num, out_num);
#endif
}

void ConnectedLayer::ForwardLayer() {
  int batch = in_blob_->shape(0);
  float *out_data = out_blob_->mutable_data();

#if !defined(USE_CUDA) & !defined(USE_CL)
  for (int b = 0; b < batch; ++b) {
    Blas::BlasCopy(out_blob_->num(), biases_->data(), 1,
                   out_data + b * out_blob_->num(), 1);
  }
  Blas::BlasSGemm(0, 0, batch, out_blob_->num(), in_blob_->num(), 1,
                  in_blob_->data(), in_blob_->num(), weights_->data(),
                  out_blob_->num(), 1, out_data, 0, out_blob_->num());
  Activations::ActivateArray(out_blob_->count(), activate_, out_data);
#else
  Kernel::BiasOutput(biases_->data(), batch, out_blob_->num(), 1, out_data);
  Blas::BlasSGemm(0, 0, batch, out_blob_->num(), in_blob_->num(), 1,
                  in_blob_->data(), in_blob_->num(), weights_->data(),
                  out_blob_->num(), 1, out_data, 0, out_blob_->num());
  Kernel::ActivateArray(out_blob_->count(), activate_, out_data);
#endif
}

void ConnectedLayer::ReleaseLayer() {
  out_blob_->clear();

  weights_->clear();
  biases_->clear();
  // std::cout << "Free ConnectedLayer!" << std::endl;
}
