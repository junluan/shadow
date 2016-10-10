#include "shadow/layers/conv_layer.hpp"
#include "shadow/util/activations.hpp"
#include "shadow/util/blas.hpp"
#include "shadow/util/image.hpp"

inline int convolutional_out_size(int s, int size, int pad, int stride) {
  return (s + 2 * pad - size) / stride + 1;
}

void ConvLayer::Setup(VecBlob *blobs) {
  Blob<float> *bottom = find_blob_by_name(*blobs, layer_param_.bottom(0));
  if (bottom != nullptr) {
    if (bottom->num() && bottom->num_axes() == 4) {
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

  num_output_ = layer_param_.convolution_param().num_output();
  kernel_size_ = layer_param_.convolution_param().kernel_size();
  stride_ = layer_param_.convolution_param().stride();
  pad_ = layer_param_.convolution_param().pad();
  activate_ = layer_param_.convolution_param().activate();
}

void ConvLayer::Reshape() {
  const Blob<float> *bottom = bottom_.at(0);
  Blob<float> *top = top_.at(0);

  int in_c = bottom->shape(1), in_h = bottom->shape(2), in_w = bottom->shape(3);
  int out_c = num_output_;
  int out_h = convolutional_out_size(in_h, kernel_size_, pad_, stride_);
  int out_w = convolutional_out_size(in_w, kernel_size_, pad_, stride_);

  *top->mutable_shape() = bottom->shape();
  top->set_shape(1, out_c);
  top->set_shape(2, out_h);
  top->set_shape(3, out_w);
  top->allocate_data(top->count());

  out_map_size_ = out_h * out_w;
  kernel_num_ = kernel_size_ * kernel_size_ * in_c;

  filters_ = new Blob<float>(kernel_num_ * out_c, layer_name_ + " filters");
  biases_ = new Blob<float>(out_c, layer_name_ + " biases");
  col_image_ =
      new Blob<float>(out_map_size_ * kernel_num_, layer_name_ + " col_image");

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> " << out_c
      << "_" << kernel_size_ << "x" << kernel_size_ << "_s" << stride_ << "_p"
      << pad_ << " -> " << Util::format_vector(top->shape(), ",", "(", ")");
  DInfo(out.str());
}

void ConvLayer::Forward() {
  const Blob<float> *bottom = bottom_.at(0);
  Blob<float> *top = top_.at(0);

  int batch = bottom->shape(0);
  int out_c = top->shape(1);
  for (int b = 0; b < batch; ++b) {
    Blas::SetArrayRepeat(out_map_size_, biases_->data(), out_c,
                         top->mutable_data(), b * top->num());
  }
  for (int b = 0; b < batch; ++b) {
    Image::Im2Col(bottom->shape(), bottom->data(), b * bottom->num(),
                  kernel_size_, stride_, pad_, top->shape(),
                  col_image_->mutable_data());
    Blas::BlasSGemm(0, 0, out_c, out_map_size_, kernel_num_, 1,
                    filters_->data(), kernel_num_, col_image_->data(),
                    out_map_size_, 1, top->mutable_data(), b * top->num(),
                    out_map_size_);
  }
  Activations::ActivateArray(top->count(), activate_, top->mutable_data());
}

void ConvLayer::Release() {
  bottom_.clear();
  top_.clear();

  filters_->clear();
  biases_->clear();
  col_image_->clear();

  // std::cout << "Free ConvLayer!" << std::endl;
}
