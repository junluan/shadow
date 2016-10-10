#include "shadow/layers/permute_layer.hpp"
#include "shadow/util/image.hpp"

void PermuteLayer::Setup(VecBlob *blobs) {
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

  num_axes_ = layer_param_.permute_param().order_size();
  if (num_axes_ != bottom->num_axes()) {
    Fatal("Number of axes mismatch!");
  }
}

void PermuteLayer::Reshape() {
  const Blob<float> *bottom = bottom_.at(0);
  Blob<float> *top = top_.at(0);

  int *permute_order = new int[num_axes_], *old_steps = new int[num_axes_],
      *new_steps = new int[num_axes_];
  for (int i = 0; i < num_axes_; ++i) {
    permute_order[i] = layer_param_.permute_param().order(i);
    top->add_shape(bottom->shape(permute_order[i]));
  }
  top->allocate_data(top->count());

  for (int i = 0; i < num_axes_; ++i) {
    if (i == num_axes_ - 1) {
      old_steps[i] = 1;
      new_steps[i] = 1;
    } else {
      old_steps[i] = bottom->count(i + 1);
      new_steps[i] = top->count(i + 1);
    }
  }

  permute_order_ =
      new Blob<int>(num_axes_, permute_order, layer_name_ + " permute_order");
  old_steps_ = new Blob<int>(num_axes_, old_steps, layer_name_ + " old_steps");
  new_steps_ = new Blob<int>(num_axes_, new_steps, layer_name_ + " new_steps");

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
      << Util::format_vector(top->shape(), ",", "(", ")");
  DInfo(out.str());
}

void PermuteLayer::Forward() {
  const Blob<float> *bottom = bottom_.at(0);
  Blob<float> *top = top_.at(0);

  Image::Permute(bottom->data(), bottom->count(), bottom->num_axes(),
                 permute_order_->data(), old_steps_->data(), new_steps_->data(),
                 top->mutable_data());
}

void PermuteLayer::Release() {
  bottom_.clear();
  top_.clear();

  permute_order_->clear();
  old_steps_->clear();
  new_steps_->clear();

  // std::cout << "Free DropoutLayer!" << std::endl;
}
