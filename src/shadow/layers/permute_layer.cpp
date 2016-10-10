#include "shadow/layers/permute_layer.hpp"
#include "shadow/util/image.hpp"

void PermuteLayer::Setup(VecBlob *blobs) {
  Blob<float> *bottom = find_blob_by_name(*blobs, layer_param_.bottom(0));
  if (bottom == nullptr)
    Fatal("Layer: " + layer_param_.name() + ", bottom " +
          layer_param_.bottom(0) + " not exist!");

  Blob<float> *top = new Blob<float>(layer_param_.top(0));

  int order_size = layer_param_.permute_param().order_size();
  if (order_size != bottom->num_axes())
    Fatal("Order size must be the same as bottom!");

  int *permute_order = new int[order_size], *old_steps = new int[order_size],
      *new_steps = new int[order_size];
  for (int i = 0; i < order_size; ++i) {
    permute_order[i] = layer_param_.permute_param().order(i);
    top->add_shape(bottom->shape(permute_order[i]));
  }

  for (int i = 0; i < order_size; ++i) {
    if (i == order_size - 1) {
      old_steps[i] = 1;
      new_steps[i] = 1;
    } else {
      old_steps[i] = bottom->count(i + 1);
      new_steps[i] = top->count(i + 1);
    }
  }

  permute_order_ = new Blob<int>(order_size, permute_order);
  old_steps_ = new Blob<int>(order_size, old_steps);
  new_steps_ = new Blob<int>(order_size, new_steps);

  top->allocate_data(top->count());

  bottom_.push_back(bottom);
  top_.push_back(top);

  blobs->push_back(top);

  std::stringstream out;
  out << layer_param_.name() << ": ("
      << Util::format_vector(bottom->shape(), ",") << ") -> ("
      << Util::format_vector(top->shape(), ",") << ")";
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

  // std::cout << "Free DropoutLayer!" << std::endl;
}
