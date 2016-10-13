#include "shadow/layers/activate_layer.hpp"
#include "shadow/util/image.hpp"

void ActivateLayer::Reshape() {
  type_ = layer_param_.activate_param().type();

  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottom_[0]->shape(), ",", "(", ")") << " -> "
      << Util::format_vector(top_[0]->shape(), ",", "(", ")");
  DInfo(out.str());
}

void ActivateLayer::Forward() {
  Image::Activate(top_[0]->mutable_data(), top_[0]->count(), type_);
}

void ActivateLayer::Release() {
  bottom_.clear();
  top_.clear();

  // DInfo("Free ActivateLayer!");
}
