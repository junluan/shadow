#include "shadow/layers/dropout_layer.hpp"

void DropoutLayer::Reshape() {
  std::stringstream out;
  out << layer_name_ << ": "
      << Util::format_vector(bottoms_[0]->shape(), ",", "(", ")") << " -> "
      << Util::format_vector(tops_[0]->shape(), ",", "(", ")");
  DInfo(out.str());
}

void DropoutLayer::Forward() {}

void DropoutLayer::Release() {
  bottoms_.clear();
  tops_.clear();

  // DInfo("Free DropoutLayer!");
}
