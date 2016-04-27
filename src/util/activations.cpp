#include "activations.hpp"
#include <cmath>

float Activate(float x, Activation a) {
  switch (a) {
  case kLinear:
    return x;
  case kRelu:
    return x * (x > 0);
  case kLeaky:
    return (x > 0) ? x : .1f * x;
  }
  return x;
}

Activation Activations::GetActivation(std::string s) {
  if (!s.compare("linear"))
    return kLinear;
  if (!s.compare("relu"))
    return kRelu;
  if (!s.compare("leaky"))
    return kLeaky;
  return kLinear;
}

void Activations::ActivateArray(const int N, const Activation a,
                                float *out_data) {
  for (int i = 0; i < N; ++i) {
    out_data[i] = Activate(out_data[i], a);
  }
}
