#ifndef SHADOW_ACTIVATIONS_H
#define SHADOW_ACTIVATIONS_H

#include <string>

enum Activation {
  kLinear,
  kRelu,
  kLeaky,
  kLogistic,
  kRelie,
  kRamp,
  kTanh,
  kPlse,
  kElu
};

class Activations {
public:
  static Activation GetActivation(std::string s);
  static void ActivateArray(const int N, const Activation a, float *out_data);
};

#endif // SHADOW_ACTIVATIONS_H
