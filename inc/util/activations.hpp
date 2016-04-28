#ifndef SHADOW_ACTIVATIONS_HPP
#define SHADOW_ACTIVATIONS_HPP

#include <string>

enum Activation { kLinear, kRelu, kLeaky };

class Activations {
public:
  static Activation GetActivation(std::string s);
  static void ActivateArray(int N, Activation a, float *out_data);
};

#endif // SHADOW_ACTIVATIONS_HPP
