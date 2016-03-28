#ifndef SHADOW_ACTIVATIONS_H
#define SHADOW_ACTIVATIONS_H

#include <string>

#ifdef USE_CL
#include "cl.h"
#endif

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
  Activations();
  ~Activations();

  static Activation GetActivation(std::string s);
  static void ActivateArray(const int N, const Activation a, float *out_data);

#ifdef USE_CL
  static void CLActivateArray(const int N, const Activation a, cl_mem out_data);
#endif

private:
  static float Activate(float x, Activation a);
};

#endif // SHADOW_ACTIVATIONS_H
