#include "activations.h"
#include <cmath>

inline float linear_activate(float x) { return x; }
inline float logistic_activate(float x) { return 1 / (1 + expf(-x)); }
inline float relu_activate(float x) { return x * (x > 0); }
inline float elu_activate(float x) {
  return (x >= 0) * x + (x < 0) * (expf(x) - 1);
}
inline float relie_activate(float x) { return x * (x > 0); }
inline float ramp_activate(float x) { return x * (x > 0) + .1f * x; }
inline float leaky_activate(float x) { return (x > 0) ? x : .1f * x; }
inline float tanh_activate(float x) {
  return (expf(2 * x) - 1) / (expf(2 * x) + 1);
}
inline float plse_activate(float x) {
  if (x < -4)
    return .01f * (x + 4);
  if (x > 4)
    return .01f * (x - 4) + 1;
  return .125f * x + .5f;
}

float Activate(float x, Activation a) {
  switch (a) {
  case kLinear:
    return linear_activate(x);
  case kRelu:
    return relu_activate(x);
  case kLeaky:
    return leaky_activate(x);
  case kLogistic:
    return logistic_activate(x);
  case kElu:
    return elu_activate(x);
  case kRelie:
    return relie_activate(x);
  case kRamp:
    return ramp_activate(x);
  case kTanh:
    return tanh_activate(x);
  case kPlse:
    return plse_activate(x);
  }
  return 0;
}

Activation Activations::GetActivation(std::string s) {
  if (!s.compare("linear"))
    return kLinear;
  if (!s.compare("relu"))
    return kRelu;
  if (!s.compare("leaky"))
    return kLeaky;
  if (!s.compare("logistic"))
    return kLogistic;
  if (!s.compare("elu"))
    return kElu;
  if (!s.compare("relie"))
    return kRelie;
  if (!s.compare("plse"))
    return kPlse;
  if (!s.compare("ramp"))
    return kRamp;
  if (!s.compare("tanh"))
    return kTanh;
  return kRelu;
}

void Activations::ActivateArray(const int N, const Activation a,
                                float *out_data) {
  for (int i = 0; i < N; ++i) {
    out_data[i] = Activate(out_data[i], a);
  }
}
