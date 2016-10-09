#ifndef SHADOW_UTIL_ACTIVATIONS_HPP
#define SHADOW_UTIL_ACTIVATIONS_HPP

#include "shadow/kernel.hpp"

namespace Activations {

inline float Activate(float x, int type) {
  switch (type) {
    case 0:
      return x;
    case 1:
      return x * (x > 0);
    case 2:
      return (x > 0) ? x : .1f * x;
  }
  return x;
}

template <typename T>
void ActivateArray(int N, int type, T *out_data) {
#if !defined(USE_CUDA) & !defined(USE_CL)
  for (int i = 0; i < N; ++i) {
    out_data[i] = Activate(out_data[i], type);
  }

#else
  Kernel::ActivateArray(N, type, out_data);
#endif
}

}  // namespace Activations

#endif  // SHADOW_UTIL_ACTIVATIONS_HPP
