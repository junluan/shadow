#ifndef SHADOW_UTIL_H
#define SHADOW_UTIL_H

#include <cmath>
#include <iostream>
#include <string>

static void inline error(std::string msg) {
  std::cerr << msg << std::endl;
  exit(1);
}

static void inline warn(std::string msg) {
  std::cout << msg << std::endl;
  exit(0);
}

static float inline rand_uniform(float min, float max) {
  return ((float)std::rand() / RAND_MAX) * (max - min) + min;
}

static float inline constrain(float min, float max, float value) {
  if (value < min)
    return min;
  if (value > max)
    return max;
  return value;
}

#endif // SHADOW_UTIL_H
