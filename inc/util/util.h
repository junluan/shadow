#ifndef SHADOW_UTIL_H
#define SHADOW_UTIL_H

#include <cmath>
#include <iostream>
#include <string>

static void inline error(std::string msg) {
  std::cerr << msg << std::endl;
  exit(1);
}

static void inline warn(std::string msg) { std::cout << msg << std::endl; }

static float inline rand_uniform(float min, float max) {
  return ((float)std::rand() / RAND_MAX) * (max - min) + min;
}

template <typename Dtype>
static Dtype inline constrain(Dtype min, Dtype max, Dtype value) {
  if (value < min)
    return min;
  if (value > max)
    return max;
  return value;
}

static std::string find_replace(const std::string str, std::string oldstr,
                                std::string newstr) {
  std::string origin(str);
  size_t index = 0;
  while ((index = origin.find(oldstr, index)) != std::string::npos) {
    origin.replace(index, oldstr.length(), newstr);
    index += newstr.length();
  }
  return origin;
}

static std::string find_replace_last(const std::string str, std::string oldstr,
                                     std::string newstr) {
  std::string origin(str);
  size_t index = origin.find_last_of(oldstr);
  origin.replace(index, oldstr.length(), newstr);
  return origin;
}

static std::string change_extension(const std::string str, std::string newext) {
  std::string origin(str);
  size_t index = origin.find_last_of(".");
  origin.replace(index, origin.length(), newext);
  return origin;
}

#endif // SHADOW_UTIL_H
