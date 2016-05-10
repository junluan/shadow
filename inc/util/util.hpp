#ifndef SHADOW_UTIL_HPP
#define SHADOW_UTIL_HPP

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

template <class Dtype> class Rect {
public:
  Rect() {}
  Rect(Dtype x_t, Dtype y_t, Dtype w_t, Dtype h_t) {
    x = x_t;
    y = y_t;
    w = w_t;
    h = h_t;
  }
  Dtype x, y, w, h;
};

template <class Dtype> class Size {
public:
  Size() {}
  Size(Dtype w_t, Dtype h_t) {
    w = w_t;
    h = h_t;
  }
  Dtype w, h;
};

typedef Rect<float> RectF;
typedef Rect<int> RectI;
typedef Size<int> SizeI;
typedef std::vector<RectF> VecRectF;
typedef std::vector<RectI> VecRectI;

static void inline error(const std::string msg) {
  std::cerr << msg << std::endl;
  exit(1);
}

static void inline warn(const std::string msg) {
  std::cout << msg << std::endl;
}

static float inline rand_uniform(float min, float max) {
  return (static_cast<float>(std::rand()) / RAND_MAX) * (max - min) + min;
}

template <typename Dtype>
static Dtype inline constrain(Dtype min, Dtype max, Dtype value) {
  if (value < min)
    return min;
  if (value > max)
    return max;
  return value;
}

static std::string find_replace(const std::string str,
                                const std::string old_str,
                                const std::string new_str) {
  std::string origin(str);
  size_t index = 0;
  while ((index = origin.find(old_str, index)) != std::string::npos) {
    origin.replace(index, old_str.length(), new_str);
    index += new_str.length();
  }
  return origin;
}

static std::string find_replace_last(const std::string str,
                                     const std::string old_str,
                                     const std::string new_str) {
  std::string origin(str);
  size_t index = origin.find_last_of(old_str);
  origin.replace(index, old_str.length(), new_str);
  return origin;
}

static std::string change_extension(const std::string str,
                                    const std::string new_ext) {
  std::string origin(str);
  size_t index = origin.find_last_of(".");
  origin.replace(index, origin.length(), new_ext);
  return origin;
}

#endif // SHADOW_UTIL_HPP
