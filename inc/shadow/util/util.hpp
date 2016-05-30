#ifndef SHADOW_UTIL_UTIL_HPP
#define SHADOW_UTIL_UTIL_HPP

#ifdef USE_GLog
#include <glog/logging.h>
#endif

#include <cmath>
#include <fcntl.h>
#include <fstream>
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

#ifdef USE_GLog
#define Info(msg)                                                              \
  { LOG(INFO) << msg; }

#define Warning(msg)                                                           \
  { LOG(WARNING) << msg; }

#define Error(msg)                                                             \
  { LOG(ERROR) << msg; }

#define Fatal(msg)                                                             \
  { LOG(FATAL) << msg; }
#else
static void inline Fatal(const std::string msg) {
  std::cerr << msg << std::endl;
  exit(1);
}

static void inline Warning(const std::string msg) {
  std::cout << msg << std::endl;
}
#endif

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

static std::vector<std::string> LoadList(std::string list_file) {
  std::cout << "Loading image list from " << list_file << " ... ";
  std::ifstream file(list_file);
  if (!file.is_open())
    Fatal("Load image list file error!");

  std::string dir;
  std::vector<std::string> image_list;
  while (getline(file, dir)) {
    if (dir.length())
      image_list.push_back(dir);
  }
  file.close();
  std::cout << "Done!" << std::endl;
  return image_list;
}

#endif // SHADOW_UTIL_UTIL_HPP
