#ifndef SHADOW_UTIL_UTIL_HPP
#define SHADOW_UTIL_UTIL_HPP

#if defined(USE_GLog)
#include <glog/logging.h>
#endif

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
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

#if defined(USE_GLog)
#define Info(msg)                                                              \
  { LOG(INFO) << msg; }

#define Warning(msg)                                                           \
  { LOG(WARNING) << msg; }

#define Error(msg)                                                             \
  { LOG(ERROR) << msg; }

#define Fatal(msg)                                                             \
  { LOG(FATAL) << msg; }
#else
inline static void Fatal(const std::string msg) {
  std::cerr << msg << std::endl;
  exit(1);
}

inline static void Warning(const std::string msg) {
  std::cout << msg << std::endl;
}
#endif

inline static float rand_uniform(float min, float max) {
  return (static_cast<float>(std::rand()) / RAND_MAX) * (max - min) + min;
}

template <typename Dtype>
inline static Dtype constrain(Dtype min, Dtype max, Dtype value) {
  if (value < min)
    return min;
  if (value > max)
    return max;
  return value;
}

template <typename Dtype> inline static std::string to_string(Dtype val) {
  std::ostringstream out;
  out << val;
  return out.str();
}

inline static std::string format_int(int n, int width, char pad = ' ') {
  std::stringstream out;
  out << std::setw(width) << std::setfill(pad) << n;
  return out.str();
}

inline static std::string find_replace(const std::string str,
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

inline static std::string find_replace_last(const std::string str,
                                            const std::string old_str,
                                            const std::string new_str) {
  std::string origin(str);
  size_t index = origin.find_last_of(old_str);
  origin.replace(index, old_str.length(), new_str);
  return origin;
}

inline static std::string change_extension(const std::string str,
                                           const std::string new_ext) {
  std::string origin(str);
  size_t index = origin.find_last_of(".");
  origin.replace(index, origin.length(), new_ext);
  return origin;
}

inline static std::vector<std::string> LoadList(const std::string list_file) {
  std::ifstream file(list_file);
  if (!file.is_open())
    Fatal("Load image list file error!");

  std::vector<std::string> image_list;
  std::string dir;
  while (std::getline(file, dir)) {
    if (dir.length())
      image_list.push_back(dir);
  }
  file.close();
  return image_list;
}

inline static std::string read_text_from_file(const std::string filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    Warning("Can't open text file " + filename);
    return std::string();
  }

  std::stringstream result;
  std::string tmp;
  while (std::getline(file, tmp))
    result << tmp << std::endl;
  file.close();
  return result.str();
}

#include <ctime>
class Timer {
public:
  Timer() : ts(clock()) {}

  void start() { ts = clock(); }
  double get_second() const {
    return static_cast<double>(clock() - ts) / CLOCKS_PER_SEC;
  }
  double get_millisecond() const {
    return 1000.0 * static_cast<double>(clock() - ts) / CLOCKS_PER_SEC;
  }
  double get_microsecond() const { return static_cast<double>(clock() - ts); }

private:
  clock_t ts;
};

#endif // SHADOW_UTIL_UTIL_HPP
