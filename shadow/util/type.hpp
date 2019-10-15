#ifndef SHADOW_UTIL_TYPE_HPP
#define SHADOW_UTIL_TYPE_HPP

#if defined(_WIN32)
#pragma warning(disable : 4244)
#define _USE_MATH_DEFINES
#endif

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <list>
#include <string>
#include <vector>

namespace Shadow {

const float EPS = 1e-5f;

class Scalar {
 public:
  Scalar() = default;
  Scalar(int r_t, int g_t, int b_t) {
    r = static_cast<unsigned char>(r_t);
    g = static_cast<unsigned char>(g_t);
    b = static_cast<unsigned char>(b_t);
  }
  unsigned char r = 0, g = 0, b = 0;
};

template <typename T>
class Point {
 public:
  Point() = default;
  Point(T x_t, T y_t, float score_t = -1) : x(x_t), y(y_t), score(score_t) {}
  Point(const Point<int> &p) : x(p.x), y(p.y), score(p.score) {}
  Point(const Point<float> &p) : x(p.x), y(p.y), score(p.score) {}
  T x = 0, y = 0;
  float score = -1;
};

template <typename T>
class Rect {
 public:
  Rect() = default;
  Rect(T x_t, T y_t, T w_t, T h_t) : x(x_t), y(y_t), w(w_t), h(h_t) {}
  Rect(const Rect<int> &rect) : x(rect.x), y(rect.y), w(rect.w), h(rect.h) {}
  Rect(const Rect<float> &rect) : x(rect.x), y(rect.y), w(rect.w), h(rect.h) {}
  T x = 0, y = 0, w = 0, h = 0;
};

template <typename T>
class Size {
 public:
  Size() = default;
  Size(T w_t, T h_t) : w(w_t), h(h_t) {}
  Size(const Size<int> &size) : w(size.w), h(size.h) {}
  Size(const Size<float> &size) : w(size.w), h(size.h) {}
  T w = 0, h = 0;
};

using PointI = Point<int>;
using PointF = Point<float>;
using RectI = Rect<int>;
using RectF = Rect<float>;
using SizeI = Size<int>;
using SizeF = Size<float>;

using VecPointI = std::vector<PointI>;
using VecPointF = std::vector<PointF>;
using VecRectI = std::vector<RectI>;
using VecRectF = std::vector<RectF>;
using VecSizeI = std::vector<SizeI>;
using VecSizeF = std::vector<SizeF>;

using VecChar = std::vector<char>;
using VecUChar = std::vector<unsigned char>;
using VecBool = std::vector<bool>;
using VecInt = std::vector<int>;
using VecFloat = std::vector<float>;
using VecDouble = std::vector<double>;
using VecString = std::vector<std::string>;
using ListChar = std::list<char>;
using ListUChar = std::list<unsigned char>;
using ListBool = std::list<bool>;
using ListInt = std::list<int>;
using ListFloat = std::list<float>;
using ListDouble = std::list<double>;
using ListString = std::list<std::string>;

}  // namespace Shadow

#endif  // SHADOW_UTIL_TYPE_HPP
