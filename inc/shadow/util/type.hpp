#ifndef SHADOW_UTIL_TYPE_HPP
#define SHADOW_UTIL_TYPE_HPP

#if !defined(__linux)
#define _USE_MATH_DEFINES
#endif

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <list>
#include <string>
#include <vector>

const float EPS = 0.000001f;

class Scalar {
 public:
  Scalar() {}
  Scalar(int r_t, int g_t, int b_t) : r(r_t), g(g_t), b(b_t) {}
  unsigned char r = 0, g = 0, b = 0;
};

template <class Dtype>
class Point {
 public:
  Point() {}
  Point(Dtype x_t, Dtype y_t, float score_t = -1)
      : x(x_t), y(y_t), score(score_t) {}
  Point(const Point<int> &p) : x(p.x), y(p.y), score(p.score) {}
  Point(const Point<float> &p) : x(p.x), y(p.y), score(p.score) {}
  Dtype x = 0, y = 0;
  float score = 0;
};

template <class Dtype>
class Rect {
 public:
  Rect() {}
  Rect(Dtype x_t, Dtype y_t, Dtype w_t, Dtype h_t)
      : x(x_t), y(y_t), w(w_t), h(h_t) {}
  Rect(const Rect<int> &rect) : x(rect.x), y(rect.y), w(rect.w), h(rect.h) {}
  Rect(const Rect<float> &rect) : x(rect.x), y(rect.y), w(rect.w), h(rect.h) {}
  Dtype x = 0, y = 0, w = 0, h = 0;
};

template <class Dtype>
class Size {
 public:
  Size() {}
  Size(Dtype w_t, Dtype h_t) : w(w_t), h(h_t) {}
  Size(const Size<int> &size) : w(size.w), h(size.h) {}
  Size(const Size<float> &size) : w(size.w), h(size.h) {}
  Dtype w = 0, h = 0;
};

typedef Point<int> PointI;
typedef Point<float> PointF;
typedef Rect<int> RectI;
typedef Rect<float> RectF;
typedef Size<int> SizeI;
typedef Size<float> SizeF;

typedef std::vector<PointI> VecPointI;
typedef std::vector<PointF> VecPointF;
typedef std::vector<RectI> VecRectI;
typedef std::vector<RectF> VecRectF;
typedef std::vector<SizeI> VecSizeI;
typedef std::vector<SizeF> VecSizeF;

typedef std::vector<int> VecInt;
typedef std::vector<float> VecFloat;
typedef std::vector<double> VecDouble;
typedef std::vector<std::string> VecString;
typedef std::list<int> ListInt;
typedef std::list<float> ListFloat;
typedef std::list<double> ListDouble;
typedef std::list<std::string> ListString;

#endif  // SHADOW_UTIL_TYPE_HPP
