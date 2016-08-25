#ifndef SHADOW_UTIL_UTIL_HPP
#define SHADOW_UTIL_UTIL_HPP

#if !defined(__linux)
#define _USE_MATH_DEFINES
#endif

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <vector>

const float EPS = 0.001f;

template <class Dtype>
class Point {
 public:
  Point() {}
  Point(Dtype x_t, Dtype y_t, float score_t = -1)
      : x(x_t), y(y_t), score(score_t) {}
  Point(const Point<int> &p) : x(p.x), y(p.y), score(p.score) {}
  Point(const Point<float> &p) : x(p.x), y(p.y), score(p.score) {}
  Dtype x, y;
  float score;
};

template <class Dtype>
class Rect {
 public:
  Rect() {}
  Rect(Dtype x_t, Dtype y_t, Dtype w_t, Dtype h_t)
      : x(x_t), y(y_t), w(w_t), h(h_t) {}
  Rect(const Rect<int> &rect) : x(rect.x), y(rect.y), w(rect.w), h(rect.h) {}
  Rect(const Rect<float> &rect) : x(rect.x), y(rect.y), w(rect.w), h(rect.h) {}
  Dtype x, y, w, h;
};

template <class Dtype>
class Size {
 public:
  Size() {}
  Size(Dtype w_t, Dtype h_t) : w(w_t), h(h_t) {}
  Size(const Size<int> &size) : w(size.w), h(size.h) {}
  Size(const Size<float> &size) : w(size.w), h(size.h) {}
  Dtype w, h;
};

typedef Point<float> PointF;
typedef Point<int> PointI;
typedef Rect<float> RectF;
typedef Rect<int> RectI;
typedef Size<float> SizeF;
typedef Size<int> SizeI;

typedef std::vector<PointF> VecPointF;
typedef std::vector<PointI> VecPointI;
typedef std::vector<RectF> VecRectF;
typedef std::vector<RectI> VecRectI;
typedef std::vector<SizeF> VecSizeF;
typedef std::vector<SizeI> VecSizeI;

typedef std::vector<float> VecFloat;
typedef std::vector<int> VecInt;
typedef std::list<float> ListFloat;
typedef std::list<int> ListInt;

#if defined(USE_GLog)
#if !defined(__linux)
#define GOOGLE_GLOG_DLL_DECL
#endif
#include <glog/logging.h>

#define Info(msg) \
  { LOG(INFO) << msg; }

#define Warning(msg) \
  { LOG(WARNING) << msg; }

#define Error(msg) \
  { LOG(ERROR) << msg; }

#define Fatal(msg) \
  { LOG(FATAL) << msg; }

#else
#define Info(msg) \
  { std::cout << __FILE__ << ":" << __LINE__ << " Info: " << msg << std::endl; }

#define Warning(msg)                                                \
  {                                                                 \
    std::cerr << __FILE__ << ":" << __LINE__ << " Warning: " << msg \
              << std::endl;                                         \
  }

#define Fatal(msg)                                                \
  {                                                               \
    std::cerr << __FILE__ << ":" << __LINE__ << " Fatal: " << msg \
              << std::endl;                                       \
    exit(1);                                                      \
  }
#endif

namespace Util {

template <typename Dtype>
inline int round(Dtype x) {
  return static_cast<int>(std::floor(x + 0.5));
}

inline float rand_uniform(float min, float max) {
  return (static_cast<float>(std::rand()) / RAND_MAX) * (max - min) + min;
}

template <typename Dtype>
inline Dtype constrain(Dtype min, Dtype max, Dtype value) {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

template <typename Dtype>
inline std::string str(Dtype val) {
  std::stringstream out;
  out << val;
  return out.str();
}

inline std::string format_int(int n, int width, char pad = ' ') {
  std::stringstream out;
  out << std::setw(width) << std::setfill(pad) << n;
  return out.str();
}

inline std::string format_vector(const std::vector<int> &shape,
                                 const std::string &split = ", ",
                                 const std::string &postfix = "") {
  std::stringstream out;
  int i = 0;
  for (; i < shape.size() - 1; ++i) {
    out << shape.at(i) << split;
  }
  out << shape.at(std::min(i, static_cast<int>(shape.size()) - 1)) << postfix;
  return out.str();
}

inline std::string find_replace(const std::string &str,
                                const std::string &old_str,
                                const std::string &new_str) {
  std::string origin(str);
  size_t index = 0;
  while ((index = origin.find(old_str, index)) != std::string::npos) {
    origin.replace(index, old_str.length(), new_str);
    index += new_str.length();
  }
  return origin;
}

inline std::string find_replace_last(const std::string &str,
                                     const std::string &old_str,
                                     const std::string &new_str) {
  std::string origin(str);
  size_t index = origin.find_last_of(old_str);
  origin.replace(index, old_str.length(), new_str);
  return origin;
}

inline std::string change_extension(const std::string &str,
                                    const std::string &new_ext) {
  std::string origin(str);
  size_t index = origin.find_last_of(".");
  origin.replace(index, origin.length(), new_ext);
  return origin;
}

inline std::vector<std::string> tokenize(const std::string &str,
                                         const std::string &split) {
  std::string::size_type last_pos = 0;
  std::string::size_type pos = str.find_first_of(split, last_pos);
  std::vector<std::string> tokens;
  while (last_pos != std::string::npos) {
    if (pos != last_pos) tokens.push_back(str.substr(last_pos, pos - last_pos));
    last_pos = pos;
    if (last_pos == std::string::npos || last_pos + 1 == str.length()) break;
    pos = str.find_first_of(split, ++last_pos);
  }
  return tokens;
}

inline std::vector<std::string> load_list(const std::string &list_file) {
  std::ifstream file(list_file);
  if (!file.is_open()) Fatal("Load image list file error!");

  std::vector<std::string> image_list;
  std::string dir;
  while (std::getline(file, dir)) {
    if (dir.length()) image_list.push_back(dir);
  }
  file.close();
  return image_list;
}

inline std::string read_text_from_file(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    Warning("Can't open text file " + filename);
    return std::string();
  }

  std::stringstream result;
  std::string tmp;
  while (std::getline(file, tmp)) result << tmp << std::endl;
  file.close();
  return result.str();
}

}  // namespace Util

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

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <linux/limits.h>
#include <unistd.h>
#endif

#include <sys/stat.h>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
class Path {
 public:
  enum PathType {
    kWindows = 0,
    kPosix = 1,
#if defined(_WIN32)
    KNative = kWindows
#else
    KNative = kPosix
#endif
  };

  Path() : type_(KNative), absolute_(false) {}
  Path(const Path &path)
      : type_(path.type_), path_(path.path_), absolute_(path.absolute_) {}
  explicit Path(const char *str) { set(str); }
  explicit Path(const std::string &str) { set(str); }

  bool is_empty() const { return path_.empty(); }

  bool is_exist() const {
#if defined(_WIN32)
    return GetFileAttributesW(wstr().c_str()) != INVALID_FILE_ATTRIBUTES;
#else
    struct stat sb;
    return stat(str().c_str(), &sb) == 0;
#endif
  }

  bool is_directory() const {
#if defined(_WIN32)
    DWORD attr = GetFileAttributesW(wstr().c_str());
    return (attr != INVALID_FILE_ATTRIBUTES &&
            (attr & FILE_ATTRIBUTE_DIRECTORY) != 0);
#else
    struct stat sb;
    if (stat(str().c_str(), &sb)) return false;
    return S_ISDIR(sb.st_mode);
#endif
  }

  bool is_file() const {
#if defined(_WIN32)
    DWORD attr = GetFileAttributesW(wstr().c_str());
    return (attr != INVALID_FILE_ATTRIBUTES &&
            (attr & FILE_ATTRIBUTE_DIRECTORY) == 0);
#else
    struct stat sb;
    if (stat(str().c_str(), &sb)) return false;
    return S_ISREG(sb.st_mode);
#endif
  }

  bool is_absolute() const { return absolute_; }

  Path make_absolute() const {
#if defined(_WIN32)
    std::wstring value = wstr(), out(MAX_PATH, '\0');
    DWORD length = GetFullPathNameW(value.c_str(), MAX_PATH, &out[0], NULL);
    if (length == 0)
      Fatal("Error in make_absolute(): " + std::to_string(GetLastError()));
    std::wstring temp = out.substr(0, length);
    return Path(std::string(temp.begin(), temp.end()));
#else
    char temp[PATH_MAX];
    if (realpath(str().c_str(), temp) == NULL)
      Fatal("Error in make_absolute(): " + std::string(strerror(errno)));
    return Path(temp);
#endif
  }

  std::string file_name() const {
    if (path_.empty() || !is_file()) return "";
    return path_[path_.size() - 1];
  }

  std::string folder_name() {
    if (path_.empty() || !is_directory()) return "";
    return path_[path_.size() - 1];
  }

  std::string name() const {
    std::string name = file_name();
    size_t pos = name.find_last_of(".");
    if (pos == std::string::npos) return "";
    return name.substr(0, pos);
  }

  std::string extension() const {
    std::string name = file_name();
    size_t pos = name.find_last_of(".");
    if (pos == std::string::npos) return "";
    return name.substr(pos + 1);
  }

  size_t length() const { return path_.size(); }

  size_t file_size() const {
#if defined(_WIN32)
    struct _stati64 sb;
    if (_wstati64(wstr().c_str(), &sb) != 0)
      Fatal("Error in file_size(): " + str());
#else
    struct stat sb;
    if (stat(str().c_str(), &sb) != 0) Fatal("Error in file_size(): " + str());
#endif
    return (size_t)sb.st_size;
  }

  Path parent_path() const {
    Path result;
    result.absolute_ = absolute_;
    if (path_.empty()) {
      if (!absolute_) result.path_.push_back("..");
    } else {
      for (size_t i = 0; i < path_.size() - 1; ++i)
        result.path_.push_back(path_[i]);
    }
    return result;
  }

  std::string str(PathType type = KNative) const {
    std::stringstream oss;
    if (type_ == kPosix && absolute_) oss << "/";
    for (size_t i = 0; i < path_.size(); ++i) {
      oss << path_[i];
      if (i + 1 < path_.size()) {
        if (type == kPosix)
          oss << '/';
        else
          oss << '\\';
      }
    }
    return oss.str();
  }

  std::wstring wstr(PathType type = KNative) const {
    std::string temp = str(type);
    return std::wstring(temp.begin(), temp.end());
  }

  void set(const std::string &str, PathType type = KNative) {
    type_ = type;
    if (type == kWindows) {
      path_ = Util::tokenize(str, "/\\");
      absolute_ = str.size() >= 2 && std::isalpha(str[0]) && str[1] == ':';
    } else {
      path_ = Util::tokenize(str, "/");
      absolute_ = !str.empty() && str[0] == '/';
    }
  }

  bool remove_file() {
#if defined(_WIN32)
    return DeleteFileW(wstr().c_str()) != 0;
#else
    return std::remove(str().c_str()) == 0;
#endif
  }

  bool resize_file(size_t target_length) {
#if defined(_WIN32)
    HANDLE handle = CreateFileW(wstr().c_str(), GENERIC_WRITE, 0, nullptr, 0,
                                FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER size;
    size.QuadPart = (LONGLONG)target_length;
    if (SetFilePointerEx(handle, size, NULL, FILE_BEGIN) == 0) {
      CloseHandle(handle);
      return false;
    }
    if (SetEndOfFile(handle) == 0) {
      CloseHandle(handle);
      return false;
    }
    CloseHandle(handle);
    return true;
#else
    return ::truncate(str().c_str(), (off_t)target_length) == 0;
#endif
  }

  static Path cwd() {
#if defined(_WIN32)
    std::wstring temp(MAX_PATH, '\0');
    if (!_wgetcwd(&temp[0], MAX_PATH))
      Fatal("Error in cwd(): " + std::to_string(GetLastError()));
    return Path(std::string(temp.begin(), temp.end()));
#else
    char temp[PATH_MAX];
    if (::getcwd(temp, PATH_MAX) == NULL)
      Fatal("Error in cwd(): " + std::string(strerror(errno)));
    return Path(temp);
#endif
  }

  Path operator/(const Path &other) const {
    if (other.absolute_) Fatal("Error in operator/: expected a relative path!");
    if (type_ != other.type_)
      Fatal("Error in operator/: expected a path of the same type!");
    Path result(*this);
    for (size_t i = 0; i < other.path_.size(); ++i)
      result.path_.push_back(other.path_[i]);
    return result;
  }
  Path operator+(const Path &other) const {
    if (other.absolute_) Fatal("Error in operator/: expected a relative path!");
    if (type_ != other.type_)
      Fatal("Error in operator/: expected a path of the same type!");
    Path result(*this);
    for (size_t i = 0; i < other.path_.size(); ++i)
      result.path_.push_back(other.path_[i]);
    return result;
  }

  Path &operator=(const Path &path) {
    type_ = path.type_;
    path_ = path.path_;
    absolute_ = path.absolute_;
    return *this;
  }

  bool operator==(const Path &p) const { return p.path_ == path_; }
  bool operator!=(const Path &p) const { return p.path_ != path_; }

  friend std::ostream &operator<<(std::ostream &os, const Path &path) {
    os << path.str();
    return os;
  }

 private:
  PathType type_;
  std::vector<std::string> path_;
  bool absolute_;
};

namespace Util {

inline bool make_directory(const Path &path) {
#if defined(_WIN32)
  return CreateDirectoryW(path.wstr().c_str(), NULL) != 0;
#else
  return mkdir(path.str().c_str(), S_IRUSR | S_IWUSR | S_IXUSR) == 0;
#endif
}

}  // namespace Util

#endif  // SHADOW_UTIL_UTIL_HPP
