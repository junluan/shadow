#ifndef SHADOW_UTIL_UTIL_HPP
#define SHADOW_UTIL_UTIL_HPP

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Util {

template <typename Dtype>
inline int round(Dtype x) {
  return static_cast<int>(std::floor(x + 0.5));
}

inline float rand_uniform(float min, float max) {
  static std::default_random_engine generator(
      static_cast<unsigned long>(time(nullptr)));
  std::uniform_real_distribution<float> distribute(min, max);
  return distribute(generator);
}

template <typename Dtype>
inline Dtype constrain(Dtype min, Dtype max, Dtype value) {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

inline bool pair_ascend(const std::pair<float, int> &lhs,
                        const std::pair<float, int> &rhs) {
  return lhs.first < rhs.first;
}

inline bool pair_descend(const std::pair<float, int> &lhs,
                         const std::pair<float, int> &rhs) {
  return lhs.first > rhs.first;
}

inline std::vector<int> top_k(const std::vector<float> &v, int K,
                              bool descend = true) {
  std::vector<std::pair<float, int>> pairs;
  for (int i = 0; i < v.size(); ++i) {
    pairs.emplace_back(v[i], i);
  }
  if (descend) {
    std::partial_sort(pairs.begin(), pairs.begin() + K, pairs.end(),
                      pair_descend);
  } else {
    std::partial_sort(pairs.begin(), pairs.begin() + K, pairs.end(),
                      pair_ascend);
  }
  std::vector<int> result;
  for (int k = 0; k < K; ++k) {
    result.push_back(pairs[k].second);
  }
  return result;
}

template <typename Dtype>
inline std::string to_string(const Dtype val) {
  std::stringstream ss;
  ss << val;
  return ss.str();
}

inline std::string format_int(int n, int width, char pad = ' ') {
  std::stringstream ss;
  ss << std::setw(width) << std::setfill(pad) << n;
  return ss.str();
}

inline std::string format_process(int current, int total) {
  int digits = total > 0 ? static_cast<int>(std::log10(total)) + 1 : 1;
  std::stringstream ss;
  ss << format_int(current, digits) << " / " << total;
  return ss.str();
}

template <typename Dtype>
inline std::string format_vector(const std::vector<Dtype> &vector,
                                 const std::string &split = ",",
                                 const std::string &prefix = "",
                                 const std::string &postfix = "") {
  std::stringstream ss;
  ss << prefix;
  for (int i = 0; i < vector.size() - 1; ++i) {
    ss << vector.at(i) << split;
  }
  if (vector.size() >= 1) {
    ss << vector.at(vector.size() - 1);
  }
  ss << postfix;
  return ss.str();
}

inline std::string find_replace(const std::string &str,
                                const std::string &old_str,
                                const std::string &new_str) {
  std::string origin(str);
  size_t pos = 0;
  while ((pos = origin.find(old_str, pos)) != std::string::npos) {
    origin.replace(pos, old_str.length(), new_str);
    pos += new_str.length();
  }
  return origin;
}

inline std::string find_replace_last(const std::string &str,
                                     const std::string &old_str,
                                     const std::string &new_str) {
  std::string origin(str);
  auto pos = origin.find_last_of(old_str);
  origin.replace(pos, old_str.length(), new_str);
  return origin;
}

inline std::string change_extension(const std::string &str,
                                    const std::string &new_ext) {
  std::string origin(str);
  auto pos = origin.find_last_of('.');
  origin.replace(pos, origin.length(), new_ext);
  return origin;
}

inline std::vector<std::string> tokenize(const std::string &str,
                                         const std::string &split) {
  std::string::size_type last_pos = 0;
  auto pos = str.find_first_of(split, last_pos);
  std::vector<std::string> tokens;
  while (last_pos != std::string::npos) {
    if (pos != last_pos) {
      tokens.push_back(str.substr(last_pos, pos - last_pos));
    }
    last_pos = pos;
    if (last_pos == std::string::npos || last_pos + 1 == str.length()) break;
    pos = str.find_first_of(split, ++last_pos);
  }
  return tokens;
}

inline std::vector<std::string> load_list(const std::string &list_file) {
  std::ifstream file(list_file);
  if (!file.is_open()) {
    throw std::runtime_error("Error when loading list file: " + list_file);
  }
  std::vector<std::string> list;
  std::string str;
  while (std::getline(file, str)) {
    if (!str.empty()) {
      list.push_back(str);
    }
  }
  file.close();
  return list;
}

inline std::string read_text_from_file(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Can't open text file " << filename;
    return std::string();
  }
  std::stringstream ss;
  std::string tmp;
  while (std::getline(file, tmp)) {
    ss << tmp << std::endl;
  }
  file.close();
  return ss.str();
}

}  // namespace Util

#if defined(__linux)
#include <linux/limits.h>
#include <unistd.h>
#include <chrono>
using namespace std::chrono;
#else
#define NOMINMAX
#include <windows.h>
#endif

#include <sys/stat.h>
class Path {
 public:
  enum PathType {
    kPosix = 0,
    kWindows = 1,
#if defined(__linux)
    KNative = kPosix
#else
    KNative = kWindows
#endif
  };

  Path() = default;
  Path(const Path &path) = default;
  explicit Path(const char *str) { set(str); }
  explicit Path(const std::string &str) { set(str); }

  bool is_empty() const { return path_.empty(); }

  bool is_exist() const {
#if defined(__linux)
    struct stat sb {};
    return stat(str().c_str(), &sb) == 0;
#else
    return GetFileAttributesW(wstr().c_str()) != INVALID_FILE_ATTRIBUTES;
#endif
  }

  bool is_directory() const {
#if defined(__linux)
    struct stat sb {};
    if (stat(str().c_str(), &sb) != 0) return false;
    return S_ISDIR(sb.st_mode);
#else
    DWORD attr = GetFileAttributesW(wstr().c_str());
    return attr != INVALID_FILE_ATTRIBUTES &&
           (attr & FILE_ATTRIBUTE_DIRECTORY) != 0;
#endif
  }

  bool is_file() const {
#if defined(__linux)
    struct stat sb {};
    if (stat(str().c_str(), &sb) != 0) return false;
    return S_ISREG(sb.st_mode);
#else
    DWORD attr = GetFileAttributesW(wstr().c_str());
    return attr != INVALID_FILE_ATTRIBUTES &&
           (attr & FILE_ATTRIBUTE_DIRECTORY) == 0;
#endif
  }

  bool is_absolute() const { return absolute_; }

  Path make_absolute() const {
#if defined(__linux)
    char temp[PATH_MAX];
    if (realpath(str().c_str(), temp) == nullptr) {
      throw std::runtime_error("Error in make_absolute(): " +
                               Util::to_string(strerror(errno)));
    }
    return Path(temp);
#else
    std::wstring value = wstr(), out(MAX_PATH, '\0');
    DWORD length = GetFullPathNameW(value.c_str(), MAX_PATH, &out[0], NULL);
    if (length == 0) {
      throw std::runtime_error("Error in make_absolute(): " +
                               Util::to_string(GetLastError()));
    }
    std::wstring temp = out.substr(0, length);
    return Path(std::string(temp.begin(), temp.end()));
#endif
  }

  std::string file_name() const {
    if (path_.empty() || !is_file()) {
      return std::string();
    }
    return path_[path_.size() - 1];
  }

  std::string folder_name() const {
    if (path_.empty() || !is_directory()) {
      return std::string();
    }
    return path_[path_.size() - 1];
  }

  std::string name() const {
    const auto &name = file_name();
    auto pos = name.find_last_of('.');
    return pos == std::string::npos ? std::string() : name.substr(0, pos);
  }

  std::string extension() const {
    const auto &name = file_name();
    auto pos = name.find_last_of('.');
    return pos == std::string::npos ? std::string() : name.substr(pos + 1);
  }

  size_t length() const { return path_.size(); }

  size_t file_size() const {
#if defined(__linux)
    struct stat sb {};
    if (stat(str().c_str(), &sb) != 0) {
      throw std::runtime_error("Error in file_size(): " + str());
    }
#else
    struct _stati64 sb {};
    if (_wstati64(wstr().c_str(), &sb) != 0) {
      throw std::runtime_error("Error in file_size(): " + str());
    }
#endif
    return static_cast<size_t>(sb.st_size);
  }

  Path parent_path() const {
    Path result;
    result.absolute_ = absolute_;
    if (path_.empty()) {
      if (!absolute_) {
        result.path_.emplace_back("..");
      }
    } else {
      for (size_t i = 0; i < path_.size() - 1; ++i) {
        result.path_.push_back(path_[i]);
      }
    }
    return result;
  }

  std::string str(PathType type = KNative) const {
    std::stringstream ss;
    if (type_ == kPosix && absolute_) {
      ss << "/";
    }
    for (size_t i = 0; i < path_.size(); ++i) {
      ss << path_[i];
      if (i + 1 < path_.size()) {
        if (type == kPosix) {
          ss << '/';
        } else {
          ss << '\\';
        }
      }
    }
    return ss.str();
  }

  std::wstring wstr(PathType type = KNative) const {
    const auto &temp = str(type);
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
#if defined(__linux)
    return std::remove(str().c_str()) == 0;
#else
    return DeleteFileW(wstr().c_str()) != 0;
#endif
  }

  bool resize_file(size_t target_length) {
#if defined(__linux)
    return ::truncate(str().c_str(), (off_t)target_length) == 0;
#else
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
#endif
  }

  static Path cwd() {
#if defined(__linux)
    char temp[PATH_MAX];
    if (::getcwd(temp, PATH_MAX) == nullptr) {
      throw std::runtime_error("Error in cwd(): " +
                               Util::to_string(strerror(errno)));
    }
    return Path(temp);
#else
    std::wstring temp(MAX_PATH, '\0');
    if (!_wgetcwd(&temp[0], MAX_PATH)) {
      throw std::runtime_error("Error in cwd(): " +
                               Util::to_string(GetLastError()));
    }
    return Path(std::string(temp.begin(), temp.end()));
#endif
  }

  Path &operator=(const Path &other) = default;

  Path operator/(const Path &other) const {
    if (other.absolute_) {
      throw std::runtime_error("Error in operator/: expected a relative path!");
    }
    if (type_ != other.type_) {
      throw std::runtime_error(
          "Error in operator/: expected a path of the same type!");
    }
    Path result(*this);
    for (const auto &path : other.path_) {
      result.path_.push_back(path);
    }
    return result;
  }
  Path operator+(const Path &other) const {
    if (other.absolute_) {
      throw std::runtime_error("Error in operator+: expected a relative path!");
    }
    if (type_ != other.type_) {
      throw std::runtime_error(
          "Error in operator+: expected a path of the same type!");
    }
    Path result(*this);
    for (const auto &path : other.path_) {
      result.path_.push_back(path);
    }
    return result;
  }

  bool operator==(const Path &other) const { return other.path_ == path_; }
  bool operator!=(const Path &other) const { return other.path_ != path_; }

  friend std::ostream &operator<<(std::ostream &os, const Path &path) {
    os << path.str();
    return os;
  }

 private:
  PathType type_{KNative};
  std::vector<std::string> path_;
  bool absolute_{false};
};

namespace Util {

inline bool make_directory(const Path &path) {
#if defined(__linux)
  return mkdir(path.str().c_str(), S_IRUSR | S_IWUSR | S_IXUSR) == 0;
#else
  return CreateDirectoryW(path.wstr().c_str(), NULL) != 0;
#endif
}

inline bool make_directory(const std::string &path) {
  return make_directory(Path(path));
}

}  // namespace Util

class Timer {
 public:
  Timer() {
#if defined(__linux)
    tstart_ = system_clock::now();
#else
    QueryPerformanceFrequency(&tfrequency_);
    QueryPerformanceCounter(&tstart_);
#endif
  }

  void start() {
#if defined(__linux)
    tstart_ = system_clock::now();
#else
    QueryPerformanceCounter(&tstart_);
#endif
  }

  double get_microsecond() {
#if defined(__linux)
    tend_ = system_clock::now();
    return duration_cast<microseconds>(tend_ - tstart_).count();
#else
    QueryPerformanceCounter(&tend_);
    return 1000000.0 * (tend_.QuadPart - tstart_.QuadPart) /
           tfrequency_.QuadPart;
#endif
  }
  double get_millisecond() { return 0.001 * get_microsecond(); }
  double get_second() { return 0.000001 * get_microsecond(); }

  static tm get_compile_time() {
    int sec, min, hour, day, month, year;
    char s_month[5], month_names[] = "JanFebMarAprMayJunJulAugSepOctNovDec";
    sscanf(__TIME__, "%d:%d:%d", &hour, &min, &sec);
    sscanf(__DATE__, "%s %d %d", s_month, &day, &year);
    month = (strstr(month_names, s_month) - month_names) / 3;
    return tm{sec, min, hour, day, month, year - 1900};
  }

  static std::string get_compile_time_str() {
    tm ltm = get_compile_time();
    return Util::find_replace(std::string(asctime(&ltm)).substr(4), "\n", "");
  }

  static tm get_current_time() {
    time_t now = time(nullptr);
    return *localtime(&now);
  }

  static std::string get_current_time_str() {
    tm ltm = get_current_time();
    return Util::find_replace(std::string(asctime(&ltm)).substr(4), "\n", "");
  }

  static bool is_expired(int year = 0, int mon = 3, int day = 0) {
    tm compile_tm = get_compile_time(), current_tm = get_current_time();
    int year_gap = current_tm.tm_year - compile_tm.tm_year;
    int mon_gap = current_tm.tm_mon - compile_tm.tm_mon;
    int day_gap = current_tm.tm_mday - compile_tm.tm_mday;
    if (year < year_gap) {
      return true;
    } else if (year > year_gap) {
      return false;
    } else {
      if (mon < mon_gap) {
        return true;
      } else if (mon > mon_gap) {
        return false;
      } else {
        return day < day_gap;
      }
    }
  }

 private:
#if defined(__linux)
  time_point<system_clock> tstart_, tend_;
#else
  LARGE_INTEGER tstart_, tend_, tfrequency_;
#endif
};

class Process {
 public:
  Process(int slice, int total, const std::string &prefix = "") {
    slice_ = slice;
    total_ = total;
    prefix_ = prefix;
    time_start_ = false;
  }

  void update(int current, std::ostream *os, int mode = 0) {
    *os << prefix_ << "[";
    int pos = slice_ * (current + 1) / total_;
    for (int i = 0; i < slice_; ++i) {
      if (i < pos) {
        *os << "=";
      } else if (i == pos) {
        *os << ">";
      } else {
        *os << " ";
      }
    }
    *os << "] ";
    if (mode == 0) {
      *os << "(" << static_cast<int>(100.f * (current + 1) / total_) << "%)";
    } else if (mode == 1) {
      *os << "(" << current + 1 << "/" << total_ << ")";
    }
    if (time_start_) {
      auto left = static_cast<int>(timer_.get_millisecond() *
                                   (total_ - current) / current);
      int left_sec = left / 1000;
      int sec = left_sec % 60;
      int left_min = left_sec / 60;
      int min = left_min % 60;
      int hour = left_min / 60;
      *os << " eta: " << Util::format_int(hour, 2, '0') << ":"
          << Util::format_int(min, 2, '0') << ":"
          << Util::format_int(sec, 2, '0');
    } else {
      timer_.start();
      time_start_ = true;
    }

    if (pos < slice_) {
      *os << "\r";
    } else {
      *os << "\n";
    }
    *os << std::flush;
  }

 private:
  int slice_, total_;
  bool time_start_;
  std::string prefix_;
  Timer timer_;
};

#endif  // SHADOW_UTIL_UTIL_HPP
