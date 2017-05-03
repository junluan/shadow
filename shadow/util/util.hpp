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
  return std::to_string(val);
}

inline std::string format_int(int n, int width, char pad = ' ') {
  std::stringstream out;
  out << std::setw(width) << std::setfill(pad) << n;
  return out.str();
}

inline std::string format_process(int current, int total) {
  int digits = total > 0 ? std::log10(total) + 1 : 1;
  std::stringstream out;
  out << format_int(current, digits) << " / " << total;
  return out.str();
}

template <typename Dtype>
inline std::string format_vector(const std::vector<Dtype> &vector,
                                 const std::string &split = ",",
                                 const std::string &prefix = "",
                                 const std::string &postfix = "") {
  std::stringstream out;
  out << prefix;
  for (int i = 0; i < vector.size() - 1; ++i) {
    out << vector.at(i) << split;
  }
  if (vector.size() > 1) {
    out << vector.at(vector.size() - 1);
  }
  out << postfix;
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
  if (!file.is_open()) {
    throw std::runtime_error("Load image list file error!");
  }

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
    std::cerr << "Can't open text file " << filename;
    return std::string();
  }

  std::stringstream result;
  std::string tmp;
  while (std::getline(file, tmp)) result << tmp << std::endl;
  file.close();
  return result.str();
}

}  // namespace Util

#if defined(__linux)
#include <linux/limits.h>
#include <unistd.h>
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

  Path() : type_(KNative), absolute_(false) {}
  Path(const Path &path)
      : type_(path.type_), path_(path.path_), absolute_(path.absolute_) {}
  explicit Path(const char *str) { set(str); }
  explicit Path(const std::string &str) { set(str); }

  bool is_empty() const { return path_.empty(); }

  bool is_exist() const {
#if defined(__linux)
    struct stat sb;
    return stat(str().c_str(), &sb) == 0;
#else
    return GetFileAttributesW(wstr().c_str()) != INVALID_FILE_ATTRIBUTES;
#endif
  }

  bool is_directory() const {
#if defined(__linux)
    struct stat sb;
    if (stat(str().c_str(), &sb)) return false;
    return S_ISDIR(sb.st_mode);
#else
    DWORD attr = GetFileAttributesW(wstr().c_str());
    return attr != INVALID_FILE_ATTRIBUTES &&
           (attr & FILE_ATTRIBUTE_DIRECTORY) != 0;
#endif
  }

  bool is_file() const {
#if defined(__linux)
    struct stat sb;
    if (stat(str().c_str(), &sb)) return false;
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
    if (realpath(str().c_str(), temp) == NULL) {
      throw std::runtime_error("Error in make_absolute(): " +
                               std::string(strerror(errno)));
    }
    return Path(temp);
#else
    std::wstring value = wstr(), out(MAX_PATH, '\0');
    DWORD length = GetFullPathNameW(value.c_str(), MAX_PATH, &out[0], NULL);
    if (length == 0) {
      throw std::runtime_error("Error in make_absolute(): " +
                               std::to_string(GetLastError()));
    }
    std::wstring temp = out.substr(0, length);
    return Path(std::string(temp.begin(), temp.end()));
#endif
  }

  std::string file_name() const {
    if (path_.empty() || !is_file()) return "";
    return path_[path_.size() - 1];
  }

  std::string folder_name() const {
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
#if defined(__linux)
    struct stat sb;
    if (stat(str().c_str(), &sb) != 0) {
      throw std::runtime_error("Error in file_size(): " + str());
    }
#else
    struct _stati64 sb;
    if (_wstati64(wstr().c_str(), &sb) != 0) {
      throw std::runtime_error("Error in file_size(): " + str());
    }
#endif
    return (size_t)sb.st_size;
  }

  Path parent_path() const {
    Path result;
    result.absolute_ = absolute_;
    if (path_.empty()) {
      if (!absolute_) result.path_.push_back("..");
    } else {
      for (size_t i = 0; i < path_.size() - 1; ++i) {
        result.path_.push_back(path_[i]);
      }
    }
    return result;
  }

  std::string str(PathType type = KNative) const {
    std::stringstream oss;
    if (type_ == kPosix && absolute_) oss << "/";
    for (size_t i = 0; i < path_.size(); ++i) {
      oss << path_[i];
      if (i + 1 < path_.size()) {
        if (type == kPosix) {
          oss << '/';
        } else {
          oss << '\\';
        }
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
    if (::getcwd(temp, PATH_MAX) == NULL) {
      throw std::runtime_error("Error in cwd(): " +
                               std::string(strerror(errno)));
    }
    return Path(temp);
#else
    std::wstring temp(MAX_PATH, '\0');
    if (!_wgetcwd(&temp[0], MAX_PATH)) {
      throw std::runtime_error("Error in cwd(): " +
                               std::to_string(GetLastError()));
    }
    return Path(std::string(temp.begin(), temp.end()));
#endif
  }

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

#include <ctime>
class Timer {
 public:
#if defined(__linux)
  Timer() : ts(clock()) {}

  void start() { ts = clock(); }
  double get_second() {
    return static_cast<double>(clock() - ts) / CLOCKS_PER_SEC;
  }
  double get_millisecond() {
    return 1000.0 * static_cast<double>(clock() - ts) / CLOCKS_PER_SEC;
  }

#else
  Timer() {
    QueryPerformanceFrequency(&tfrequency_);
    QueryPerformanceCounter(&tstart_);
  }

  void start() { QueryPerformanceCounter(&tstart_); }
  double get_second() {
    QueryPerformanceCounter(&tend_);
    return static_cast<double>(tend_.QuadPart - tstart_.QuadPart) /
           tfrequency_.QuadPart;
  }
  double get_millisecond() {
    QueryPerformanceCounter(&tend_);
    return 1000.0 * static_cast<double>(tend_.QuadPart - tstart_.QuadPart) /
           tfrequency_.QuadPart;
  }
#endif

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
    time_t now = time(0);
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
  clock_t ts;

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
    time_start_ = 0;
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
      int left = timer_.get_millisecond() * (total_ - current) / current;
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
      time_start_ = 1;
    }

    if (pos < slice_) {
      *os << "\r";
    } else {
      *os << "\n";
    }
    *os << std::flush;
  }

 private:
  int slice_, total_, time_start_;
  std::string prefix_;
  Timer timer_;
};

#endif  // SHADOW_UTIL_UTIL_HPP
