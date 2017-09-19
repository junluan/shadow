#ifndef SHADOW_UTIL_LOG_HPP
#define SHADOW_UTIL_LOG_HPP

#if defined(USE_GLog)
#if defined(_WIN32)
#define GOOGLE_GLOG_DLL_DECL
#define GLOG_NO_ABBREVIATED_SEVERITIES
#endif
#include <glog/logging.h>

#else
#include <iostream>
#include <sstream>
#include <string>
#define LOG_INFO LogMessage("INFO", __FILE__, __LINE__)
#define LOG_WARNING LogMessage("WARNING", __FILE__, __LINE__)
#define LOG_ERROR LogMessage("ERROR", __FILE__, __LINE__)
#define LOG_FATAL LogMessage("FATAL", __FILE__, __LINE__)

#define LOG(severity) LOG_##severity.stream()
#define LOG_IF(severity, condition) \
  if ((condition)) LOG(severity)

#define CHECK_NOTNULL(condition) \
  if ((condition) == nullptr)    \
  LOG(FATAL) << "Check Failed: '" #condition << "' Must be non NULL "

#define CHECK(condition) \
  if (!(condition)) LOG(FATAL) << "Check Failed: " #condition << " "

#define CHECK_OP(condition, val1, val2)                                        \
  if (!((val1)condition(val2)))                                                \
  LOG(FATAL) << "Check Failed: " #val1 " " #condition " " #val2 " (" << (val1) \
             << " vs " << (val2) << ") "

#define CHECK_EQ(val1, val2) CHECK_OP(==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(!=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(<=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(<, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(>=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(>, val1, val2)

#if defined(NDEBUG)
#define DLOG(severity) \
  while (false) LOG(severity)
#define DLOG_IF(severity, condition) \
  while (false) LOG_IF(severity, condition)

#define DCHECK(condition) \
  while (false) CHECK(condition)
#define DCHECK_EQ(val1, val2) \
  while (false) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) \
  while (false) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) \
  while (false) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) \
  while (false) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) \
  while (false) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) \
  while (false) CHECK_GT(val1, val2)

#else
#define DLOG(severity) LOG(severity)
#define DLOG_IF(severity, condition) LOG_IF(severity, condition)

#define DCHECK(condition) CHECK(condition)
#define DCHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define DCHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define DCHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define DCHECK_GE(val1, val2) CHECK_GE(val1, val2)
#define DCHECK_GT(val1, val2) CHECK_GT(val1, val2)
#endif

#if defined(__ANDROID__) || defined(ANDROID)
#include <android/log.h>
#endif

#include <cstdlib>
class LogMessage {
 public:
  LogMessage(const std::string& severity, const char* file, int line)
      : severity_(severity) {
    std::string file_str(file);
    std::string file_name = file_str.substr(file_str.find_last_of("/\\") + 1);
    stream_ << severity << ": " << file_name << ":" << line << "] ";
  }
  ~LogMessage() {
    stream_ << '\n';
#if defined(__ANDROID__) || defined(ANDROID)
    if (severity_ == "INFO") {
      __android_log_print(ANDROID_LOG_INFO, tag_, "%s", stream_.str().c_str());
    } else if (severity_ == "WARNING") {
      __android_log_print(ANDROID_LOG_WARN, tag_, "%s", stream_.str().c_str());
    } else if (severity_ == "ERROR") {
      __android_log_print(ANDROID_LOG_ERROR, tag_, "%s", stream_.str().c_str());
    } else if (severity_ == "FATAL") {
      __android_log_print(ANDROID_LOG_FATAL, tag_, "%s", stream_.str().c_str());
    }
#else
    std::cerr << stream_.str();
#endif
    if (severity_ == "FATAL") {
      abort();
    }
  }
  std::stringstream& stream() { return stream_; }

  LogMessage(const LogMessage&) = delete;
  void operator=(const LogMessage&) = delete;

 private:
#if defined(__ANDROID__) || defined(ANDROID)
  const char* tag_ = "native";
#else
  const char* tag_ = "";
#endif

  std::string severity_;
  std::stringstream stream_;
};
#endif

#endif  // SHADOW_UTIL_LOG_HPP
