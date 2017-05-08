#ifndef SHADOW_UTIL_LOG_HPP
#define SHADOW_UTIL_LOG_HPP

#if defined(USE_GLog)
#if !defined(__linux)
#define GOOGLE_GLOG_DLL_DECL
#define GLOG_NO_ABBREVIATED_SEVERITIES
#endif
#include <glog/logging.h>

#else
#include <iostream>
#include <string>
#define LOG_INFO LogMessage("INFO", __FILE__, __LINE__)
#define LOG_WARNING LogMessage("WARNING", __FILE__, __LINE__)
#define LOG_ERROR LogMessage("ERROR", __FILE__, __LINE__)
#define LOG_FATAL LogMessage("FATAL", __FILE__, __LINE__)

#define LOG(severity) LOG_##severity.stream()
#define LOG_IF(severity, condition) \
  !(condition) ? (void)0 : LogMessageVoidify() & LOG(severity)

#define CHECK(condition) \
  if (!(condition)) LOG(FATAL) << "Check Failed: " #condition << ' '

#define CHECK_OP(condition, val1, val2)                                      \
  if (!((val1)condition(val2)))                                              \
  LOG(FATAL) << "Check Failed: " #val1 " " #condition " " #val2 " (" << val1 \
             << " vs " << val2 << ") "

#define CHECK_EQ(val1, val2) CHECK_OP(==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(!=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(<=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(<, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(>=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(>, val1, val2)

#define CHECK_NOTNULL(condition)                                          \
  ((condition) == nullptr                                                 \
   ? LOG(FATAL) << "Check failed: '" #condition << "' Must be non NULL ", \
   (condition) : (condition))

#if defined(NDEBUG)
#define DLOG(severity) true ? (void)0 : LogMessageVoidify() & LOG(severity)
#define DLOG_IF(severity, condition) \
  (true || !(condition)) ? (void)0 : LogMessageVoidify() & LOG(severity)

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

#include <cstdlib>
class LogMessage {
 public:
  LogMessage(const std::string& severity, const char* file, int line)
      : severity_(severity), log_stream_(std::cerr) {
    std::string file_str(file);
    std::string file_name = file_str.substr(file_str.find_last_of("/\\") + 1);
    log_stream_ << severity << ": " << file_name << ":" << line << "] ";
  }
  ~LogMessage() {
    if (!severity_.compare("FATAL")) {
      abort();
    }
    log_stream_ << '\n';
  }
  std::ostream& stream() { return log_stream_; }

 private:
  LogMessage(const LogMessage&);
  void operator=(const LogMessage&);

  std::string severity_;
  std::ostream& log_stream_;
};

class LogMessageVoidify {
 public:
  LogMessageVoidify() {}
  void operator&(std::ostream&) {}
};
#endif

#endif  // SHADOW_UTIL_LOG_HPP