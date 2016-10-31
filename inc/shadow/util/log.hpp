#ifndef SHADOW_UTIL_LOG_HPP
#define SHADOW_UTIL_LOG_HPP

#if defined(USE_GLog)
#if !defined(__linux)
#define GOOGLE_GLOG_DLL_DECL
#define GLOG_NO_ABBREVIATED_SEVERITIES
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
#include <iostream>
#include <string>
#define __FILE_NAME__ \
  std::string(__FILE__).substr(std::string(__FILE__).find_last_of("/\\") + 1)

#define Info(msg)                                                            \
  {                                                                          \
    std::cerr << "Info: " << __FILE_NAME__ << ":" << __LINE__ << "] " << msg \
              << std::endl;                                                  \
  }

#define Warning(msg)                                                     \
  {                                                                      \
    std::cerr << "Warning: " << __FILE_NAME__ << ":" << __LINE__ << "] " \
              << msg << std::endl;                                       \
  }

#define Error(msg)                                                            \
  {                                                                           \
    std::cerr << "Error: " << __FILE_NAME__ << ":" << __LINE__ << "] " << msg \
              << std::endl;                                                   \
    exit(1);                                                                  \
  }

#define Fatal(msg)                                                            \
  {                                                                           \
    std::cerr << "Fatal: " << __FILE_NAME__ << ":" << __LINE__ << "] " << msg \
              << std::endl;                                                   \
    exit(1);                                                                  \
  }

#define CHECK(condition)                      \
  {                                           \
    if (!condition) {                         \
      std::stringstream out;                  \
      out << "Check Failed: " #condition " "; \
      Fatal(out.str());                       \
    }                                         \
  }
#define CHECK_OP(condition, val1, val2)                                   \
  {                                                                       \
    if (!(val1 condition val2)) {                                         \
      std::stringstream out;                                              \
      out << "Check Failed: " #val1 " " #condition " " #val2 " (" << val1 \
          << " vs " << val2 << ") ";                                      \
      Fatal(out.str());                                                   \
    }                                                                     \
  }
#define CHECK_EQ(val1, val2) CHECK_OP(==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(!=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(<=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(<, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(>=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(>, val1, val2)
#endif

#if defined(NDEBUG)
#define DInfo(msg)
#define DWarning(msg)
#define DError(msg)
#define DFatal(msg)

#else
#define DInfo(msg) Info(msg)
#define DWarning(msg) Warning(msg)
#define DError(msg) Error(msg)
#define DFatal(msg) Fatal(msg)
#endif

#endif  // SHADOW_UTIL_LOG_HPP
