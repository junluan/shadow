#ifndef SHADOW_CORE_JSON_HELPER_HPP
#define SHADOW_CORE_JSON_HELPER_HPP

#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "core/params.hpp"

namespace Shadow {

#define JSON_DICT_ENTRY(out, prefix, T)                               \
  out << "\"" #T "\":";                                               \
  json_forwarder<std::decay<decltype((prefix).T)>::type>::encode(out, \
                                                                 (prefix).T)
#define JSON_DICT_ENTRIES2(out, prefix, T1, T2) \
  JSON_DICT_ENTRY(out, prefix, T1);             \
  out << ",";                                   \
  JSON_DICT_ENTRY(out, prefix, T2)
#define JSON_DICT_ENTRIES3(out, prefix, T1, T2, T3) \
  JSON_DICT_ENTRIES2(out, prefix, T1, T2);          \
  out << ",";                                       \
  JSON_DICT_ENTRY(out, prefix, T3)
#define JSON_DICT_ENTRIES4(out, prefix, T1, T2, T3, T4) \
  JSON_DICT_ENTRIES3(out, prefix, T1, T2, T3);          \
  out << ",";                                           \
  JSON_DICT_ENTRY(out, prefix, T4)
#define JSON_DICT_ENTRIES5(out, prefix, T1, T2, T3, T4, T5) \
  JSON_DICT_ENTRIES4(out, prefix, T1, T2, T3, T4);          \
  out << ",";                                               \
  JSON_DICT_ENTRY(out, prefix, T5)
#define JSON_DICT_ENTRIES6(out, prefix, T1, T2, T3, T4, T5, T6) \
  JSON_DICT_ENTRIES5(out, prefix, T1, T2, T3, T4, T5);          \
  out << ",";                                                   \
  JSON_DICT_ENTRY(out, prefix, T6)
#define JSON_DICT_ENTRIES7(out, prefix, T1, T2, T3, T4, T5, T6, T7) \
  JSON_DICT_ENTRIES6(out, prefix, T1, T2, T3, T4, T5, T6);          \
  out << ",";                                                       \
  JSON_DICT_ENTRY(out, prefix, T7)
#define JSON_DICT_ENTRIES8(out, prefix, T1, T2, T3, T4, T5, T6, T7, T8) \
  JSON_DICT_ENTRIES7(out, prefix, T1, T2, T3, T4, T5, T6, T7);          \
  out << ",";                                                           \
  JSON_DICT_ENTRY(out, prefix, T8)

#define JSON_ENTRIES_GET_MACRO(ph1, ph2, ph3, ph4, ph5, ph6, ph7, ph8, NAME, \
                               ...)                                          \
  NAME
// workaround due to the way VC handles "..."
#define JSON_ENTRIES_GET_MACRO_(tuple) JSON_ENTRIES_GET_MACRO tuple
#define JSON_DICT_ENTRIES(out, prefix, ...)                        \
  out << "{";                                                      \
  JSON_ENTRIES_GET_MACRO_(                                         \
      (__VA_ARGS__, JSON_DICT_ENTRIES8, JSON_DICT_ENTRIES7,        \
       JSON_DICT_ENTRIES6, JSON_DICT_ENTRIES5, JSON_DICT_ENTRIES4, \
       JSON_DICT_ENTRIES3, JSON_DICT_ENTRIES2, JSON_DICT_ENTRY))   \
  (out, prefix, __VA_ARGS__);                                      \
  out << "}"

#define DEFINE_JSON_SERIALIZATION(...)            \
  const std::string json_state() const {          \
    std::stringstream out;                        \
    JSON_DICT_ENTRIES(out, *this, __VA_ARGS__);   \
    return out.str();                             \
  }                                               \
  void json_state(std::stringstream& out) const { \
    JSON_DICT_ENTRIES(out, *this, __VA_ARGS__);   \
  }

template <typename>
class json_forwarder;

template <typename T>
inline void json_encode(std::stringstream& out, const T& t) {
  out << t;
}

inline void json_encode(std::stringstream& out, const std::string& t) {
  out << "\"" << t << "\"";
}

inline void json_encode(std::stringstream& out, const char* t) {
  out << "\"" << t << "\"";
}

template <typename T>
inline void json_encode_iterable(std::stringstream& out, const T& t) {
  out << "[";
  for (auto it = t.begin(); it != t.end(); ++it) {
    json_forwarder<typename std::decay<decltype(*it)>::type>::encode(out, *it);
    if (std::next(it) != t.end()) {
      out << ",";
    }
  }
  out << "]";
}

template <typename T>
inline void json_encode_map(std::stringstream& out, const T& t) {
  out << "{";
  for (auto it = t.begin(); it != t.end(); ++it) {
    json_forwarder<typename std::decay<decltype(it->first)>::type>::encode(
        out, it->first);
    out << ":";
    json_forwarder<typename std::decay<decltype(it->second)>::type>::encode(
        out, it->second);
    if (std::next(it) != t.end()) {
      out << ",";
    }
  }
  out << "}";
}

template <typename T>
inline void json_encode(std::stringstream& out, const std::vector<T>& t) {
  json_encode_iterable(out, t);
}

template <typename T>
inline void json_encode(std::stringstream& out, const std::set<T>& t) {
  json_encode_iterable(out, t);
}

template <typename T1, typename T2>
inline void json_encode(std::stringstream& out,
                        const std::unordered_map<T1, T2>& t) {
  json_encode_map(out, t);
}

template <typename T1, typename T2>
inline void json_encode(std::stringstream& out, const std::map<T1, T2>& t) {
  json_encode_map(out, t);
}

template <typename T>
inline void json_encode(std::stringstream& out, const std::shared_ptr<T>& t) {
  json_encode(out, *t);
}

inline void json_encode(std::stringstream& out, const shadow::OpParam& t) {
  out << "{";
  for (int n = 0; n < t.arg_size(); ++n) {
    const auto& arg = t.arg(n);
    json_encode(out, arg.name());
    out << ":";
    if (arg.has_s_f()) {
      json_encode(out, arg.s_f());
    } else if (arg.has_s_i()) {
      json_encode(out, arg.s_i());
    } else if (arg.has_s_s()) {
      json_encode(out, arg.s_s());
    } else if (arg.v_f_size() > 0) {
      json_encode_iterable(out, arg.v_f());
    } else if (arg.v_i_size() > 0) {
      json_encode_iterable(out, arg.v_i());
    } else if (arg.v_s_size() > 0) {
      json_encode_iterable(out, arg.v_s());
    }
    if (n < t.arg_size() - 1) {
      out << ",";
    }
  }
  out << "}";
}

template <typename T>
class json_forwarder {
 public:
  static void encode(std::stringstream& out, const T& t) {
    encode_inner(out, t, has_json_state{}, p_has_json_state{});
  }

 private:
  // check if C has C.json_state(sstream&) function
  template <typename C>
  static auto check_json_state(C*) ->
      typename std::is_same<decltype(std::declval<C>().json_state(
                                std::declval<std::stringstream&>())),
                            void>::type;

  template <typename>
  static std::false_type check_json_state(...);

  // check if C has C->json_state(sstream&) function
  template <typename C>
  static auto p_check_json_state(C*) ->
      typename std::is_same<decltype(std::declval<C>()->json_state(
                                std::declval<std::stringstream&>())),
                            void>::type;

  template <typename>
  static std::false_type p_check_json_state(...);

  typedef decltype(check_json_state<T>(0)) has_json_state;
  typedef decltype(p_check_json_state<T>(0)) p_has_json_state;

  static void encode_inner(std::stringstream& out, const T& t, std::true_type,
                           std::false_type) {
    t.json_state(out);
  }
  static void encode_inner(std::stringstream& out, const T& t, std::false_type,
                           std::true_type) {
    t->json_state(out);
  }
  static void encode_inner(std::stringstream& out, const T& t, std::true_type,
                           std::true_type) {
    t->json_state(out);
  }
  static void encode_inner(std::stringstream& out, const T& t, std::false_type,
                           std::false_type) {
    json_encode(out, t);
  }
};

}  // namespace Shadow

#endif  // SHADOW_CORE_JSON_HELPER_HPP
