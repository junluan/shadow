#ifndef SHADOW_CORE_HELPER_HPP
#define SHADOW_CORE_HELPER_HPP

#include "params.hpp"

#include <map>

namespace Shadow {

class ArgumentHelper {
 public:
  ArgumentHelper() = default;
  explicit ArgumentHelper(const shadow::NetParam &def);
  explicit ArgumentHelper(const shadow::OpParam &def);

  bool HasArgument(const std::string &name) const;

  template <typename T>
  T GetSingleArgument(const std::string &name, const T &default_value) const;
  template <typename T>
  std::vector<T> GetRepeatedArgument(
      const std::string &name,
      const std::vector<T> &default_value = std::vector<T>()) const;

  template <typename T>
  void AddSingleArgument(const std::string &name, const T &value);
  template <typename T>
  void AddRepeatedArgument(const std::string &name,
                           const std::vector<T> &value);

 private:
  std::map<std::string, shadow::Argument> arg_map_;
};

#define DECLARE_ADD_SINGLE_ARGUMENT(fieldname, T)                        \
  void add_##fieldname(shadow::NetParam *param, const std::string &name, \
                       const T &value);                                  \
  void add_##fieldname(shadow::OpParam *param, const std::string &name,  \
                       const T &value);

DECLARE_ADD_SINGLE_ARGUMENT(s_f, float);
DECLARE_ADD_SINGLE_ARGUMENT(s_i, int);
DECLARE_ADD_SINGLE_ARGUMENT(s_i, bool);
DECLARE_ADD_SINGLE_ARGUMENT(s_s, std::string);
#undef DECLARE_ADD_SINGLE_ARGUMENT

#define DECLARE_ADD_REPEATED_ARGUMENT(fieldname, T)                      \
  void add_##fieldname(shadow::NetParam *param, const std::string &name, \
                       const std::vector<T> &value);                     \
  void add_##fieldname(shadow::OpParam *param, const std::string &name,  \
                       const std::vector<T> &value);

DECLARE_ADD_REPEATED_ARGUMENT(v_f, float);
DECLARE_ADD_REPEATED_ARGUMENT(v_i, int);
DECLARE_ADD_REPEATED_ARGUMENT(v_i, bool);
DECLARE_ADD_REPEATED_ARGUMENT(v_s, std::string);
#undef DECLARE_ADD_REPEATED_ARGUMENT

}  // namespace Shadow

#endif  // SHADOW_CORE_HELPER_HPP
