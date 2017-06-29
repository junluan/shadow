#include "params.hpp"
#include "util/log.hpp"

namespace Shadow {

ArgumentHelper::ArgumentHelper(const shadow::OpParam& def) {
  for (auto& arg : def.arg()) {
    CHECK(!arg_map_.count(arg.name()))
        << "Duplicated argument name: " << arg.name()
        << " found in operator def: " << def.name();
    arg_map_[arg.name()] = &arg;
  }
}

bool ArgumentHelper::HasArgument(const std::string& name) const {
  return static_cast<bool>(arg_map_.count(name));
}

#define INSTANTIATE_GET_SINGLE_ARGUMENT(T, fieldname)                      \
  template <>                                                              \
  T ArgumentHelper::GetSingleArgument<T>(const std::string& name,          \
                                         const T& default_value) const {   \
    if (arg_map_.count(name) == 0) {                                       \
      return default_value;                                                \
    }                                                                      \
    CHECK(arg_map_.at(name)->has_##fieldname());                           \
    return arg_map_.at(name)->fieldname();                                 \
  }                                                                        \
  template <>                                                              \
  bool ArgumentHelper::HasSingleArgumentOfType<T>(const std::string& name) \
      const {                                                              \
    if (arg_map_.count(name) == 0) {                                       \
      return false;                                                        \
    }                                                                      \
    return arg_map_.at(name)->has_##fieldname();                           \
  }

INSTANTIATE_GET_SINGLE_ARGUMENT(float, s_f);
INSTANTIATE_GET_SINGLE_ARGUMENT(int, s_i);
INSTANTIATE_GET_SINGLE_ARGUMENT(bool, s_i);
INSTANTIATE_GET_SINGLE_ARGUMENT(std::string, s_s);
#undef INSTANTIATE_GET_SINGLE_ARGUMENT

#define INSTANTIATE_GET_REPEATED_ARGUMENT(T, fieldname)                     \
  template <>                                                               \
  std::vector<T> ArgumentHelper::GetRepeatedArgument<T>(                    \
      const std::string& name, const std::vector<T>& default_value) const { \
    if (arg_map_.count(name) == 0) {                                        \
      return default_value;                                                 \
    }                                                                       \
    std::vector<T> values;                                                  \
    for (const auto v : arg_map_.at(name)->fieldname()) {                   \
      values.push_back(v);                                                  \
    }                                                                       \
    return values;                                                          \
  }

INSTANTIATE_GET_REPEATED_ARGUMENT(float, v_f);
INSTANTIATE_GET_REPEATED_ARGUMENT(int, v_i);
INSTANTIATE_GET_REPEATED_ARGUMENT(bool, v_i);
INSTANTIATE_GET_REPEATED_ARGUMENT(std::string, v_s);
#undef INSTANTIATE_GET_REPEATED_ARGUMENT

}  // namespace Shadow
