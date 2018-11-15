#include "params.hpp"

#include "util/log.hpp"

namespace Shadow {

ArgumentHelper::ArgumentHelper(const shadow::NetParam& def) {
  for (auto& arg : def.arg()) {
    CHECK(!arg_map_.count(arg.name()))
        << "Duplicated argument name: " << arg.name()
        << " found in network def: " << def.name();
    arg_map_[arg.name()] = arg;
  }
}

ArgumentHelper::ArgumentHelper(const shadow::OpParam& def) {
  for (auto& arg : def.arg()) {
    CHECK(!arg_map_.count(arg.name()))
        << "Duplicated argument name: " << arg.name()
        << " found in operator def: " << def.name();
    arg_map_[arg.name()] = arg;
  }
}

bool ArgumentHelper::HasArgument(const std::string& name) const {
  return static_cast<bool>(arg_map_.count(name));
}

#define INSTANTIATE_GET_SINGLE_ARGUMENT(T, fieldname)                    \
  template <>                                                            \
  T ArgumentHelper::GetSingleArgument<T>(const std::string& name,        \
                                         const T& default_value) const { \
    if (arg_map_.count(name) == 0) {                                     \
      return default_value;                                              \
    }                                                                    \
    CHECK(arg_map_.at(name).has_##fieldname());                          \
    return arg_map_.at(name).fieldname();                                \
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
    for (const auto v : arg_map_.at(name).fieldname()) {                    \
      values.push_back(v);                                                  \
    }                                                                       \
    return values;                                                          \
  }

INSTANTIATE_GET_REPEATED_ARGUMENT(float, v_f);
INSTANTIATE_GET_REPEATED_ARGUMENT(int, v_i);
INSTANTIATE_GET_REPEATED_ARGUMENT(bool, v_i);
INSTANTIATE_GET_REPEATED_ARGUMENT(std::string, v_s);
#undef INSTANTIATE_GET_REPEATED_ARGUMENT

#define INSTANTIATE_SET_SINGLE_ARGUMENT(fieldname, P, T)                    \
  void set_##fieldname(P* param, const std::string& name, const T& value) { \
    auto* arg = param->add_arg();                                           \
    arg->set_name(name);                                                    \
    arg->set_##fieldname(value);                                            \
  }

INSTANTIATE_SET_SINGLE_ARGUMENT(s_f, shadow::NetParam, float);
INSTANTIATE_SET_SINGLE_ARGUMENT(s_i, shadow::NetParam, int);
INSTANTIATE_SET_SINGLE_ARGUMENT(s_i, shadow::NetParam, bool);
INSTANTIATE_SET_SINGLE_ARGUMENT(s_s, shadow::NetParam, std::string);
INSTANTIATE_SET_SINGLE_ARGUMENT(s_f, shadow::OpParam, float);
INSTANTIATE_SET_SINGLE_ARGUMENT(s_i, shadow::OpParam, int);
INSTANTIATE_SET_SINGLE_ARGUMENT(s_i, shadow::OpParam, bool);
INSTANTIATE_SET_SINGLE_ARGUMENT(s_s, shadow::OpParam, std::string);
#undef INSTANTIATE_SET_SINGLE_ARGUMENT

#define INSTANTIATE_SET_REPEATED_ARGUMENT(fieldname, P, T) \
  void set_##fieldname(P* param, const std::string& name,  \
                       const std::vector<T>& value) {      \
    auto* arg = param->add_arg();                          \
    arg->set_name(name);                                   \
    for (const auto v : value) {                           \
      arg->add_##fieldname(v);                             \
    }                                                      \
  }

INSTANTIATE_SET_REPEATED_ARGUMENT(v_f, shadow::NetParam, float);
INSTANTIATE_SET_REPEATED_ARGUMENT(v_i, shadow::NetParam, int);
INSTANTIATE_SET_REPEATED_ARGUMENT(v_i, shadow::NetParam, bool);
INSTANTIATE_SET_REPEATED_ARGUMENT(v_s, shadow::NetParam, std::string);
INSTANTIATE_SET_REPEATED_ARGUMENT(v_f, shadow::OpParam, float);
INSTANTIATE_SET_REPEATED_ARGUMENT(v_i, shadow::OpParam, int);
INSTANTIATE_SET_REPEATED_ARGUMENT(v_i, shadow::OpParam, bool);
INSTANTIATE_SET_REPEATED_ARGUMENT(v_s, shadow::OpParam, std::string);
#undef INSTANTIATE_SET_REPEATED_ARGUMENT

}  // namespace Shadow
