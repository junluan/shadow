#include "helper.hpp"

#include "util/log.hpp"

namespace Shadow {

template <typename T>
ArgumentHelper::ArgumentHelper(const T& def) {
  for (const auto& arg : def.arg()) {
    CHECK(!HasArgument(arg.name()))
        << "Duplicated argument name: " << arg.name()
        << " found in def: " << def.name();
    arguments_[arg.name()] = arg;
  }
}

template ArgumentHelper::ArgumentHelper(const shadow::NetParam&);
template ArgumentHelper::ArgumentHelper(const shadow::OpParam&);

template <>
ArgumentHelper::ArgumentHelper(
    const std::map<std::string, shadow::Argument>& arguments) {
  arguments_ = arguments;
}

bool ArgumentHelper::HasArgument(const std::string& name) const {
  return arguments_.count(name) > 0;
}

#define INSTANTIATE_SINGLE_ARGUMENT(fieldname, T)                        \
  template <>                                                            \
  T ArgumentHelper::GetSingleArgument<T>(const std::string& name,        \
                                         const T& default_value) const { \
    if (!HasArgument(name)) {                                            \
      return default_value;                                              \
    }                                                                    \
    CHECK(arguments_.at(name).has_##fieldname());                        \
    return arguments_.at(name).fieldname();                              \
  }                                                                      \
  template <>                                                            \
  void ArgumentHelper::AddSingleArgument<T>(const std::string& name,     \
                                            const T& value) {            \
    CHECK(!HasArgument(name)) << "Duplicated argument name: " << name;   \
    arguments_[name] = {};                                               \
    arguments_[name].set_name(name);                                     \
    arguments_[name].set_##fieldname(value);                             \
  }

INSTANTIATE_SINGLE_ARGUMENT(s_f, float);
INSTANTIATE_SINGLE_ARGUMENT(s_i, int);
INSTANTIATE_SINGLE_ARGUMENT(s_i, bool);
INSTANTIATE_SINGLE_ARGUMENT(s_s, std::string);
#undef INSTANTIATE_SINGLE_ARGUMENT

#define INSTANTIATE_REPEATED_ARGUMENT(fieldname, T)                          \
  template <>                                                                \
  std::vector<T> ArgumentHelper::GetRepeatedArgument<T>(                     \
      const std::string& name, const std::vector<T>& default_value) const {  \
    if (!HasArgument(name)) {                                                \
      return default_value;                                                  \
    }                                                                        \
    std::vector<T> values;                                                   \
    for (auto v : arguments_.at(name).fieldname()) {                         \
      values.push_back(v);                                                   \
    }                                                                        \
    return values;                                                           \
  }                                                                          \
  template <>                                                                \
  void ArgumentHelper::AddRepeatedArgument<T>(const std::string& name,       \
                                              const std::vector<T>& value) { \
    CHECK(!HasArgument(name)) << "Duplicated argument name: " << name;       \
    arguments_[name] = {};                                                   \
    arguments_[name].set_name(name);                                         \
    for (auto v : value) {                                                   \
      arguments_[name].add_##fieldname(v);                                   \
    }                                                                        \
  }

INSTANTIATE_REPEATED_ARGUMENT(v_f, float);
INSTANTIATE_REPEATED_ARGUMENT(v_i, int);
INSTANTIATE_REPEATED_ARGUMENT(v_i, bool);
INSTANTIATE_REPEATED_ARGUMENT(v_s, std::string);
#undef INSTANTIATE_REPEATED_ARGUMENT

#define INSTANTIATE_ADD_SINGLE_ARGUMENT(fieldname, P, T)                    \
  void add_##fieldname(P* param, const std::string& name, const T& value) { \
    auto* arg = param->add_arg();                                           \
    arg->set_name(name);                                                    \
    arg->set_##fieldname(value);                                            \
  }

INSTANTIATE_ADD_SINGLE_ARGUMENT(s_f, shadow::NetParam, float);
INSTANTIATE_ADD_SINGLE_ARGUMENT(s_i, shadow::NetParam, int);
INSTANTIATE_ADD_SINGLE_ARGUMENT(s_i, shadow::NetParam, bool);
INSTANTIATE_ADD_SINGLE_ARGUMENT(s_s, shadow::NetParam, std::string);
INSTANTIATE_ADD_SINGLE_ARGUMENT(s_f, shadow::OpParam, float);
INSTANTIATE_ADD_SINGLE_ARGUMENT(s_i, shadow::OpParam, int);
INSTANTIATE_ADD_SINGLE_ARGUMENT(s_i, shadow::OpParam, bool);
INSTANTIATE_ADD_SINGLE_ARGUMENT(s_s, shadow::OpParam, std::string);
#undef INSTANTIATE_ADD_SINGLE_ARGUMENT

#define INSTANTIATE_ADD_REPEATED_ARGUMENT(fieldname, P, T) \
  void add_##fieldname(P* param, const std::string& name,  \
                       const std::vector<T>& value) {      \
    auto* arg = param->add_arg();                          \
    arg->set_name(name);                                   \
    for (auto v : value) {                                 \
      arg->add_##fieldname(v);                             \
    }                                                      \
  }

INSTANTIATE_ADD_REPEATED_ARGUMENT(v_f, shadow::NetParam, float);
INSTANTIATE_ADD_REPEATED_ARGUMENT(v_i, shadow::NetParam, int);
INSTANTIATE_ADD_REPEATED_ARGUMENT(v_i, shadow::NetParam, bool);
INSTANTIATE_ADD_REPEATED_ARGUMENT(v_s, shadow::NetParam, std::string);
INSTANTIATE_ADD_REPEATED_ARGUMENT(v_f, shadow::OpParam, float);
INSTANTIATE_ADD_REPEATED_ARGUMENT(v_i, shadow::OpParam, int);
INSTANTIATE_ADD_REPEATED_ARGUMENT(v_i, shadow::OpParam, bool);
INSTANTIATE_ADD_REPEATED_ARGUMENT(v_s, shadow::OpParam, std::string);
#undef INSTANTIATE_ADD_REPEATED_ARGUMENT

}  // namespace Shadow
