#ifndef SHADOW_TOOLS_TRANSFORMER_HPP
#define SHADOW_TOOLS_TRANSFORMER_HPP

#include "proto/shadow.pb.h"

#include "util/io.hpp"
#include "util/log.hpp"
#include "util/util.hpp"

namespace Shadow {

#define INSTANTIATE_SET_SINGLE_ARGUMENT(T, fieldname)                    \
  static void set_##fieldname(shadow::OpParam* op_param,                 \
                              const std::string& name, const T& value) { \
    auto shadow_arg = op_param->add_arg();                               \
    shadow_arg->set_name(name);                                          \
    shadow_arg->set_##fieldname(value);                                  \
  }

INSTANTIATE_SET_SINGLE_ARGUMENT(float, s_f);
INSTANTIATE_SET_SINGLE_ARGUMENT(int, s_i);
INSTANTIATE_SET_SINGLE_ARGUMENT(unsigned int, s_i);
INSTANTIATE_SET_SINGLE_ARGUMENT(bool, s_i);
INSTANTIATE_SET_SINGLE_ARGUMENT(std::string, s_s);
#undef INSTANTIATE_SET_SINGLE_ARGUMENT

#define INSTANTIATE_SET_REPEATED_ARGUMENT(T, fieldname)      \
  static void set_##fieldname(shadow::OpParam* op_param,     \
                              const std::string& name,       \
                              const std::vector<T>& value) { \
    auto shadow_arg = op_param->add_arg();                   \
    shadow_arg->set_name(name);                              \
    for (const auto v : value) {                             \
      shadow_arg->add_##fieldname(v);                        \
    }                                                        \
  }

INSTANTIATE_SET_REPEATED_ARGUMENT(float, v_f);
INSTANTIATE_SET_REPEATED_ARGUMENT(int, v_i);
INSTANTIATE_SET_REPEATED_ARGUMENT(unsigned int, v_i);
INSTANTIATE_SET_REPEATED_ARGUMENT(bool, v_i);
INSTANTIATE_SET_REPEATED_ARGUMENT(std::string, v_s);
#undef INSTANTIATE_SET_REPEATED_ARGUMENT

const std::string ConvertCustom(const shadow::NetParam& shadow_net);

void WriteDefines(const shadow::NetParam& shadow_net, const std::string& root,
                  const std::string& model_name);

void WriteWeights(const shadow::NetParam& shadow_net, const std::string& root,
                  const std::string& model_name);

void WriteProtoToFiles(const shadow::NetParam& shadow_net,
                       const std::string& root, const std::string& model_name);

void WriteProtoToBinary(const IO::Message& proto, const std::string& root,
                        const std::string& model_name);

}  // namespace Shadow

#endif  // SHADOW_TOOLS_TRANSFORMER_HPP
