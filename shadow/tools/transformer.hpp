#ifndef SHADOW_TOOLS_TRANSFORMER_HPP
#define SHADOW_TOOLS_TRANSFORMER_HPP

#include "proto/shadow.pb.h"

#include "util/io.hpp"
#include "util/log.hpp"
#include "util/util.hpp"

namespace Shadow {

const std::string ConvertCustom(const shadow::NetParam& shadow_net);

void WriteDefines(const shadow::NetParam& shadow_net, const std::string& root,
                  const std::string& model_name);

void WriteWeights(const shadow::NetParam& shadow_net, const std::string& root,
                  const std::string& model_name);

void WriteProtoToFiles(const shadow::NetParam& shadow_net,
                       const std::string& root, const std::string& model_name);

}  // namespace Shadow

#endif  // SHADOW_TOOLS_TRANSFORMER_HPP
