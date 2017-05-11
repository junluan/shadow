#ifndef SHADOW_CORE_PARSER_HPP
#define SHADOW_CORE_PARSER_HPP

#include "json.hpp"
#include "params.hpp"

namespace Shadow {

#if !defined(USE_Protobuf)
namespace Parser {

void ParseNet(const std::string &proto_text, shadow::NetParameter *net);

void ParseCommon(const JValue &root, shadow::LayerParameter *layer);

const shadow::LayerParameter ParseActivate(const JValue &root);
const shadow::LayerParameter ParseBatchNorm(const JValue &root);
const shadow::LayerParameter ParseBias(const JValue &root);
const shadow::LayerParameter ParseConcat(const JValue &root);
const shadow::LayerParameter ParseConnected(const JValue &root);
const shadow::LayerParameter ParseConvolution(const JValue &root);
const shadow::LayerParameter ParseData(const JValue &root);
const shadow::LayerParameter ParseFlatten(const JValue &root);
const shadow::LayerParameter ParseLRN(const JValue &root);
const shadow::LayerParameter ParseNormalize(const JValue &root);
const shadow::LayerParameter ParsePermute(const JValue &root);
const shadow::LayerParameter ParsePooling(const JValue &root);
const shadow::LayerParameter ParsePriorBox(const JValue &root);
const shadow::LayerParameter ParseReorg(const JValue &root);
const shadow::LayerParameter ParseReshape(const JValue &root);
const shadow::LayerParameter ParseScale(const JValue &root);
const shadow::LayerParameter ParseSoftmax(const JValue &root);

}  // namespace Parser
#endif

}  // namespace Shadow

#endif  // SHADOW_CORE_PARSER_HPP
