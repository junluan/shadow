#ifndef SHADOW_CORE_PARSER_HPP
#define SHADOW_CORE_PARSER_HPP

#include "json.hpp"
#include "params.hpp"

namespace Shadow {

namespace Parser {

#if !defined(USE_Protobuf)
void ParseNet(const std::string &proto_text, shadow::NetParam *net);

void ParseCommon(const JValue &root, shadow::OpParam *op);

const shadow::OpParam ParseActivate(const JValue &root);
const shadow::OpParam ParseBatchNorm(const JValue &root);
const shadow::OpParam ParseBias(const JValue &root);
const shadow::OpParam ParseConcat(const JValue &root);
const shadow::OpParam ParseConnected(const JValue &root);
const shadow::OpParam ParseConvolution(const JValue &root);
const shadow::OpParam ParseData(const JValue &root);
const shadow::OpParam ParseEltwise(const JValue &root);
const shadow::OpParam ParseFlatten(const JValue &root);
const shadow::OpParam ParseLRN(const JValue &root);
const shadow::OpParam ParseNormalize(const JValue &root);
const shadow::OpParam ParsePermute(const JValue &root);
const shadow::OpParam ParsePooling(const JValue &root);
const shadow::OpParam ParsePriorBox(const JValue &root);
const shadow::OpParam ParseReorg(const JValue &root);
const shadow::OpParam ParseReshape(const JValue &root);
const shadow::OpParam ParseScale(const JValue &root);
const shadow::OpParam ParseSoftmax(const JValue &root);
#endif

}  // namespace Parser

}  // namespace Shadow

#endif  // SHADOW_CORE_PARSER_HPP
