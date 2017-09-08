#ifndef SHADOW_CORE_PARSER_HPP
#define SHADOW_CORE_PARSER_HPP

#include "json.hpp"
#include "params.hpp"

namespace Shadow {

namespace Parser {

#if defined(USE_JSON)
void ParseJsonNet(const std::string &proto_text, shadow::NetParam *net);
void ParseJsonCommon(const JValue &root, shadow::OpParam *op);
const shadow::OpParam ParseJsonActivate(const JValue &root);
const shadow::OpParam ParseJsonBatchNorm(const JValue &root);
const shadow::OpParam ParseJsonBias(const JValue &root);
const shadow::OpParam ParseJsonBinary(const JValue &root);
const shadow::OpParam ParseJsonConcat(const JValue &root);
const shadow::OpParam ParseJsonConnected(const JValue &root);
const shadow::OpParam ParseJsonConv(const JValue &root);
const shadow::OpParam ParseJsonData(const JValue &root);
const shadow::OpParam ParseJsonEltwise(const JValue &root);
const shadow::OpParam ParseJsonFlatten(const JValue &root);
const shadow::OpParam ParseJsonLRN(const JValue &root);
const shadow::OpParam ParseJsonNormalize(const JValue &root);
const shadow::OpParam ParseJsonPermute(const JValue &root);
const shadow::OpParam ParseJsonPooling(const JValue &root);
const shadow::OpParam ParseJsonPriorBox(const JValue &root);
const shadow::OpParam ParseJsonReorg(const JValue &root);
const shadow::OpParam ParseJsonReshape(const JValue &root);
const shadow::OpParam ParseJsonScale(const JValue &root);
const shadow::OpParam ParseJsonSoftmax(const JValue &root);
const shadow::OpParam ParseJsonUnary(const JValue &root);
#endif

}  // namespace Parser

}  // namespace Shadow

#endif  // SHADOW_CORE_PARSER_HPP
