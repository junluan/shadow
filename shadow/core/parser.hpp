#ifndef SHADOW_CORE_PARSER_HPP
#define SHADOW_CORE_PARSER_HPP

#include "json.hpp"
#include "params.hpp"

namespace Shadow {

#if !defined(USE_Protobuf)
namespace Parser {

void ParseNet(const std::string &proto_text, shadow::NetParam *net);

}  // namespace Parser
#endif

}  // namespace Shadow

#endif  // SHADOW_CORE_PARSER_HPP
