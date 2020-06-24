#ifndef SHADOW_CORE_PARSER_HPP_
#define SHADOW_CORE_PARSER_HPP_

#include "params.hpp"

namespace Shadow {

namespace Parser {

void ParseNet(const std::string& proto_text, shadow::NetParam* net);

}  // namespace Parser

}  // namespace Shadow

#endif  // SHADOW_CORE_PARSER_HPP_
