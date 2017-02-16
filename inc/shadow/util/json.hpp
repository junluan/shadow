#ifndef SHADOW_UTIL_JSON_HPP
#define SHADOW_UTIL_JSON_HPP

#include "shadow/util/type.hpp"

#if !defined(USE_Protobuf)
#include "rapidjson/document.h"

using JValue = rapidjson::Value;
using JDocument = rapidjson::Document;

namespace Json {

const JDocument GetDocument(const std::string &json_text);

const JValue &GetValue(const JValue &root, const std::string &name);

const bool GetBool(const JValue &root, const std::string &name,
                   const bool &def);

const int GetInt(const JValue &root, const std::string &name, const int &def);

const float GetFloat(const JValue &root, const std::string &name,
                     const float &def);

const std::string GetString(const JValue &root, const std::string &name,
                            const std::string &def);

const VecBool GetVecBool(const JValue &root, const std::string &name,
                         const VecBool &def = {});

const VecInt GetVecInt(const JValue &root, const std::string &name,
                       const VecInt &def = {});

const VecFloat GetVecFloat(const JValue &root, const std::string &name,
                           const VecFloat &def = {});

const VecString GetVecString(const JValue &root, const std::string &name,
                             const VecString &def = {});

}  // namespace Json
#endif

#endif  // SHADOW_UTIL_JSON_HPP
