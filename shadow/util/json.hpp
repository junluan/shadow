#ifndef SHADOW_UTIL_JSON_HPP_
#define SHADOW_UTIL_JSON_HPP_

#if defined(USE_JSON)
#include "document.h"
using JValue = rapidjson::Value;
using JDocument = rapidjson::Document;
#endif

#include <string>
#include <vector>

namespace Shadow {

namespace Json {

#if defined(USE_JSON)

const JDocument GetDocument(const std::string& json_text);

const JValue& GetValue(const JValue& root, const std::string& name);

const bool GetBool(const JValue& root, const std::string& name,
                   const bool& def);

const int GetInt(const JValue& root, const std::string& name, const int& def);

const float GetFloat(const JValue& root, const std::string& name,
                     const float& def);

const std::string GetString(const JValue& root, const std::string& name,
                            const std::string& def);

const std::vector<bool> GetVecBool(const JValue& root, const std::string& name,
                                   const std::vector<bool>& def = {});

const std::vector<int> GetVecInt(const JValue& root, const std::string& name,
                                 const std::vector<int>& def = {});

const std::vector<float> GetVecFloat(const JValue& root,
                                     const std::string& name,
                                     const std::vector<float>& def = {});

const std::vector<std::string> GetVecString(
    const JValue& root, const std::string& name,
    const std::vector<std::string>& def = {});

#endif

}  // namespace Json

}  // namespace Shadow

#endif  // SHADOW_UTIL_JSON_HPP_
