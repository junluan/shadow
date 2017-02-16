#include "shadow/util/json.hpp"
#include "shadow/util/log.hpp"

#if !defined(USE_Protobuf)
namespace Json {

const JDocument GetDocument(const std::string &json_text) {
  JDocument document;
  document.Parse(json_text.c_str());
  return document;
}

const JValue &GetValue(const JValue &root, const std::string &name) {
  CHECK(root.HasMember(name.c_str()));
  return root[name.c_str()];
}

const bool GetBool(const JValue &root, const std::string &name,
                   const bool &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    CHECK(value.IsBool());
    return value.GetBool();
  } else {
    return def;
  }
}

const int GetInt(const JValue &root, const std::string &name, const int &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    CHECK(value.IsNumber());
    return value.GetInt();
  } else {
    return def;
  }
}

const float GetFloat(const JValue &root, const std::string &name,
                     const float &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    CHECK(value.IsNumber());
    return value.GetFloat();
  } else {
    return def;
  }
}

const std::string GetString(const JValue &root, const std::string &name,
                            const std::string &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    CHECK(value.IsString());
    return value.GetString();
  } else {
    return def;
  }
}

const VecBool GetVecBool(const JValue &root, const std::string &name,
                         const VecBool &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    CHECK(value.IsArray());
    VecBool array;
    for (int i = 0; i < value.Size(); ++i) {
      CHECK(value[i].IsBool());
      array.push_back(value[i].GetBool());
    }
    return array;
  } else {
    return def;
  }
}

const VecInt GetVecInt(const JValue &root, const std::string &name,
                       const VecInt &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    CHECK(value.IsArray());
    VecInt array;
    for (int i = 0; i < value.Size(); ++i) {
      CHECK(value[i].IsNumber());
      array.push_back(value[i].GetInt());
    }
    return array;
  } else {
    return def;
  }
}

const VecFloat GetVecFloat(const JValue &root, const std::string &name,
                           const VecFloat &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    CHECK(value.IsArray());
    VecFloat array;
    for (int i = 0; i < value.Size(); ++i) {
      CHECK(value[i].IsNumber());
      array.push_back(value[i].GetFloat());
    }
    return array;
  } else {
    return def;
  }
}

const VecString GetVecString(const JValue &root, const std::string &name,
                             const VecString &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    CHECK(value.IsArray());
    VecString array;
    for (int i = 0; i < value.Size(); ++i) {
      CHECK(value[i].IsString());
      array.push_back(value[i].GetString());
    }
    return array;
  } else {
    return def;
  }
}

}  // namespace Json
#endif
