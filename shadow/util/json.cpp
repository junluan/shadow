#include "json.hpp"

#include "log.hpp"

namespace Shadow {

namespace Json {

#if defined(USE_JSON)

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

const std::vector<bool> GetVecBool(const JValue &root, const std::string &name,
                                   const std::vector<bool> &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    CHECK(value.IsArray());
    std::vector<bool> array;
    for (int i = 0; i < value.Size(); ++i) {
      CHECK(value[i].IsBool());
      array.push_back(value[i].GetBool());
    }
    return array;
  } else {
    return def;
  }
}

const std::vector<int> GetVecInt(const JValue &root, const std::string &name,
                                 const std::vector<int> &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    CHECK(value.IsArray());
    std::vector<int> array;
    for (int i = 0; i < value.Size(); ++i) {
      CHECK(value[i].IsNumber());
      array.push_back(value[i].GetInt());
    }
    return array;
  } else {
    return def;
  }
}

const std::vector<float> GetVecFloat(const JValue &root,
                                     const std::string &name,
                                     const std::vector<float> &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    CHECK(value.IsArray());
    std::vector<float> array;
    for (int i = 0; i < value.Size(); ++i) {
      CHECK(value[i].IsNumber());
      array.push_back(value[i].GetFloat());
    }
    return array;
  } else {
    return def;
  }
}

const std::vector<std::string> GetVecString(
    const JValue &root, const std::string &name,
    const std::vector<std::string> &def) {
  if (root.HasMember(name.c_str())) {
    const auto &value = root[name.c_str()];
    CHECK(value.IsArray());
    std::vector<std::string> array;
    for (int i = 0; i < value.Size(); ++i) {
      CHECK(value[i].IsString());
      array.emplace_back(value[i].GetString());
    }
    return array;
  } else {
    return def;
  }
}

#endif

}  // namespace Json

}  // namespace Shadow
