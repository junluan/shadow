#ifndef SHADOW_CORE_REGISTRY_HPP_
#define SHADOW_CORE_REGISTRY_HPP_

#include "common.hpp"

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace Shadow {

template <typename SrcType, typename ObjectType, typename... Args>
class Registry {
 public:
  using Creator = std::function<ObjectType*(Args...)>;

  Registry() = default;

  void Register(const SrcType& key, Creator creator,
                const std::string& help_msg = "") {
    if (Has(key)) {
      std::cerr << "Key: " << key << " is already registered" << std::endl;
      std::exit(1);
    }
    registry_[key] = creator;
    help_message_[key] = help_msg;
  }

  ObjectType* Create(const SrcType& key, Args... args) {
    return Has(key) ? registry_[key](args...) : nullptr;
  }

  bool Has(const SrcType& key) const { return registry_.count(key) > 0; }

  std::vector<SrcType> Keys() const {
    std::vector<SrcType> keys;
    for (const auto& it : registry_) {
      keys.push_back(it.first);
    }
    return keys;
  }

  std::string HelpMessage(const SrcType& key) const {
    if (help_message_.count(key)) {
      return help_message_[key];
    } else {
      return std::string();
    }
  }

 private:
  std::map<SrcType, Creator> registry_;
  std::map<SrcType, std::string> help_message_;

  DISABLE_COPY_AND_ASSIGN(Registry);
};

template <typename SrcType, typename ObjectType, typename... Args>
class Register {
 public:
  Register(const SrcType& key, Registry<SrcType, ObjectType, Args...>* registry,
           typename Registry<SrcType, ObjectType, Args...>::Creator creator,
           const std::string& help_msg = "") {
    registry->Register(key, creator, help_msg);
  }

  template <typename DerivedType>
  static ObjectType* DefaultCreator(Args... args) {
    return new DerivedType(args...);
  }
};

#define SHADOW_DECLARE_TYPED_REGISTRY(RegistryName, SrcType, ObjectType, ...) \
  Registry<SrcType, ObjectType, ##__VA_ARGS__>* RegistryName();               \
  using Register##RegistryName = Register<SrcType, ObjectType, ##__VA_ARGS__>;

#define SHADOW_DEFINE_TYPED_REGISTRY(RegistryName, SrcType, ObjectType, ...) \
  Registry<SrcType, ObjectType, ##__VA_ARGS__>* RegistryName() {             \
    static auto registry =                                                   \
        std::make_shared<Registry<SrcType, ObjectType, ##__VA_ARGS__>>();    \
    return registry.get();                                                   \
  }

#define SHADOW_REGISTER_TYPED_CLASS(RegistryName, key, ...)                  \
  namespace {                                                                \
  static Register##RegistryName SHADOW_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key, RegistryName(),                                                   \
      Register##RegistryName::DefaultCreator<__VA_ARGS__>);                  \
  }

#define SHADOW_DECLARE_REGISTRY(RegistryName, ObjectType, ...)         \
  SHADOW_DECLARE_TYPED_REGISTRY(RegistryName, std::string, ObjectType, \
                                ##__VA_ARGS__)

#define SHADOW_DEFINE_REGISTRY(RegistryName, ObjectType, ...)         \
  SHADOW_DEFINE_TYPED_REGISTRY(RegistryName, std::string, ObjectType, \
                               ##__VA_ARGS__)

#define SHADOW_REGISTER_CLASS(RegistryName, key, ...) \
  SHADOW_REGISTER_TYPED_CLASS(RegistryName, #key, __VA_ARGS__)

}  // namespace Shadow

#endif  // SHADOW_CORE_REGISTRY_HPP_
