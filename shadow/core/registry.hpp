#ifndef SHADOW_CORE_FACTORY_HPP
#define SHADOW_CORE_FACTORY_HPP

#include "common.hpp"

#include <functional>
#include <iostream>
#include <map>
#include <string>

namespace Shadow {

template <class SrcType, class ObjectType, class... Args>
class Registry {
 public:
  using Creator = std::function<ObjectType*(Args...)>;

  Registry() = default;

  void Register(const SrcType& key, Creator creator) {
    if (registry_.count(key) != 0) {
      std::cerr << "Key " << key << " already registered." << std::endl;
      std::exit(1);
    }
    registry_[key] = creator;
  }

  void Register(const SrcType& key, Creator creator,
                const std::string& help_msg) {
    Register(key, creator);
    help_message_[key] = help_msg;
  }

  inline bool Has(const SrcType& key) { return (registry_.count(key) != 0); }

  ObjectType* Create(const SrcType& key, Args... args) {
    if (registry_.count(key) == 0) {
      std::cerr << "Unknown Key " << key << std::endl;
      std::exit(1);
    }
    return registry_[key](args...);
  }

  std::vector<SrcType> Keys() {
    std::vector<SrcType> keys;
    for (const auto& it : registry_) {
      keys.push_back(it.first);
    }
    return keys;
  }

  const std::map<SrcType, std::string>& HelpMessage() const {
    return help_message_;
  }

  const std::string HelpMessage(const SrcType& key) const {
    auto it = help_message_.find(key);
    if (it == help_message_.end()) {
      return std::string();
    }
    return it->second;
  }

 private:
  std::map<SrcType, Creator> registry_;
  std::map<SrcType, std::string> help_message_;

  DISABLE_COPY_AND_ASSIGN(Registry);
};

template <class SrcType, class ObjectType, class... Args>
class Register {
 public:
  Register(const SrcType& key, Registry<SrcType, ObjectType, Args...>* registry,
           typename Registry<SrcType, ObjectType, Args...>::Creator creator,
           const std::string& help_msg = "") {
    registry->Register(key, creator, help_msg);
  }

  template <class DerivedType>
  static ObjectType* DefaultCreator(Args... args) {
    return reinterpret_cast<ObjectType*>(new DerivedType(args...));
  }
};

#define SHADOW_CONCATENATE_IMPL(s1, s2) s1##s2
#define SHADOW_CONCATENATE(s1, s2) SHADOW_CONCATENATE_IMPL(s1, s2)
#ifdef __COUNTER__
#define SHADOW_ANONYMOUS_VARIABLE(str) SHADOW_CONCATENATE(str, __COUNTER__)
#else
#define SHADOW_ANONYMOUS_VARIABLE(str) SHADOW_CONCATENATE(str, __LINE__)
#endif

#define SHADOW_DECLARE_TYPED_REGISTRY(RegistryName, SrcType, ObjectType, ...) \
  Registry<SrcType, ObjectType, ##__VA_ARGS__>* RegistryName();               \
  using Register##RegistryName = Register<SrcType, ObjectType, ##__VA_ARGS__>;

#define SHADOW_DEFINE_TYPED_REGISTRY(RegistryName, SrcType, ObjectType, ...) \
  Registry<SrcType, ObjectType, ##__VA_ARGS__>* RegistryName() {             \
    static auto* registry =                                                  \
        new Registry<SrcType, ObjectType, ##__VA_ARGS__>();                  \
    return registry;                                                         \
  }

#define SHADOW_REGISTER_TYPED_CREATOR(RegistryName, key, ...)                \
  namespace {                                                                \
  static Register##RegistryName SHADOW_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key, RegistryName(), __VA_ARGS__);                                     \
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

#define SHADOW_REGISTER_CREATOR(RegistryName, key, ...) \
  SHADOW_REGISTER_TYPED_CREATOR(RegistryName, #key, __VA_ARGS__)

#define SHADOW_REGISTER_CLASS(RegistryName, key, ...) \
  SHADOW_REGISTER_TYPED_CLASS(RegistryName, #key, __VA_ARGS__)

}  // namespace Shadow

#endif  // SHADOW_CORE_FACTORY_HPP
