#ifndef SHADOW_CORE_FACTORY_HPP
#define SHADOW_CORE_FACTORY_HPP

#include "operator.hpp"
#include "params.hpp"
#include "workspace.hpp"

#include "util/util.hpp"

namespace Shadow {

class Operator;

class OpRegistry {
 public:
  typedef Operator* (*Creator)(const shadow::OpParam&, Workspace*);
  typedef std::map<std::string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static auto* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  static void AddCreator(const std::string& type, Creator creator) {
    auto& registry = Registry();
    CHECK_EQ(registry.count(type), 0) << "Op type " << type
                                      << " already registered.";
    registry[type] = creator;
  }

  static Operator* CreateOp(const shadow::OpParam& op_param, Workspace* ws) {
    const auto& type = op_param.type();
    auto& registry = Registry();
    CHECK_EQ(registry.count(type), 1)
        << "Unknown operator type: " << type
        << " (known types: " << Util::format_vector(OpTypeList()) << ")";
    return registry[type](op_param, ws);
  }

  static std::vector<std::string> OpTypeList() {
    auto& registry = Registry();
    std::vector<std::string> op_types;
    for (auto reg : registry) {
      op_types.push_back(reg.first);
    }
    return op_types;
  }

 private:
  OpRegistry() {}
};

class OpRegister {
 public:
  OpRegister(const std::string& type,
             Operator* (*creator)(const shadow::OpParam&, Workspace*)) {
    OpRegistry::AddCreator(type, creator);
  }
};

#define REGISTER_OP_CREATOR(type, creator) \
  static OpRegister g_creator_##type(#type, creator);

#define REGISTER_OP_CLASS(type)                                 \
  Operator* Creator_##type##Op(const shadow::OpParam& op_param, \
                               Workspace* ws) {                 \
    return new type##Op(op_param, ws);                          \
  }                                                             \
  REGISTER_OP_CREATOR(type, Creator_##type##Op)

}  // namespace Shadow

#endif  // SHADOW_CORE_FACTORY_HPP
