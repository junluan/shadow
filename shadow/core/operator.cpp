#include "operator.hpp"

namespace Shadow {

Operator::Operator(const shadow::OpParam &op_param, Workspace *ws)
    : op_param_(op_param), arg_helper_(op_param_), op_ws_(ws) {
  op_name_ = op_param_.name();
  op_type_ = op_param_.type();
  bottom_names_.clear(), top_names_.clear();
  for (const auto &bottom_name : op_param_.bottom()) {
    CHECK(ws->HasBlob(bottom_name))
        << op_name_ << ": Failed to check bottom blob " << bottom_name;
    bottom_names_.push_back(bottom_name);
  }
  for (const auto &top_name : op_param_.top()) {
    const auto &top_type =
        get_single_argument<std::string>(top_name + "_type", "float");
    void *top_blob = nullptr;
    if (top_type == "float") {
      top_blob = ws->CreateBlob<float>(top_name);
    } else if (top_type == "int") {
      top_blob = ws->CreateBlob<int>(top_name);
    } else if (top_type == "unsigned char") {
      top_blob = ws->CreateBlob<unsigned char>(top_name);
    } else {
      LOG(FATAL) << op_name_ << ": Failed to create top blob " << top_name
                 << ", asked for type " << top_type;
    }
    CHECK_NOTNULL(top_blob)
        << op_name_ << ": Failed to create top blob " << top_name;
    top_names_.push_back(top_name);
  }
}

Operator::~Operator() { op_param_.Clear(); }

Operator *CreateOperator(const shadow::OpParam &op_param, Workspace *ws) {
  static StaticLinkingProtector g_protector;
  auto *registry = OperatorRegistry();
  return registry->Create(op_param.type(), op_param, ws);
}

SHADOW_DEFINE_REGISTRY(OperatorRegistry, Operator, const shadow::OpParam &,
                       Workspace *);

}  // namespace Shadow
