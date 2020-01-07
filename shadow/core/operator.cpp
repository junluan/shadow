#include "operator.hpp"

namespace Shadow {

Operator::Operator(const shadow::OpParam &op_param, Workspace *ws)
    : op_param_(op_param), arg_helper_(op_param), op_ws_(ws) {
  op_name_ = op_param_.name(), op_type_ = op_param_.type();
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
    if (top_type == "int") {
      top_blob = ws->CreateBlob<int>(top_name);
    } else if (top_type == "float") {
      top_blob = ws->CreateBlob<float>(top_name);
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

Operator::~Operator() { op_ws_ = nullptr; }

std::string Operator::debug_log() const {
  VecString bottom_str, top_str;
  for (int n = 0; n < bottoms_size(); ++n) {
    const auto *bottom = bottoms<float>(n);
    const auto &shape = bottom->shape();
    const auto &name = bottom->name();
    bottom_str.push_back(Util::format_vector(shape, ",", name + "(", ")"));
  }
  for (int n = 0; n < tops_size(); ++n) {
    const auto *top = tops<float>(n);
    const auto &shape = top->shape();
    const auto &name = top->name();
    top_str.push_back(Util::format_vector(shape, ",", name + "(", ")"));
  }
  std::stringstream ss;
  ss << op_name_ << "(" << op_type_
     << "): " << Util::format_vector(bottom_str, " + ") << " -> "
     << Util::format_vector(top_str, " + ");
  ss << " <- ";
  json_state(ss);
  return ss.str();
}

class StaticLinkingProtector {
 public:
  StaticLinkingProtector() {
    const auto &registered_ops = OperatorRegistry()->Keys();
    LOG_IF(FATAL, registered_ops.empty())
        << "You might have made a build error: the Shadow library does not "
           "seem to be linked with whole-static library option. To do so, use "
           "-Wl,-force_load (clang) or -Wl,--whole-archive (gcc) to link the "
           "Shadow library.";
  }
};

Operator *CreateOperator(const shadow::OpParam &op_param, Workspace *ws) {
  static StaticLinkingProtector g_protector;
  auto *op = OperatorRegistry()->Create(op_param.type(), op_param, ws);
  LOG_IF(FATAL, op == nullptr)
      << "Op type: " << op_param.type() << " is not registered";
  return op;
}

SHADOW_DEFINE_REGISTRY(OperatorRegistry, Operator, const shadow::OpParam &,
                       Workspace *);

}  // namespace Shadow
