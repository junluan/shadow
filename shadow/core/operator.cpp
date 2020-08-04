#include "operator.hpp"

namespace Shadow {

Operator::Operator(const shadow::OpParam& op_param, Workspace* ws)
    : op_param_(op_param), arg_helper_(op_param), ws_(ws) {
  for (const auto& input_name : op_param_.bottom()) {
    CHECK(ws->HasBlob(input_name))
        << name() << ": Failed to check input blob " << input_name;
  }
  for (const auto& output_name : op_param_.top()) {
    const auto& output_type =
        get_single_argument<std::string>(output_name + "_type", "float");
    CHECK_NOTNULL(ws->CreateBlob(output_name, output_type))
        << name() << ": Failed to create output blob " << output_name
        << ", asked for type " << output_type;
  }
}

std::string Operator::debug_log(
    const std::vector<std::shared_ptr<Blob>>& inputs,
    const std::vector<std::shared_ptr<Blob>>& outputs) const {
  auto format_info = [](const std::vector<std::shared_ptr<Blob>>& blobs) {
    VecString info_str;
    for (const auto& blob : blobs) {
      info_str.push_back(
          Util::format_vector(blob->shape(), ",", blob->name() + "(", ")"));
    }
    return info_str;
  };
  std::stringstream ss;
  ss << name() << "(" << type()
     << "): " << Util::format_vector(format_info(inputs), " + ") << " -> "
     << Util::format_vector(format_info(outputs), " + ");
  ss << " <- ";
  json_state(ss);
  return ss.str();
}

std::shared_ptr<Operator> CreateOperator(const shadow::OpParam& op_param,
                                         Workspace* ws) {
  auto op = std::shared_ptr<Operator>(
      OperatorRegistry()->Create(op_param.type(), op_param, ws));
  CHECK_NOTNULL(op) << "Op type: " << op_param.type() << " is not registered";
  return op;
}

SHADOW_DEFINE_REGISTRY(OperatorRegistry, Operator, const shadow::OpParam&,
                       Workspace*);

}  // namespace Shadow
