#include "operator.hpp"

namespace Shadow {

Operator::Operator(const shadow::OpParam &op_param, Workspace *ws)
    : op_param_(op_param), arg_helper_(op_param_), op_ws_(ws) {
  op_name_ = op_param_.name();
  op_type_ = op_param_.type();
  bottom_names_.clear(), top_names_.clear(), blob_names_.clear();
  for (const auto &bottom_name : op_param_.bottom()) {
    CHECK(ws->HasBlob(bottom_name))
        << op_name_ << ": Failed to check bottom blob " << bottom_name;
    bottom_names_.push_back(bottom_name);
  }
  for (const auto &top_name : op_param_.top()) {
    void *top_blob = nullptr;
    top_blob = ws->CreateBlob<float>(top_name);
    CHECK_NOTNULL(top_blob) << op_name_ << ": Failed to create top blob"
                            << top_name;
    top_names_.push_back(top_name);
  }
  int blob_count = 0;
  for (const auto &proto_blob : op_param_.blobs()) {
    VecInt shape;
    int cc = 1;
    for (const auto dim : proto_blob.shape()) {
      cc *= dim;
      shape.push_back(dim);
    }
    const auto &blob_name =
        op_name_ + "_" + op_type_ + "_params_" + Util::to_string(blob_count++);
    auto *blob = ws->CreateBlob<float>(shape, blob_name, true);
    int data_f_size = proto_blob.data_f_size();
    if (data_f_size > 0) {
      CHECK_EQ(data_f_size, cc)
          << "Blob float data size and blob shape are mismatch";
      blob->set_data(proto_blob.data_f().data(), data_f_size);
    }
    blob_names_.push_back(blob_name);
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
