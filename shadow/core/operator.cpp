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
    CHECK_NOTNULL(top_blob) << op_name_ << ": Failed to create top blob "
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
    const auto blob_name =
        op_name_ + "_" + op_type_ + "_params_" + Util::to_string(blob_count++);
    const auto blob_type = proto_blob.has_type() ? proto_blob.type() : "float";
    int data_f_size = proto_blob.data_f_size();
    int data_i_size = proto_blob.data_i_size();
    int data_b_size = 0;
    if (proto_blob.data_b_size() > 0) {
      CHECK_EQ(proto_blob.data_b_size(), 1);
      data_b_size = static_cast<int>(proto_blob.data_b(0).size());
    }
    if (blob_type == "float") {
      auto *blob = ws->CreateBlob<float>(shape, blob_name, true);
      CHECK_NOTNULL(blob) << op_name_
                          << ": Failed to create operator float blob "
                          << blob_name;
      if (data_f_size > 0) {
        CHECK_EQ(data_f_size, cc)
            << "Blob float data size and blob shape are mismatch";
        blob->set_data(proto_blob.data_f().data(), data_f_size);
      }
    } else if (blob_type == "int") {
      auto *blob = ws->CreateBlob<int>(shape, blob_name, true);
      CHECK_NOTNULL(blob) << op_name_ << ": Failed to create operator int blob "
                          << blob_name;
      if (data_i_size > 0) {
        CHECK_EQ(data_i_size, cc)
            << "Blob int data size and blob shape are mismatch";
        blob->set_data(proto_blob.data_i().data(), data_i_size);
      }
    } else if (blob_type == "unsigned char") {
      auto *blob = ws->CreateBlob<unsigned char>(shape, blob_name, true);
      CHECK_NOTNULL(blob) << op_name_
                          << ": Failed to create operator float blob "
                          << blob_name;
      if (data_b_size > 0) {
        CHECK_EQ(data_b_size, cc)
            << "Blob unsigned char data size and blob shape are mismatch";
        auto uc_data_ptr = static_cast<unsigned char *>(static_cast<void *>(
            const_cast<char *>(proto_blob.data_b(0).data())));
        blob->set_data(uc_data_ptr, data_b_size);
      }
    } else {
      LOG(FATAL) << op_name_ << ": Failed to create operator blob " << blob_name
                 << ", asked for type " << blob_type;
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
