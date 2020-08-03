#include "native.hpp"

#include "util/io.hpp"

namespace Shadow {

void Native::LoadModel(const shadow::NetParam& net_param) {
  Initial(net_param);
}

void Native::LoadModel(const void* proto_data, int proto_size) {
  shadow::NetParam net_param;
  LoadProtoData(proto_data, proto_size, &net_param);
  Initial(net_param);
}

void Native::LoadModel(const std::string& proto_bin) {
  shadow::NetParam net_param;
  LoadProtoBin(proto_bin, &net_param);
  Initial(net_param);
}

void Native::LoadModel(const std::string& proto_str,
                       const std::vector<const void*>& weights) {
  shadow::NetParam net_param;
  LoadProtoStrOrText(proto_str, &net_param);
  Initial(net_param);
  CopyWeights(net_param, weights);
}

void Native::LoadModel(const std::string& proto_str, const void* weights_data) {
  shadow::NetParam net_param;
  LoadProtoStrOrText(proto_str, &net_param);
  Initial(net_param);
  CopyWeights(net_param, weights_data);
}

void Native::Forward(const std::map<std::string, void*>& data_map,
                     const std::map<std::string, std::vector<int>>& shape_map) {
  if (ops_.empty()) return;

  for (const auto& in_map : data_map) {
    const auto& blob_name = in_map.first;
    const auto* blob_data = in_map.second;
    CHECK_NOTNULL(blob_data) << blob_name << " has null data";

    const auto& blob_shape = shape_map.count(blob_name)
                                 ? shape_map.at(blob_name)
                                 : std::vector<int>();

    const auto& blob_type = ws_->GetBlobDataType(blob_name);

    if (blob_type == DataType::kI32) {
      SetInputData<std::int32_t>(blob_name, blob_shape, blob_data);
    } else if (blob_type == DataType::kI16) {
      SetInputData<std::int16_t>(blob_name, blob_shape, blob_data);
    } else if (blob_type == DataType::kI8) {
      SetInputData<std::int8_t>(blob_name, blob_shape, blob_data);
    } else if (blob_type == DataType::kU32) {
      SetInputData<std::uint32_t>(blob_name, blob_shape, blob_data);
    } else if (blob_type == DataType::kU16) {
      SetInputData<std::uint16_t>(blob_name, blob_shape, blob_data);
    } else if (blob_type == DataType::kU8) {
      SetInputData<std::uint8_t>(blob_name, blob_shape, blob_data);
    } else if (blob_type == DataType::kF32) {
      SetInputData<float>(blob_name, blob_shape, blob_data);
    } else {
      LOG(FATAL) << "Blob " << blob_name << " has unsupported type";
    }
  }

  for (auto& op : ops_) {
    std::vector<std::shared_ptr<Blob>> inputs, outputs;
    for (const auto& name : op->op_param().bottom()) {
      inputs.push_back(ws_->GetBlob(name));
    }
    for (const auto& name : op->op_param().top()) {
      outputs.push_back(ws_->GetBlob(name));
    }

    op->Run(inputs, outputs);

    LOG_IF(INFO, debug_) << op->debug_log(inputs, outputs);
  }

  LOG_IF(INFO, debug_) << "Forward Network!";
}

void Native::SaveEngine(const std::string& save_path,
                        std::map<std::string, std::vector<char>>* save_data) {}

void Native::LoadProtoData(const void* proto_data, int proto_size,
                           shadow::NetParam* net_param) {
#if defined(USE_Protobuf)
  CHECK(IO::ReadProtoFromArray(proto_data, proto_size, net_param))
      << "Error when loading proto array data";

#else
  LOG(FATAL)
      << "Unsupported load proto array model, recompiled with USE_Protobuf";
#endif
}

void Native::LoadProtoBin(const std::string& proto_bin,
                          shadow::NetParam* net_param) {
#if defined(USE_Protobuf)
  CHECK(IO::ReadProtoFromBinaryFile(proto_bin, net_param))
      << "Error when loading proto binary file: " << proto_bin;

#else
  LOG(FATAL)
      << "Unsupported load proto binary model, recompiled with USE_Protobuf";
#endif
}

void Native::LoadProtoStrOrText(const std::string& proto_str_or_text,
                                shadow::NetParam* net_param) {
  bool success;
  Path path(proto_str_or_text);
  if (path.is_file()) {
    success = IO::ReadProtoFromTextFile(proto_str_or_text, net_param);
  } else {
    success = IO::ReadProtoFromText(proto_str_or_text, net_param);
  }
  CHECK(!proto_str_or_text.empty() && success)
      << "Error when loading proto: " << proto_str_or_text;
}

void Native::Initial(const shadow::NetParam& net_param) {
  for (const auto& blob : net_param.blob()) {
    std::vector<int> blob_shape(blob.shape().begin(), blob.shape().end());

    const auto& blob_name = blob.name();
    const auto& blob_type =
        blob.has_type() ? blob.type() : std::string("float");

    auto blob_ptr = ws_->CreateBlob(blob_name, blob_type);
    CHECK_NOTNULL(blob_ptr) << "Failed to create blob " << blob_name
                            << ", asked for type " << blob_type;

    blob_ptr->reshape(blob_shape);

    if (blob_ptr->data_type() == DataType::kF32) {
      if (blob.data_f_size() > 0) {
        CHECK_EQ(blob.data_f_size(), blob_ptr->count());
        SetWeightData(blob_name, blob_shape, blob.data_f().data(), false);
      }
    } else {
      int data_b_size = 0;
      if (blob.data_b_size() > 0) {
        data_b_size = static_cast<int>(blob.data_b(0).size());
      }
      if (data_b_size > 0) {
        CHECK_EQ(data_b_size, blob_ptr->raw_size());
        SetWeightData(blob_name, blob_shape, blob.data_b(0).data(), false);
      }
    }
  }

  ops_.clear(), in_blob_.clear();
  for (const auto& op_param : net_param.op()) {
    ops_.emplace_back(CreateOperator(op_param, ws_));

    if (op_param.type() == "Input") {
      CHECK(in_blob_.empty());
      for (const auto& blob_name : op_param.top()) {
        ws_->GetBlob(blob_name)->reshape(
            ops_.back()->get_repeated_argument<int>(blob_name));
        in_blob_.push_back(blob_name);
      }
    }
  }

  arg_helper_ = ArgumentHelper(net_param);

  CHECK(arg_helper_.HasArgument("out_blob"))
      << "Network must have out_blob argument";
  out_blob_ = arg_helper_.GetRepeatedArgument<std::string>("out_blob");

  LOG_IF(INFO, debug_) << "Initial Network!";
}

template <typename T>
void Native::SetInputData(const std::string& blob_name,
                          const std::vector<int>& blob_shape,
                          const void* blob_data) {
  auto blob = ws_->GetBlob(blob_name);
  CHECK_NOTNULL(blob) << "Can not find blob " << blob_name;
  if (!blob_shape.empty()) {
    blob->reshape(blob_shape);
  }
  if (device_input_) {
#if defined(USE_CUDA)
    ws_->Ctx()->allocator()->copy(blob->raw_size(), blob_data,
                                  blob->mutable_data<T>());
#else
    LOG(FATAL) << "device input is only supported when USE_CUDA is ON";
#endif
  } else {
    blob->set_data<T>(blob_data, blob->count());
  }
}

size_t Native::SetWeightData(const std::string& blob_name,
                             const std::vector<int>& blob_shape,
                             const void* blob_data, bool share_data) {
  auto blob = ws_->GetBlob(blob_name);
  CHECK_NOTNULL(blob) << "Can not find blob " << blob_name;
  if (share_data) {
    blob->share_data(blob_data, blob_shape);
  } else {
    blob->reshape(blob_shape);
    const auto& data_type = blob->data_type();
    if (data_type == DataType::kI32) {
      blob->set_data<std::int32_t>(blob_data, blob->count());
    } else if (data_type == DataType::kI16) {
      blob->set_data<std::int16_t>(blob_data, blob->count());
    } else if (data_type == DataType::kI8) {
      blob->set_data<std::int8_t>(blob_data, blob->count());
    } else if (data_type == DataType::kU32) {
      blob->set_data<std::uint32_t>(blob_data, blob->count());
    } else if (data_type == DataType::kU16) {
      blob->set_data<std::uint16_t>(blob_data, blob->count());
    } else if (data_type == DataType::kU8) {
      blob->set_data<std::uint8_t>(blob_data, blob->count());
    } else if (data_type == DataType::kF32) {
      blob->set_data<float>(blob_data, blob->count());
    } else {
      LOG(FATAL) << "Invalid data type";
    }
  }
  return blob->raw_size();
}

void Native::CopyWeights(const shadow::NetParam& net_param,
                         const std::vector<const void*>& weights) {
  bool share_weight =
      arg_helper_.GetSingleArgument<bool>("share_weight", false);
  CHECK_EQ(net_param.blob_size(), weights.size());
  for (int n = 0; n < net_param.blob_size(); ++n) {
    const auto& blob = net_param.blob(n);
    std::vector<int> blob_shape(blob.shape().begin(), blob.shape().end());
    SetWeightData(blob.name(), blob_shape, weights[n], share_weight);
  }
}

void Native::CopyWeights(const shadow::NetParam& net_param,
                         const void* weights_data) {
  bool share_weight =
      arg_helper_.GetSingleArgument<bool>("share_weight", false);
  for (const auto& blob : net_param.blob()) {
    std::vector<int> blob_shape(blob.shape().begin(), blob.shape().end());
    auto offset =
        SetWeightData(blob.name(), blob_shape, weights_data, share_weight);
    weights_data = static_cast<const unsigned char*>(weights_data) + offset;
  }
}

REGISTER_BACKEND(Native, Native);

}  // namespace Shadow
