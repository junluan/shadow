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
      SetInputData<int>(blob_name, blob_shape, blob_data);
    } else if (blob_type == DataType::kF32) {
      SetInputData<float>(blob_name, blob_shape, blob_data);
    } else if (blob_type == DataType::kU8) {
      SetInputData<unsigned char>(blob_name, blob_shape, blob_data);
    } else {
      LOG(FATAL) << "Blob " << blob_name << " has unsupported type";
    }
  }

  for (auto& op : ops_) {
    op->Run();
    DLOG(INFO) << op->debug_log();
  }

  DLOG(INFO) << "Forward Network!";
}

void Native::SaveEngine(const std::string& save_path,
                        std::vector<char>* save_data) {}

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
    std::vector<int> shape;
    int cc = 1;
    for (auto dim : blob.shape()) {
      cc *= dim;
      shape.push_back(dim);
    }
    const auto& blob_name = blob.name();
    const auto blob_type = blob.has_type() ? blob.type() : std::string("float");
    if (blob_type == "int") {
      auto blob_ptr = ws_->CreateBlob(blob_name, DataType::kI32);
      int data_i_size = blob.data_i_size();
      if (data_i_size > 0) {
        CHECK_EQ(data_i_size, cc)
            << "Blob int data size and blob shape are mismatch";
        blob_ptr->reshape(shape);
        blob_ptr->set_data<int>(blob.data_i().data(), data_i_size);
      }
    } else if (blob_type == "float") {
      auto blob_ptr = ws_->CreateBlob(blob_name, DataType::kF32);
      int data_f_size = blob.data_f_size();
      if (data_f_size > 0) {
        CHECK_EQ(data_f_size, cc)
            << "Blob float data size and blob shape are mismatch";
        blob_ptr->reshape(shape);
        blob_ptr->set_data<float>(blob.data_f().data(), data_f_size);
      }
    } else if (blob_type == "unsigned char") {
      auto blob_ptr = ws_->CreateBlob(blob_name, DataType::kU8);
      int data_b_size = 0;
      if (blob.data_b_size() > 0) {
        CHECK_EQ(blob.data_b_size(), 1);
        data_b_size = static_cast<int>(blob.data_b(0).size());
      }
      if (data_b_size > 0) {
        CHECK_EQ(data_b_size, cc)
            << "Blob unsigned char data size and blob shape are mismatch";
        auto uc_data_ptr = static_cast<unsigned char*>(
            static_cast<void*>(const_cast<char*>(blob.data_b(0).data())));
        blob_ptr->reshape(shape);
        blob_ptr->set_data<unsigned char>(uc_data_ptr, data_b_size);
      }
    } else {
      LOG(FATAL) << "Failed to create blob " << blob_name << ", asked for type "
                 << blob_type;
    }
  }

  ops_.clear();
  for (const auto& op_param : net_param.op()) {
    std::shared_ptr<Operator> op(CreateOperator(op_param, ws_));
    ops_.push_back(op);
  }

  arg_helper_ = ArgumentHelper(net_param);

  in_blob_.clear();
  for (const auto& op_param : net_param.op()) {
    if (op_param.type() == "Input") {
      for (const auto& blob_name : op_param.top()) {
        in_blob_.push_back(blob_name);
      }
      break;
    }
  }

  CHECK(arg_helper_.HasArgument("out_blob"))
      << "Network must have out_blob argument";
  out_blob_ = arg_helper_.GetRepeatedArgument<std::string>("out_blob");

  DLOG(INFO) << "Initial Network!";
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
      blob->set_data<int>(blob_data, blob->count());
    } else if (data_type == DataType::kF32) {
      blob->set_data<float>(blob_data, blob->count());
    } else if (data_type == DataType::kU8) {
      blob->set_data<unsigned char>(blob_data, blob->count());
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
    std::vector<int> blob_shape;
    for (auto dim : blob.shape()) {
      blob_shape.push_back(dim);
    }
    SetWeightData(blob.name(), blob_shape, weights[n], share_weight);
  }
}

void Native::CopyWeights(const shadow::NetParam& net_param,
                         const void* weights_data) {
  bool share_weight =
      arg_helper_.GetSingleArgument<bool>("share_weight", false);
  for (const auto& blob : net_param.blob()) {
    std::vector<int> blob_shape;
    for (auto dim : blob.shape()) {
      blob_shape.push_back(dim);
    }
    auto offset =
        SetWeightData(blob.name(), blob_shape, weights_data, share_weight);
    weights_data = static_cast<const unsigned char*>(weights_data) + offset;
  }
}

REGISTER_BACKEND(Native, Native);

}  // namespace Shadow
