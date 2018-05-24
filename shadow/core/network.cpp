#include "network.hpp"

#include "util/io.hpp"

namespace Shadow {

void Network::Setup(int device_id) { Kernel::Setup(device_id); }

void Network::LoadModel(const std::string &proto_bin) {
  LoadProtoBin(proto_bin, &net_param_);
  Initial();
}

void Network::LoadModel(const shadow::NetParam &net_param) {
  net_param_ = net_param;
  Initial();
}

void Network::LoadModel(const std::string &proto_str,
                        const std::vector<const void *> &weights) {
  LoadProtoStrOrText(proto_str, &net_param_);
  Initial();
  CopyWeights(weights);
}

void Network::LoadModel(const std::string &proto_str,
                        const float *weights_data) {
  LoadProtoStrOrText(proto_str, &net_param_);
  Initial();
  CopyWeights(weights_data);
}

void Network::Reshape(
    const std::map<std::string, std::vector<int>> &shape_map) {
  if (ops_.empty()) return;
  for (const auto &in_map : shape_map) {
    CHECK(!in_map.second.empty()) << in_map.first << " has empty shape";
    auto *in_blob = GetBlobByName<float>(in_map.first);
    if (in_blob != nullptr) {
      in_blob->reshape(in_map.second);
    } else {
      LOG(FATAL) << "Can not find blob " << in_map.first;
    }
  }
  for (auto &op : ops_) {
    op->Reshape();
  }

  DLOG(INFO) << "Reshape Network!";
}

void Network::Forward(const std::map<std::string, float *> &data_map) {
  if (ops_.empty()) return;
  for (const auto &in_map : data_map) {
    CHECK_NOTNULL(in_map.second) << in_map.first << " has null data";
    auto *in_blob = GetBlobByName<float>(in_map.first);
    if (in_blob != nullptr) {
      in_blob->set_data(in_map.second, in_blob->count());
    } else {
      LOG(FATAL) << "Can not find blob " << in_map.first;
    }
  }
  for (auto &op : ops_) {
    op->Forward();
  }

  DLOG(INFO) << "Forward Network!";
}

void Network::Release() {
  net_param_.Clear();

  for (auto &op : ops_) {
    delete op;
    op = nullptr;
  }
  ops_.clear();

  Kernel::Release();

  DLOG(INFO) << "Release Network!";
}

void Network::LoadProtoBin(const std::string &proto_bin,
                           shadow::NetParam *net_param) {
#if defined(USE_Protobuf)
  CHECK(IO::ReadProtoFromBinaryFile(proto_bin, net_param))
      << "Error when loading proto binary file: " << proto_bin;

#else
  LOG(FATAL) << "Unsupported load binary model, recompiled with USE_Protobuf";
#endif
}

void Network::LoadProtoStrOrText(const std::string &proto_str_or_text,
                                 shadow::NetParam *net_param) {
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

void Network::Initial() {
  arg_helper_ = ArgumentHelper(net_param_);
  CHECK_GT(net_param_.op_size(), 0);
  const auto &input_op_param = net_param_.op(0);
  CHECK(input_op_param.type().find("Input") != std::string::npos)
      << "The first Op must be Input operator!";
  ArgumentHelper arg_helper(input_op_param);
  for (const auto &input_name : input_op_param.top()) {
    const auto &input_shape =
        arg_helper.GetRepeatedArgument<int>(input_name, VecInt{});
    ws_.CreateBlob<float>(input_shape, input_name);
  }

  ops_.clear();
  for (const auto &op_param : net_param_.op()) {
    auto *op = CreateOperator(op_param, &ws_);
    op->Reshape();
    ops_.push_back(op);
  }

  DLOG(INFO) << "Initial Network!";
}

void Network::CopyWeights(const std::vector<const void *> &weights) {
  int weights_count = 0;
  for (auto &op : ops_) {
    for (int n = 0; n < op->blobs_size(); ++n) {
      CHECK_LT(weights_count, weights.size());
      const auto &blob_type = op->blobs_type(n);
      const auto *weight = weights[weights_count++];
      if (blob_type == int_id) {
        const auto *weight_data = static_cast<const int *>(weight);
        op->set_blobs<int>(n, op->blobs<int>(n)->count(), weight_data);
      } else if (blob_type == float_id) {
        const auto *weight_data = static_cast<const float *>(weight);
        op->set_blobs<float>(n, op->blobs<float>(n)->count(), weight_data);
      } else if (blob_type == uchar_id) {
        const auto *weight_data = static_cast<const unsigned char *>(weight);
        op->set_blobs<unsigned char>(n, op->blobs<unsigned char>(n)->count(),
                                     weight_data);
      } else {
        LOG(FATAL) << "Unknown blob type " << blob_type;
      }
    }
  }
}

void Network::CopyWeights(const float *weights_data) {
  for (auto &op : ops_) {
    for (int n = 0; n < op->blobs_size(); ++n) {
      int blob_count = op->blobs<float>(n)->count();
      op->set_blobs<float>(n, blob_count, weights_data);
      weights_data += blob_count;
    }
  }
}

}  // namespace Shadow
