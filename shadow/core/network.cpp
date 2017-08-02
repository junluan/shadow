#include "network.hpp"
#include "util/io.hpp"

namespace Shadow {

void Network::Setup(int device_id) { Kernel::Setup(device_id); }

void Network::LoadModel(const std::string &proto_bin, int batch) {
  LoadProtoBin(proto_bin, &net_param_);
  Reshape(batch);
}

void Network::LoadModel(const std::string &proto_str,
                        const std::vector<const float *> &weights, int batch) {
  LoadProtoStrOrText(proto_str, &net_param_);
  Reshape(batch);
  CopyWeights(weights);
}

void Network::LoadModel(const std::string &proto_str, const float *weights_data,
                        int batch) {
  LoadProtoStrOrText(proto_str, &net_param_);
  Reshape(batch);
  CopyWeights(weights_data);
}

void Network::Forward(const float *data) {
  CHECK_NOTNULL(data);
  if (ops_.size() == 0) return;
  CHECK(ops_[0]->type() == "Data") << "The first Op must be Data operator!";
  auto in_blob = ops_[0]->mutable_bottoms<float>(0);
  in_blob->set_data(data, in_blob->count());
  for (auto &op : ops_) {
    op->Forward();
  }
}

void Network::Release() {
  net_param_.Clear();
  in_shape_.clear();

  for (auto &op : ops_) {
    delete op;
    op = nullptr;
  }
  ops_.clear();

  Kernel::Release();

  DLOG(INFO) << "Release Network!";
}

const Operator *Network::GetOpByName(const std::string &op_name) {
  for (const auto &op : ops_) {
    if (op_name == op->name()) return op;
  }
  return nullptr;
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
  CHECK(proto_str_or_text != "" && success) << "Error when loading proto: "
                                            << proto_str_or_text;
}

void Network::Reshape(int batch) {
  CHECK_GT(net_param_.op_size(), 0);
  const auto &data_op = net_param_.op(0);
  CHECK(data_op.type() == "Data");
  ArgumentHelper arg_helper(data_op);
  in_shape_ = arg_helper.GetRepeatedArgument<int>("data_shape", VecInt{});
  CHECK_EQ(in_shape_.size(), 4) << "data_shape dimension must be four!";
  if (batch > 0) {
    in_shape_[0] = batch;
  }

  ws_.CreateBlob<float>(in_shape_, "in_blob");

  ops_.clear();
  for (const auto &op_param : net_param_.op()) {
    auto *op = CreateOperator(op_param, &ws_);
    op->Setup();
    op->Reshape();
    ops_.push_back(op);
  }
}

void Network::CopyWeights(const std::vector<const float *> &weights) {
  int weights_count = 0;
  for (auto &op : ops_) {
    if (op->blobs_size() > 0) {
      CHECK_LT(weights_count, weights.size());
      const float *weights_data = weights[weights_count++];
      for (int n = 0; n < op->blobs_size(); ++n) {
        int blob_count = op->blobs<float>(n)->count();
        op->set_blobs<float>(n, blob_count, weights_data);
        weights_data += blob_count;
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
