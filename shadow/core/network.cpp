#include "network.hpp"
#include "util/io.hpp"

#if defined(USE_NNPACK)
#include "nnpack.h"
#endif

namespace Shadow {

void Network::Setup(int device_id) {
#if defined(USE_CUDA) | defined(USE_CL)
  Kernel::Setup(device_id);
#endif

#if defined(USE_NNPACK)
  CHECK_EQ(nnp_initialize(), nnp_status_success);
#endif
}

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

void Network::SaveModel(const std::string &proto_bin) {
#if defined(USE_Protobuf)
  for (int o = 0; o < ops_.size(); ++o) {
    net_param_.mutable_op(o)->clear_blobs();
    for (const auto &blob : ops_[o]->blobs()) {
      auto op_blob = net_param_.mutable_op(o)->add_blobs();
      for (const auto &dim : blob->shape()) {
        op_blob->add_shape(dim);
      }
      VecFloat blob_data(blob->count());
      blob->read_data(blob_data.data(), blob_data.size());
      for (const auto &data : blob_data) {
        op_blob->add_data(data);
      }
    }
  }
  IO::WriteProtoToBinaryFile(net_param_, proto_bin);

#else
  LOG(FATAL) << "Unsupported save binary model, recompiled with USE_Protobuf";
#endif
}

void Network::Forward(const float *data) {
  CHECK_NOTNULL(data);
  if (ops_.size() == 0) return;
  CHECK(!ops_[0]->type().compare("Data"))
      << "The first Op must be Data operator!";
  auto in_blob = ops_[0]->mutable_bottoms(0);
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

#if defined(USE_CUDA) | defined(USE_CL)
  Kernel::Release();
#endif

#if defined(USE_NNPACK)
  CHECK_EQ(nnp_deinitialize(), nnp_status_success);
#endif

  DLOG(INFO) << "Release Network!";
}

const Operator *Network::GetOpByName(const std::string &op_name) {
  for (const auto &op : ops_) {
    if (!op_name.compare(op->name())) return op;
  }
  return nullptr;
}

const BlobF *Network::GetBlobByName(const std::string &blob_name) {
  return ws_.GetBlob(blob_name);
}

const float *Network::GetBlobDataByName(const std::string &blob_name) {
  auto *blob = ws_.GetBlob(blob_name);
  if (blob == nullptr) {
    LOG(FATAL) << "Unknown blob: " + blob_name;
  } else {
    return blob->cpu_data();
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
  CHECK(proto_str_or_text.compare("") && success)
      << "Error when loading proto: " << proto_str_or_text;
}

void Network::Reshape(int batch) {
  CHECK_GT(net_param_.op_size(), 0);
  const auto &data_op = net_param_.op(0);
  CHECK(!data_op.type().compare("Data"));
  ArgumentHelper arg_helper(data_op);
  in_shape_ = arg_helper.GetRepeatedArgument<int>("data_shape", VecInt{});
  CHECK_EQ(in_shape_.size(), 4) << "data_shape dimension must be four!";
  if (batch > 0) {
    in_shape_[0] = batch;
  }

  ws_.CreateBlob(in_shape_, "in_blob");

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
        int blob_count = op->blobs(n)->count();
        op->set_blob(n, blob_count, weights_data);
        weights_data += blob_count;
      }
    }
  }
}

void Network::CopyWeights(const float *weights_data) {
  for (auto &op : ops_) {
    for (int n = 0; n < op->blobs_size(); ++n) {
      int blob_count = op->blobs(n)->count();
      op->set_blob(n, blob_count, weights_data);
      weights_data += blob_count;
    }
  }
}

}  // namespace Shadow
