#include "network.hpp"
#include "util/io.hpp"

namespace Shadow {

void Network::Setup(int device_id) { Kernel::Setup(device_id); }

void Network::LoadModel(const std::string &proto_bin, const VecInt &in_shape) {
  LoadProtoBin(proto_bin, &net_param_);
  Initial(in_shape);
}

void Network::LoadModel(const std::string &proto_str,
                        const std::vector<const void *> &weights,
                        const VecInt &in_shape) {
  LoadProtoStrOrText(proto_str, &net_param_);
  Initial(in_shape);
  CopyWeights(weights);
}

void Network::LoadModel(const std::string &proto_str, const float *weights_data,
                        const VecInt &in_shape) {
  LoadProtoStrOrText(proto_str, &net_param_);
  Initial(in_shape);
  CopyWeights(weights_data);
}

void Network::Reshape(const VecInt &in_shape) {
  CHECK_EQ(in_shape.size(), 4) << "in_shape dimension must be four!";
  if (in_shape != in_shape_) {
    if (ops_.empty()) return;
    auto *data_op = ops_[0];
    CHECK(data_op->type().find("Data") != std::string::npos)
        << "The first Op must be Data operator!";
    auto *in_blob = data_op->mutable_bottoms<float>(0);
    if (in_blob != nullptr) {
      in_blob->reshape(in_shape);
      for (auto &op : ops_) {
        op->Reshape();
      }
      in_shape_ = in_shape;
    } else {
      LOG(FATAL) << "in_blob is nullptr";
    }
  } else {
    DLOG(INFO) << "in_shape is the same, skip Reshape";
  }
}

void Network::Forward(const float *data) {
  CHECK_NOTNULL(data);
  if (ops_.empty()) return;
  auto *data_op = ops_[0];
  CHECK(data_op->type().find("Data") != std::string::npos)
      << "The first Op must be Data operator!";
  auto *in_blob = data_op->mutable_bottoms<float>(0);
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
  CHECK(!proto_str_or_text.empty() && success) << "Error when loading proto: "
                                               << proto_str_or_text;
}

void Network::Initial(const VecInt &in_shape) {
  CHECK_GT(net_param_.op_size(), 0);
  const auto &data_op = net_param_.op(0);
  CHECK(data_op.type().find("Data") != std::string::npos)
      << "The first Op must be Data operator!";
  ArgumentHelper arg_helper(data_op);
  in_shape_ = arg_helper.GetRepeatedArgument<int>("data_shape", VecInt{});
  CHECK_EQ(in_shape_.size(), 4) << "data_shape dimension must be four!";
  if (!in_shape.empty()) {
    CHECK_LE(in_shape.size(), 4);
    for (int i = 0; i < in_shape.size(); ++i) {
      int dim = in_shape[i];
      if (dim > 0) {
        in_shape_[i] = dim;
      }
    }
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
