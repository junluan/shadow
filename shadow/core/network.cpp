#include "network.hpp"
#include "util/io.hpp"

#include "operators/activate_op.hpp"
#include "operators/batch_norm_op.hpp"
#include "operators/bias_op.hpp"
#include "operators/concat_op.hpp"
#include "operators/connected_op.hpp"
#include "operators/convolution_op.hpp"
#include "operators/data_op.hpp"
#include "operators/eltwise_op.hpp"
#include "operators/flatten_op.hpp"
#include "operators/lrn_op.hpp"
#include "operators/normalize_op.hpp"
#include "operators/permute_op.hpp"
#include "operators/pooling_op.hpp"
#include "operators/prior_box_op.hpp"
#include "operators/reorg_op.hpp"
#include "operators/reshape_op.hpp"
#include "operators/scale_op.hpp"
#include "operators/softmax_op.hpp"

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

void Network::LoadModel(const std::string &proto_str, const float *weights_data,
                        int batch) {
  LoadProtoStrOrText(proto_str, &net_param_);
  Reshape(batch);
  CopyWeights(weights_data);
}

void Network::LoadModel(const std::string &proto_str,
                        const std::vector<const float *> &weights, int batch) {
  LoadProtoStrOrText(proto_str, &net_param_);
  Reshape(batch);
  CopyWeights(weights);
}

void Network::SaveModel(const std::string &proto_bin) {
#if defined(USE_Protobuf)
  for (int l = 0; l < ops_.size(); ++l) {
    net_param_.mutable_op(l)->clear_blobs();
    for (const auto &blob : ops_[l]->blobs()) {
      auto op_blob = net_param_.mutable_op(l)->add_blobs();
      for (const auto &dim : blob->shape()) {
        op_blob->mutable_shape()->add_dim(dim);
      }
      VecFloat blob_data(blob->count());
      blob->read_data(blob_data.data());
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
  ops_[0]->mutable_bottoms(0)->set_data(data);
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

  for (auto &blob : blobs_) {
    delete blob;
    blob = nullptr;
  }
  blobs_.clear();

  for (auto &blob_data : blobs_data_) {
    blob_data.second.clear();
  }
  blobs_data_.clear();

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
  return get_blob_by_name(blobs_, blob_name);
}

const float *Network::GetBlobDataByName(const std::string &blob_name) {
  const BlobF *blob = GetBlobByName(blob_name);
  if (blob == nullptr) {
    LOG(FATAL) << "Unknown blob: " + blob_name;
  } else if (blobs_data_.find(blob_name) == blobs_data_.end()) {
    blobs_data_[blob_name] = VecFloat(blob->count(), 0);
  }
  VecFloat &blob_data = blobs_data_.find(blob_name)->second;
  blob->read_data(blob_data.data());
  return blob_data.data();
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
  in_shape_.clear();
  CHECK(!net_param_.op(0).type().compare("Data"));
  for (const auto &dim : net_param_.op(0).data_param().data_shape().dim()) {
    in_shape_.push_back(dim);
  }
  CHECK_EQ(in_shape_.size(), 4) << "data_shape dimension must be four!";
  if (batch > 0) in_shape_[0] = batch;

  blobs_.clear();
  blobs_.push_back(new BlobF(in_shape_, "in_blob"));

  ops_.clear();
  for (const auto &op_param : net_param_.op()) {
    Operator *op = OpFactory(op_param);
    op->Setup(&blobs_);
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

Operator *Network::OpFactory(const shadow::OpParam &op_param) {
  Operator *op = nullptr;
  const auto &op_type = op_param.type();
  if (!op_type.compare("Activate")) {
    op = new ActivateOp(op_param);
  } else if (!op_type.compare("BatchNorm")) {
    op = new BatchNormOp(op_param);
  } else if (!op_type.compare("Bias")) {
    op = new BiasOp(op_param);
  } else if (!op_type.compare("Concat")) {
    op = new ConcatOp(op_param);
  } else if (!op_type.compare("Connected")) {
    op = new ConnectedOp(op_param);
  } else if (!op_type.compare("Convolution")) {
    op = new ConvolutionOp(op_param);
  } else if (!op_type.compare("Data")) {
    op = new DataOp(op_param);
  } else if (!op_type.compare("Eltwise")) {
    op = new EltwiseOp(op_param);
  } else if (!op_type.compare("Flatten")) {
    op = new FlattenOp(op_param);
  } else if (!op_type.compare("LRN")) {
    op = new LRNOp(op_param);
  } else if (!op_type.compare("Normalize")) {
    op = new NormalizeOp(op_param);
  } else if (!op_type.compare("Permute")) {
    op = new PermuteOp(op_param);
  } else if (!op_type.compare("Pooling")) {
    op = new PoolingOp(op_param);
  } else if (!op_type.compare("PriorBox")) {
    op = new PriorBoxOp(op_param);
  } else if (!op_type.compare("Reorg")) {
    op = new ReorgOp(op_param);
  } else if (!op_type.compare("Reshape")) {
    op = new ReshapeOp(op_param);
  } else if (!op_type.compare("Scale")) {
    op = new ScaleOp(op_param);
  } else if (!op_type.compare("Softmax")) {
    op = new SoftmaxOp(op_param);
  } else {
    LOG(FATAL) << "Error when making op: " << op_param.name()
               << ", op type: " << op_type << " is not recognized!";
  }
  return op;
}

}  // namespace Shadow
