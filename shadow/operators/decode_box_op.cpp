#include "decode_box_op.hpp"

namespace Shadow {

void DecodeBoxOp::Forward() {
  auto top = tops(0);

  int out_stride = output_max_score_ ? 6 : (4 + num_classes_);

  if (method_ == kSSD) {
    CHECK_EQ(bottoms_size(), 3);

    const auto mbox_loc = bottoms(0);
    const auto mbox_conf = bottoms(1);
    const auto mbox_priorbox = bottoms(2);

    int batch = mbox_loc->shape(0), num_priors = mbox_loc->shape(1) / 4;

    CHECK_EQ(mbox_conf->shape(1), num_priors * num_classes_);
    CHECK_EQ(mbox_priorbox->count(1), num_priors * 8);

    top->reshape({batch, num_priors, out_stride});

    Vision::DecodeSSDBoxes(mbox_loc->data<float>(), mbox_conf->data<float>(),
                           mbox_priorbox->data<float>(), batch, num_priors,
                           num_classes_, output_max_score_,
                           top->mutable_data<float>(), ws_->Ctx());
  } else if (method_ == kRefineDet) {
    CHECK_EQ(bottoms_size(), 5);

    const auto odm_loc = bottoms(0);
    const auto odm_conf = bottoms(1);
    const auto arm_priorbox = bottoms(2);
    const auto arm_conf = bottoms(3);
    const auto arm_loc = bottoms(4);

    int batch = odm_loc->shape(0), num_priors = odm_loc->shape(1) / 4;

    CHECK_EQ(odm_conf->shape(1), num_priors * num_classes_);
    CHECK_EQ(arm_priorbox->count(1), num_priors * 8);
    CHECK_EQ(arm_conf->shape(1), num_priors * 2);
    CHECK_EQ(arm_loc->shape(1), num_priors * 4);

    top->reshape({batch, num_priors, out_stride});

    Vision::DecodeRefineDetBoxes(
        odm_loc->data<float>(), odm_conf->data<float>(),
        arm_priorbox->data<float>(), arm_conf->data<float>(),
        arm_loc->data<float>(), batch, num_priors, num_classes_,
        background_label_id_, objectness_score_, output_max_score_,
        top->mutable_data<float>(), ws_->Ctx());
  } else if (method_ == kYoloV3) {
    CHECK_EQ(bottoms_size(), masks_.size() + 1);

    const auto biases = bottoms(bottoms_size() - 1);

    CHECK_EQ(biases->count(),
             std::accumulate(masks_.begin(), masks_.end(), 0) * 2);

    int num_priors = 0;
    for (int n = 0; n < masks_.size(); ++n) {
      int mask = masks_[n];
      const auto bottom = bottoms(n);
      CHECK_EQ(bottom->shape(3), (4 + 1 + num_classes_) * mask);
      num_priors += bottom->count(1, 3) * mask;
    }

    int batch = bottoms(0)->shape(0);

    top->reshape({batch, num_priors, out_stride});

    const auto *biases_data = biases->data<float>();
    auto *top_data = top->mutable_data<float>();

    for (int n = 0; n < masks_.size(); ++n) {
      const auto bottom = bottoms(n);
      int mask = masks_[n], out_h = bottom->shape(1), out_w = bottom->shape(2);
      Vision::DecodeYoloV3Boxes(bottom->data<float>(), biases_data, batch,
                                num_priors, out_h, out_w, mask, num_classes_,
                                output_max_score_, top_data, ws_->Ctx());
      biases_data += mask * 2;
      top_data += (out_h * out_w * mask) * out_stride;
    }
  } else {
    LOG(FATAL) << "Currently only support SSD, RefineDet or YoloV3";
  }
}

REGISTER_OPERATOR(DecodeBox, DecodeBoxOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
inline void decode(const T *encode_box, const T *prior_box, const T *prior_var,
                   T *decode_box) {
  T prior_w = prior_box[2] - prior_box[0];
  T prior_h = prior_box[3] - prior_box[1];
  T prior_c_x = (prior_box[0] + prior_box[2]) / 2;
  T prior_c_y = (prior_box[1] + prior_box[3]) / 2;

  T decode_box_c_x = prior_var[0] * encode_box[0] * prior_w + prior_c_x;
  T decode_box_c_y = prior_var[1] * encode_box[1] * prior_h + prior_c_y;
  T decode_box_w = std::exp(prior_var[2] * encode_box[2]) * prior_w;
  T decode_box_h = std::exp(prior_var[3] * encode_box[3]) * prior_h;

  decode_box[0] = decode_box_c_x - decode_box_w / 2;
  decode_box[1] = decode_box_c_y - decode_box_h / 2;
  decode_box[2] = decode_box_c_x + decode_box_w / 2;
  decode_box[3] = decode_box_c_y + decode_box_h / 2;

  decode_box[0] = std::max(std::min(decode_box[0], T(1)), T(0));
  decode_box[1] = std::max(std::min(decode_box[1], T(1)), T(0));
  decode_box[2] = std::max(std::min(decode_box[2], T(1)), T(0));
  decode_box[3] = std::max(std::min(decode_box[3], T(1)), T(0));
}

template <typename T>
void DecodeSSDBoxes(const T *mbox_loc, const T *mbox_conf,
                    const T *mbox_priorbox, int batch, int num_priors,
                    int num_classes, bool output_max_score, T *decode_box,
                    Context *context) {
  for (int b = 0; b < batch; ++b) {
    const auto *prior_box = mbox_priorbox;
    const auto *prior_var = mbox_priorbox + num_priors * 4;
    for (int n = 0; n < num_priors; ++n) {
      if (output_max_score) {
        decode(mbox_loc, prior_box, prior_var, decode_box + 2);
        int max_index = -1;
        auto max_score = std::numeric_limits<T>::lowest();
        for (int c = 0; c < num_classes; ++c) {
          auto score = mbox_conf[c];
          if (score > max_score) {
            max_index = c;
            max_score = score;
          }
        }
        decode_box[0] = max_index, decode_box[1] = max_score;
        decode_box += 6;
      } else {
        decode(mbox_loc, prior_box, prior_var, decode_box);
        for (int c = 0; c < num_classes; ++c) {
          decode_box[4 + c] = mbox_conf[c];
        }
        decode_box += 4 + num_classes;
      }
      prior_box += 4, prior_var += 4;
      mbox_loc += 4, mbox_conf += num_classes;
    }
  }
}

template void DecodeSSDBoxes(const float *, const float *, const float *, int,
                             int, int, bool, float *, Context *);

template <typename T>
void DecodeRefineDetBoxes(const T *odm_loc, const T *odm_conf,
                          const T *arm_priorbox, const T *arm_conf,
                          const T *arm_loc, int batch, int num_priors,
                          int num_classes, int background_label_id,
                          float objectness_score, bool output_max_score,
                          T *decode_box, Context *context) {
  for (int b = 0; b < batch; ++b) {
    const auto *prior_box = arm_priorbox;
    const auto *prior_var = arm_priorbox + num_priors * 4;
    for (int n = 0; n < num_priors; ++n) {
      bool is_background = arm_conf[1] < objectness_score;
      if (output_max_score) {
        decode(arm_loc, prior_box, prior_var, decode_box + 2);
        decode(odm_loc, decode_box + 2, prior_var, decode_box + 2);
        if (is_background) {
          decode_box[0] = background_label_id, decode_box[1] = 1;
        } else {
          int max_index = -1;
          auto max_score = std::numeric_limits<T>::lowest();
          for (int c = 0; c < num_classes; ++c) {
            auto score = odm_conf[c];
            if (score > max_score) {
              max_index = c;
              max_score = score;
            }
          }
          decode_box[0] = max_index, decode_box[1] = max_score;
        }
        decode_box += 6;
      } else {
        decode(arm_loc, prior_box, prior_var, decode_box);
        decode(odm_loc, decode_box, prior_var, decode_box);
        if (is_background) {
          memset(decode_box + 4, 0, num_classes * sizeof(T));
          decode_box[4 + background_label_id] = 1;
        } else {
          memcpy(decode_box + 4, odm_conf, num_classes * sizeof(T));
        }
        decode_box += 4 + num_classes;
      }
      prior_box += 4, prior_var += 4;
      odm_loc += 4, odm_conf += num_classes;
      arm_conf += 2, arm_loc += 4;
    }
  }
}

template void DecodeRefineDetBoxes(const float *, const float *, const float *,
                                   const float *, const float *, int, int, int,
                                   int, float, bool, float *, Context *);

template <typename T>
void DecodeYoloV3Boxes(const T *in_data, const T *biases, int batch,
                       int num_priors, int out_h, int out_w, int mask,
                       int num_classes, bool output_max_score, T *decode_box,
                       Context *context) {
  int out_stride = output_max_score ? 6 : (4 + num_classes);
  for (int b = 0; b < batch; ++b) {
    auto *box = decode_box + b * num_priors * out_stride;
    for (int n = 0; n < out_h * out_w * mask; ++n) {
      int s = n / mask, k = n % mask;
      int h_out = s / out_w, w_out = s % out_w;

      float x = (1.f / (1 + std::exp(-in_data[0])) + w_out) / out_w;
      float y = (1.f / (1 + std::exp(-in_data[1])) + h_out) / out_h;
      float w = std::exp(in_data[2]) * biases[2 * k];
      float h = std::exp(in_data[3]) * biases[2 * k + 1];

      float scale = 1.f / (1 + std::exp(-in_data[4]));

      if (output_max_score) {
        int max_index = -1;
        auto max_score = std::numeric_limits<float>::lowest();
        for (int c = 0; c < num_classes; ++c) {
          float score = scale / (1 + std::exp(-in_data[5 + c]));
          if (score > max_score) {
            max_index = c;
            max_score = score;
          }
        }
        box[0] = max_index, box[1] = max_score;
        box[2] = x, box[3] = y, box[4] = w, box[5] = h;
      } else {
        box[0] = x, box[1] = y, box[2] = w, box[3] = h;
        for (int c = 0; c < num_classes; ++c) {
          box[4 + c] = scale / (1 + std::exp(-in_data[5 + c]));
        }
      }

      in_data += 4 + 1 + num_classes;
      box += out_stride;
    }
  }
}

template void DecodeYoloV3Boxes(const float *in_data, const float *biases,
                                int batch, int num_priors, int out_h, int out_w,
                                int mask, int num_classes, bool,
                                float *decode_box, Context *context);
#endif

}  // namespace Vision

}  // namespace Shadow
