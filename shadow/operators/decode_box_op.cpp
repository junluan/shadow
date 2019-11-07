#include "decode_box_op.hpp"

namespace Shadow {

void DecodeBoxOp::Forward() {
  auto *top = mutable_tops<float>(0);

  if (method_ == kSSD) {
    CHECK_EQ(bottoms_size(), 3);

    const auto *mbox_loc = bottoms<float>(0);
    const auto *mbox_conf = bottoms<float>(1);
    const auto *mbox_priorbox = bottoms<float>(2);

    int batch = mbox_loc->shape(0), num_priors = mbox_loc->shape(1) / 4;

    CHECK_EQ(mbox_conf->shape(1), num_priors * num_classes_);
    CHECK_EQ(mbox_priorbox->count(1), num_priors * 8);

    top->reshape({batch, num_priors, 6});

    Vision::DecodeSSDBoxes(mbox_loc->data(), mbox_conf->data(),
                           mbox_priorbox->data(), batch, num_priors,
                           num_classes_, top->mutable_data());
  } else if (method_ == kRefineDet) {
    CHECK_EQ(bottoms_size(), 5);

    const auto *odm_loc = bottoms<float>(0);
    const auto *odm_conf = bottoms<float>(1);
    const auto *arm_priorbox = bottoms<float>(2);
    const auto *arm_conf = bottoms<float>(3);
    const auto *arm_loc = bottoms<float>(4);

    int batch = odm_loc->shape(0), num_priors = odm_loc->shape(1) / 4;

    CHECK_EQ(odm_conf->shape(1), num_priors * num_classes_);
    CHECK_EQ(arm_priorbox->count(1), num_priors * 8);
    CHECK_EQ(arm_conf->shape(1), num_priors * 2);
    CHECK_EQ(arm_loc->shape(1), num_priors * 4);

    top->reshape({batch, num_priors, 6});

    Vision::DecodeRefineDetBoxes(
        odm_loc->data(), odm_conf->data(), arm_priorbox->data(),
        arm_conf->data(), arm_loc->data(), batch, num_priors, num_classes_,
        background_label_id_, objectness_score_, top->mutable_data());
  } else {
    LOG(FATAL) << "Currently only support SSD or RefineDet";
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
                    int num_classes, T *decode_box) {
  for (int b = 0; b < batch; ++b) {
    auto *prior_box = mbox_priorbox;
    auto *prior_var = mbox_priorbox + num_priors * 4;
    for (int n = 0; n < num_priors; ++n) {
      decode<T>(mbox_loc, prior_box, prior_var, decode_box + 2);

      int max_index = -1;
      T max_score = std::numeric_limits<T>::lowest();
      for (int c = 0; c < num_classes; ++c) {
        T score = mbox_conf[c];
        if (score > max_score) {
          max_index = c;
          max_score = score;
        }
      }
      decode_box[0] = max_index;
      decode_box[1] = max_score;

      prior_box += 4, prior_var += 4;
      mbox_loc += 4, mbox_conf += num_classes;
      decode_box += 6;
    }
  }
}

template void DecodeSSDBoxes(const float *, const float *, const float *, int,
                             int, int, float *);

template <typename T>
void DecodeRefineDetBoxes(const T *odm_loc, const T *odm_conf,
                          const T *arm_priorbox, const T *arm_conf,
                          const T *arm_loc, int batch, int num_priors,
                          int num_classes, int background_label_id,
                          float objectness_score, T *decode_box) {
  for (int b = 0; b < batch; ++b) {
    auto *prior_box = arm_priorbox;
    auto *prior_var = arm_priorbox + num_priors * 4;
    for (int n = 0; n < num_priors; ++n) {
      decode<T>(arm_loc, prior_box, prior_var, decode_box + 2);
      decode<T>(odm_loc, decode_box + 2, prior_var, decode_box + 2);

      if (arm_conf[1] < objectness_score) {
        decode_box[0] = background_label_id;
        decode_box[1] = 1;
      } else {
        int max_index = -1;
        T max_score = std::numeric_limits<T>::lowest();
        for (int c = 0; c < num_classes; ++c) {
          T score = odm_conf[c];
          if (score > max_score) {
            max_index = c;
            max_score = score;
          }
        }
        decode_box[0] = max_index;
        decode_box[1] = max_score;
      }

      prior_box += 4, prior_var += 4;
      odm_loc += 4, odm_conf += num_classes;
      arm_conf += 2, arm_loc += 4;
      decode_box += 6;
    }
  }
}

template void DecodeRefineDetBoxes(const float *, const float *, const float *,
                                   const float *, const float *, int, int, int,
                                   int, float, float *);
#endif

}  // namespace Vision

}  // namespace Shadow
