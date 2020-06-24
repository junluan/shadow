#include "decode_box.hpp"

#include <cmath>

namespace Shadow {

namespace Vision {

inline void decode(const float* encode_box, const float* prior_box,
                   const float* prior_var, float* decode_box) {
  auto prior_w = prior_box[2] - prior_box[0];
  auto prior_h = prior_box[3] - prior_box[1];
  auto prior_c_x = (prior_box[0] + prior_box[2]) / 2.f;
  auto prior_c_y = (prior_box[1] + prior_box[3]) / 2.f;

  auto decode_box_c_x = prior_var[0] * encode_box[0] * prior_w + prior_c_x;
  auto decode_box_c_y = prior_var[1] * encode_box[1] * prior_h + prior_c_y;
  auto decode_box_w = std::exp(prior_var[2] * encode_box[2]) * prior_w;
  auto decode_box_h = std::exp(prior_var[3] * encode_box[3]) * prior_h;

  decode_box[0] = decode_box_c_x - decode_box_w / 2.f;
  decode_box[1] = decode_box_c_y - decode_box_h / 2.f;
  decode_box[2] = decode_box_c_x + decode_box_w / 2.f;
  decode_box[3] = decode_box_c_y + decode_box_h / 2.f;

  decode_box[0] = std::max(std::min(decode_box[0], 1.f), 0.f);
  decode_box[1] = std::max(std::min(decode_box[1], 1.f), 0.f);
  decode_box[2] = std::max(std::min(decode_box[2], 1.f), 0.f);
  decode_box[3] = std::max(std::min(decode_box[3], 1.f), 0.f);
}

template <>
void DecodeSSDBoxes<DeviceType::kCPU, float>(
    const float* mbox_loc, const float* mbox_conf, const float* mbox_priorbox,
    int batch, int num_priors, int num_classes, bool output_max_score,
    float* decode_box, Context* context) {
  int out_stride = output_max_score ? 6 : (4 + num_classes);
  for (int b = 0; b < batch; ++b) {
    const auto* prior_box = mbox_priorbox;
    const auto* prior_var = mbox_priorbox + num_priors * 4;
    for (int n = 0; n < num_priors; ++n) {
      if (output_max_score) {
        decode(mbox_loc, prior_box, prior_var, decode_box + 2);
        int max_index = -1;
        auto max_score = std::numeric_limits<float>::lowest();
        for (int c = 0; c < num_classes; ++c) {
          auto score = mbox_conf[c];
          if (score > max_score) {
            max_index = c;
            max_score = score;
          }
        }
        decode_box[0] = max_index, decode_box[1] = max_score;
      } else {
        decode(mbox_loc, prior_box, prior_var, decode_box);
        for (int c = 0; c < num_classes; ++c) {
          decode_box[4 + c] = mbox_conf[c];
        }
      }
      prior_box += 4, prior_var += 4;
      mbox_loc += 4, mbox_conf += num_classes;
      decode_box += out_stride;
    }
  }
}

template <>
void DecodeRefineDetBoxes<DeviceType::kCPU, float>(
    const float* odm_loc, const float* odm_conf, const float* arm_priorbox,
    const float* arm_conf, const float* arm_loc, int batch, int num_priors,
    int num_classes, int background_label_id, float objectness_score,
    bool output_max_score, float* decode_box, Context* context) {
  int out_stride = output_max_score ? 6 : (4 + num_classes);
  for (int b = 0; b < batch; ++b) {
    const auto* prior_box = arm_priorbox;
    const auto* prior_var = arm_priorbox + num_priors * 4;
    for (int n = 0; n < num_priors; ++n) {
      bool is_background = arm_conf[1] < objectness_score;
      if (output_max_score) {
        decode(arm_loc, prior_box, prior_var, decode_box + 2);
        decode(odm_loc, decode_box + 2, prior_var, decode_box + 2);
        if (is_background) {
          decode_box[0] = background_label_id, decode_box[1] = 1;
        } else {
          int max_index = -1;
          auto max_score = std::numeric_limits<float>::lowest();
          for (int c = 0; c < num_classes; ++c) {
            auto score = odm_conf[c];
            if (score > max_score) {
              max_index = c;
              max_score = score;
            }
          }
          decode_box[0] = max_index, decode_box[1] = max_score;
        }
      } else {
        decode(arm_loc, prior_box, prior_var, decode_box);
        decode(odm_loc, decode_box, prior_var, decode_box);
        if (is_background) {
          memset(decode_box + 4, 0, num_classes * sizeof(float));
          decode_box[4 + background_label_id] = 1;
        } else {
          memcpy(decode_box + 4, odm_conf, num_classes * sizeof(float));
        }
      }
      prior_box += 4, prior_var += 4;
      odm_loc += 4, odm_conf += num_classes;
      arm_conf += 2, arm_loc += 4;
      decode_box += out_stride;
    }
  }
}

template <>
void DecodeYoloV3Boxes<DeviceType::kCPU, float>(
    const float* in_data, const float* biases, int batch, int num_priors,
    int out_h, int out_w, int mask, int num_classes, bool output_max_score,
    float* decode_box, Context* context) {
  int out_stride = output_max_score ? 6 : (4 + num_classes);
  for (int b = 0; b < batch; ++b) {
    auto* box = decode_box + b * num_priors * out_stride;
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

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(DecodeBoxCPU,
                           DecodeBoxKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
