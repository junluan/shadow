#include "decode_box.hpp"

namespace Shadow {

namespace Vision {

__device__ void decode(const float* encode_box, const float* prior_box,
                       const float* prior_var, float* decode_box) {
  auto prior_w = prior_box[2] - prior_box[0];
  auto prior_h = prior_box[3] - prior_box[1];
  auto prior_c_x = (prior_box[0] + prior_box[2]) / 2.f;
  auto prior_c_y = (prior_box[1] + prior_box[3]) / 2.f;

  auto decode_box_c_x = prior_var[0] * encode_box[0] * prior_w + prior_c_x;
  auto decode_box_c_y = prior_var[1] * encode_box[1] * prior_h + prior_c_y;
  auto decode_box_w = expf(prior_var[2] * encode_box[2]) * prior_w;
  auto decode_box_h = expf(prior_var[3] * encode_box[3]) * prior_h;

  decode_box[0] = decode_box_c_x - decode_box_w / 2.f;
  decode_box[1] = decode_box_c_y - decode_box_h / 2.f;
  decode_box[2] = decode_box_c_x + decode_box_w / 2.f;
  decode_box[3] = decode_box_c_y + decode_box_h / 2.f;

  decode_box[0] = fmaxf(fminf(decode_box[0], 1.f), 0.f);
  decode_box[1] = fmaxf(fminf(decode_box[1], 1.f), 0.f);
  decode_box[2] = fmaxf(fminf(decode_box[2], 1.f), 0.f);
  decode_box[3] = fmaxf(fminf(decode_box[3], 1.f), 0.f);
}

__global__ void KernelDecodeSSDBoxes(int count, const float* mbox_loc,
                                     const float* mbox_conf,
                                     const float* mbox_priorbox, int num_priors,
                                     int num_classes, bool output_max_score,
                                     float* decode_box) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int prior_index = globalid % num_priors;

    const auto* prior_box = mbox_priorbox + prior_index * 4;
    const auto* prior_var = mbox_priorbox + (num_priors + prior_index) * 4;

    const auto* mbox_loc_offset = mbox_loc + globalid * 4;
    const auto* mbox_conf_offset = mbox_conf + globalid * num_classes;

    auto* box =
        decode_box + globalid * (output_max_score ? 6 : (4 + num_classes));

    if (output_max_score) {
      decode(mbox_loc_offset, prior_box, prior_var, box + 2);
      int max_index = -1;
      auto max_score = -FLT_MAX;
      for (int c = 0; c < num_classes; ++c) {
        auto score = mbox_conf_offset[c];
        if (score > max_score) {
          max_index = c;
          max_score = score;
        }
      }
      box[0] = max_index, box[1] = max_score;
    } else {
      decode(mbox_loc_offset, prior_box, prior_var, box);
      for (int c = 0; c < num_classes; ++c) {
        box[4 + c] = mbox_conf_offset[c];
      }
    }
  }
}

template <>
void DecodeSSDBoxes<DeviceType::kGPU, float>(
    const float* mbox_loc, const float* mbox_conf, const float* mbox_priorbox,
    int batch, int num_priors, int num_classes, bool output_max_score,
    float* decode_box, Context* context) {
  int count = batch * num_priors;
  KernelDecodeSSDBoxes<<<GetBlocks(count), NumThreads, 0,
                         cudaStream_t(context->stream())>>>(
      count, mbox_loc, mbox_conf, mbox_priorbox, num_priors, num_classes,
      output_max_score, decode_box);
  CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void KernelDecodeRefineDetBoxes(
    int count, const float* odm_loc, const float* odm_conf,
    const float* arm_priorbox, const float* arm_conf, const float* arm_loc,
    int num_priors, int num_classes, int background_label_id,
    float objectness_score, bool output_max_score, float* decode_box) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int prior_index = globalid % num_priors;

    const auto* prior_box = arm_priorbox + prior_index * 4;
    const auto* prior_var = arm_priorbox + (num_priors + prior_index) * 4;

    const auto* odm_loc_offset = odm_loc + globalid * 4;
    const auto* odm_conf_offset = odm_conf + globalid * num_classes;
    const auto* arm_conf_offset = arm_conf + globalid * 2;
    const auto* arm_loc_offset = arm_loc + globalid * 4;

    auto* box =
        decode_box + globalid * (output_max_score ? 6 : (4 + num_classes));

    bool is_background = arm_conf_offset[1] < objectness_score;

    if (output_max_score) {
      decode(arm_loc_offset, prior_box, prior_var, box + 2);
      decode(odm_loc_offset, box + 2, prior_var, box + 2);
      if (is_background) {
        box[0] = background_label_id, box[1] = 1;
      } else {
        int max_index = -1;
        auto max_score = -FLT_MAX;
        for (int c = 0; c < num_classes; ++c) {
          auto score = odm_conf_offset[c];
          if (score > max_score) {
            max_index = c;
            max_score = score;
          }
        }
        box[0] = max_index, box[1] = max_score;
      }
    } else {
      decode(arm_loc_offset, prior_box, prior_var, box);
      decode(odm_loc_offset, box, prior_var, box);
      for (int c = 0; c < num_classes; ++c) {
        box[4 + c] = is_background ? 0 : odm_conf_offset[c];
      }
      if (is_background) {
        box[4 + background_label_id] = 1;
      }
    }
  }
}

template <>
void DecodeRefineDetBoxes<DeviceType::kGPU, float>(
    const float* odm_loc, const float* odm_conf, const float* arm_priorbox,
    const float* arm_conf, const float* arm_loc, int batch, int num_priors,
    int num_classes, int background_label_id, float objectness_score,
    bool output_max_score, float* decode_box, Context* context) {
  int count = batch * num_priors;
  KernelDecodeRefineDetBoxes<<<GetBlocks(count), NumThreads, 0,
                               cudaStream_t(context->stream())>>>(
      count, odm_loc, odm_conf, arm_priorbox, arm_conf, arm_loc, num_priors,
      num_classes, background_label_id, objectness_score, output_max_score,
      decode_box);
  CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void KernelDecodeYoloV3Boxes(int count, const float* in_data,
                                        const float* biases, int num_priors,
                                        int out_h, int out_w, int mask,
                                        int num_classes, bool output_max_score,
                                        float* decode_box) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / mask;
    int km_out = globalid % mask;
    int w_out = temp % out_w;
    temp /= out_w;
    int h_out = temp % out_h;
    int b_out = temp / out_h;

    const auto* in_data_offset = in_data + globalid * (4 + 1 + num_classes);

    float x = (1.f / (1 + expf(-in_data_offset[0])) + w_out) / out_w;
    float y = (1.f / (1 + expf(-in_data_offset[1])) + h_out) / out_h;
    float w = expf(in_data_offset[2]) * biases[2 * km_out];
    float h = expf(in_data_offset[3]) * biases[2 * km_out + 1];

    float scale = 1.f / (1 + expf(-in_data_offset[4]));

    int out_num =
        (b_out * num_priors + (h_out * out_w + w_out) * mask + km_out);

    auto* box =
        decode_box + out_num * (output_max_score ? 6 : (4 + num_classes));

    if (output_max_score) {
      int max_index = -1;
      auto max_score = -FLT_MAX;
      for (int c = 0; c < num_classes; ++c) {
        float score = scale / (1 + expf(-in_data_offset[5 + c]));
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
        box[4 + c] = scale / (1 + expf(-in_data_offset[5 + c]));
      }
    }
  }
}

template <>
void DecodeYoloV3Boxes<DeviceType::kGPU, float>(
    const float* in_data, const float* biases, int batch, int num_priors,
    int out_h, int out_w, int mask, int num_classes, bool output_max_score,
    float* decode_box, Context* context) {
  int count = batch * out_h * out_w * mask;
  KernelDecodeYoloV3Boxes<<<GetBlocks(count), NumThreads, 0,
                            cudaStream_t(context->stream())>>>(
      count, in_data, biases, num_priors, out_h, out_w, mask, num_classes,
      output_max_score, decode_box);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(DecodeBoxGPU,
                           DecodeBoxKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
