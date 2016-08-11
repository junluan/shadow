#ifndef SHADOW_UTIL_JIMAGE_PROC_HPP
#define SHADOW_UTIL_JIMAGE_PROC_HPP

#include "shadow/util/jimage.hpp"

namespace JImageProc {

void Filter2D(const JImage &im_src, JImage *im_filter, const float *kernel,
              int height, int width);
void GaussianBlur(const JImage &im_src, JImage *im_blur, int kernel_size,
                  float sigma = 0);
void Canny(const JImage &im_src, JImage *im_canny, float thresh_low,
           float thresh_high, bool L2 = false);

void GetGaussianKernel(float *kernel, int n, float sigma = 0);
void Gradient(const JImage &im_src, int *grad_x, int *grad_y, int *magnitude,
              bool L2 = false);

}  // namespace JImageProc

#endif  // SHADOW_UTIL_JIMAGE_PROC_HPP
