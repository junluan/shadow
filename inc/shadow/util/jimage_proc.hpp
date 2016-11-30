#ifndef SHADOW_UTIL_JIMAGE_PROC_HPP
#define SHADOW_UTIL_JIMAGE_PROC_HPP

#include "shadow/util/jimage.hpp"
#include "shadow/util/type.hpp"

namespace JImageProc {

VecPointI GetLinePoints(const PointI &start, const PointI &end,
                        const int step = 1, const int slice_axis = -1);

template <typename Dtype>
void Line(JImage *im, const Point<Dtype> &start, const Point<Dtype> &end,
          const Scalar &scalar = Scalar(0, 255, 0));
template <typename Dtype>
void Rectangle(JImage *im, const Rect<Dtype> &rect,
               const Scalar &scalar = Scalar(0, 255, 0));

void Color2Gray(const JImage &im_src, JImage *im_gray);

void Resize(const JImage &im_src, JImage *im_res, int height, int width);

template <typename Dtype>
void Crop(const JImage &im_src, JImage *im_crop, const Rect<Dtype> &crop);
template <typename Dtype>
void CropResize(const JImage &im_src, JImage *im_res, const Rect<Dtype> &crop,
                int height, int width);
template <typename Dtype>
void CropResize2Gray(const JImage &im_src, JImage *im_gray,
                     const Rect<Dtype> &crop, int height, int width);
#ifdef USE_ArcSoft
template <typename Dtype>
void CropResize(const ASVLOFFSCREEN &im_arc, float *batch_data,
                const Rect<Dtype> &crop, int height, int width);
template <typename Dtype>
void CropResize2Gray(const ASVLOFFSCREEN &im_arc, JImage *im_gray,
                     const Rect<Dtype> &crop, int height, int width);
#endif

void Filter1D(const JImage &im_src, JImage *im_filter, const float *kernel,
              int kernel_size, int direction = 0);
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
