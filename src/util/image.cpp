#include "image.h"

Image::Image() {}
Image::~Image() {}

void Image::GetFloatData(unsigned char *image, int width, int height,
                         int channel, float *data) {
  int step = width * channel;
  int count = 0;
  for (int k = channel - 1; k >= 0; --k) {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        data[count++] = image[i * step + j * channel + k] / 255.0f;
      }
    }
  }
}

float Image::Im2ColGetPixel(float *image, int in_h, int in_w, int im_row,
                            int im_col, int channel, int pad) {
  im_row -= pad;
  im_col -= pad;
  if (im_row < 0 || im_col < 0 || im_row >= in_h || im_col >= in_w)
    return 0;
  return image[im_col + in_w * (im_row + in_h * channel)];
}

void Image::Im2Col(float *im_data, int in_c, int in_h, int in_w, int ksize,
                   int stride, int pad, int out_h, int out_w, float *col_data) {
  int kernel_num_ = in_c * ksize * ksize;
  for (int c = 0; c < kernel_num_; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    for (int h = 0; h < out_h; ++h) {
      for (int w = 0; w < out_w; ++w) {
        int im_row = h_offset + h * stride;
        int im_col = w_offset + w * stride;
        int col_index = (c * out_h + h) * out_w + w;
        col_data[col_index] =
            Im2ColGetPixel(im_data, in_h, in_w, im_row, im_col, c_im, pad);
      }
    }
  }

  //#pragma omp parallel for
  //  for (int p = 0; p < in_c * out_h * out_w; ++p) {
  //    int c_out = (p / out_h / out_w) % in_c;
  //    int i_out = (p / out_w) % out_h;
  //    int j_out = p % out_w;
  //    int i_inp = -pad + i_out * stride;
  //    int j_inp = -pad + j_out * stride;
  //
  //    int im_offset = c_out * in_h * in_w;
  //    int col_offset = (c_out * ksize * ksize * out_h + i_out) * out_w +
  //    j_out;
  //    for (int ki = 0; ki < ksize; ++ki) {
  //      for (int kj = 0; kj < ksize; ++kj) {
  //        int i = i_inp + ki;
  //        int j = j_inp + kj;
  //        int col_index = col_offset + (ki * ksize + kj) * out_h * out_w;
  //        col_data[col_index] = (i >= 0 && j >= 0 && i < in_h && j < in_w)
  //                                  ? im_data[im_offset + i * in_w + j]
  //                                  : 0;
  //      }
  //    }
  //  }
}

#ifdef USE_CL
void Image::CLIm2Col(cl_mem im_data, int offset, int in_c, int in_h, int in_w,
                     int ksize, int stride, int pad, int out_h, int out_w,
                     cl_mem col_data) {
  cl_kernel kernel = CL::cl_im2col_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &im_data);
  clSetKernelArg(kernel, 1, sizeof(int), &offset);
  clSetKernelArg(kernel, 2, sizeof(int), &in_c);
  clSetKernelArg(kernel, 3, sizeof(int), &in_h);
  clSetKernelArg(kernel, 4, sizeof(int), &in_w);
  clSetKernelArg(kernel, 5, sizeof(int), &ksize);
  clSetKernelArg(kernel, 6, sizeof(int), &stride);
  clSetKernelArg(kernel, 7, sizeof(int), &pad);
  clSetKernelArg(kernel, 8, sizeof(int), &out_h);
  clSetKernelArg(kernel, 9, sizeof(int), &out_w);
  clSetKernelArg(kernel, 10, sizeof(cl_mem), &col_data);
  size_t global = in_c * out_h * out_w;
  clEnqueueNDRangeKernel(*CL::easyCL->queue, kernel, 1, NULL, &global, NULL, 0,
                         NULL, NULL);
  clFinish(*CL::easyCL->queue);
}
#endif
