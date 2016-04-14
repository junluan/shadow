#include "image.h"
#include "util.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

image CopyImage(const image &im) {
  image copy = Image::MakeImage(im.c, im.h, im.w, im.order);
  memcpy(copy.data, im.data, im.c * im.h * im.w);
  return copy;
}

void inline FreeImage(image &im) { delete im.data; }

void FlattenImage(const unsigned char *data, Order order, image &im_fl) {
  bool is_rgb = false;
  if (order == kRGB) {
    is_rgb = true;
  } else if (order == kBGR) {
    is_rgb = false;
  } else {
    error("Unsupported FRGB to flatten!");
  }
  int step = im_fl.w * im_fl.c;
  int ch_src;
  for (int c = 0; c < im_fl.c; ++c) {
    for (int h = 0; h < im_fl.h; ++h) {
      for (int w = 0; w < im_fl.w; ++w) {
        ch_src = is_rgb ? c : im_fl.c - c - 1;
        im_fl.data[(c * im_fl.h + h) * im_fl.w + w] =
            data[h * step + w * im_fl.c + ch_src];
      }
    }
  }
  im_fl.order = kFRGB;
}

void InverseImage(const unsigned char *data, Order order, image &im_inv) {
  bool is_rgb2bgr = false;
  if (order == kRGB) {
    is_rgb2bgr = true;
  } else if (order == kBGR) {
    is_rgb2bgr = false;
  } else {
    error("Unsupported FRGB to inverse!");
  }
  int ch_src, ch_inv;
  int step = im_inv.w * im_inv.c;
  for (int c = 0; c < im_inv.c; ++c) {
    for (int h = 0; h < im_inv.h; ++h) {
      for (int w = 0; w < im_inv.w; ++w) {
        ch_src = is_rgb2bgr ? c : im_inv.c - c - 1;
        ch_inv = is_rgb2bgr ? im_inv.c - c - 1 : c;
        im_inv.data[h * step + w * im_inv.c + ch_inv] =
            data[h * step + w * im_inv.c + ch_src];
      }
    }
  }
  im_inv.order = is_rgb2bgr ? kBGR : kRGB;
}

#ifdef USE_OpenCV
void Image::MatToImage(cv::Mat &im_mat, image &im) {
  InverseImage(im_mat.data, kBGR, im);
}

void Image::ImRectangle(cv::Mat &im_mat, VecBox &boxes, bool console_show) {
  for (int b = 0; b < boxes.size(); ++b) {
    int classindex = boxes[b].class_index;
    if (classindex == -1)
      continue;

    cv::Scalar scalar;
    if (classindex == 0)
      scalar = cv::Scalar(0, 255, 0);
    else
      scalar = cv::Scalar(255, 0, 0);

    Box box = boxes[b];
    cv::rectangle(im_mat,
                  cv::Point(static_cast<int>(box.x), static_cast<int>(box.y)),
                  cv::Point(static_cast<int>(box.x + box.w),
                            static_cast<int>(box.y + box.h)),
                  scalar, 2, 8, 0);
    if (console_show) {
      std::cout << "x = " << box.x << ", y = " << box.y << ", w = " << box.w
                << ", h = " << box.h << ", score = " << box.score << std::endl;
    }
  }
}
#endif

image Image::MakeImage(int channel, int height, int width, Order order) {
  image out;
  out.c = channel;
  out.h = height;
  out.w = width;
  out.data = new unsigned char[channel * height * width];
  out.order = order;
  return out;
}

image Image::ImRead(std::string im_path) {
  int channel, height, width;
  unsigned char *im_data =
      stbi_load(im_path.c_str(), &width, &height, &channel, 3);
  if (im_data == NULL)
    error("Failed to read image " + im_path);
  image im_fl = MakeImage(channel, height, width, kRGB);
  memcpy(im_fl.data, im_data, channel * height * width);
  delete im_data;
  return im_fl;
}

void Image::ImWrite(std::string im_path, const image im) {
  int is_ok = -1;
  int step = im.w * im.c;
  if (im.order == kRGB) {
    is_ok = stbi_write_png(im_path.c_str(), im.w, im.h, im.c, im.data, step);
  } else if (im.order == kBGR) {
    image out = MakeImage(im.c, im.h, im.w);
    InverseImage(im.data, im.order, out);
    is_ok = stbi_write_png(im_path.c_str(), im.w, im.h, im.c, out.data, step);
    FreeImage(out);
  } else {
    error("Unsupported FRGB to disk!");
  }
  if (!is_ok)
    error("Failed to write image " + im_path);
}

void Image::ImShow(std::string show_name, const image im) {
#ifdef USE_OpenCV
  if (im.order == kRGB) {
    image out = MakeImage(im.c, im.h, im.w);
    InverseImage(im.data, im.order, out);
    cv::Mat im_mat(out.h, out.w, CV_8UC3, out.data);
    cv::imshow(show_name, im_mat);
    cv::waitKey(0);
    FreeImage(out);
  } else if (im.order == kBGR) {
    cv::Mat im_mat(im.h, im.w, CV_8UC3, im.data);
    cv::imshow(show_name, im_mat);
    cv::waitKey(0);
  } else {
    error("Unsupported FRGB to show!");
  }
#else
  warn("Not compiled with OpenCV, saving image to " + show_name + ".png");
  ImWrite(show_name + ".png", im);
#endif
}

void Image::ImResize(const image im, image &im_res) {
  if (im.order == kRGB) {
    stbir_resize_uint8(im.data, im.w, im.h, im.w * im.c, im_res.data, im_res.w,
                       im_res.h, im_res.w * im.c, im.c);
  } else if (im.order == kBGR) {
    image im_inv = MakeImage(im.c, im.h, im.w);
    InverseImage(im.data, im.order, im_inv);
    stbir_resize_uint8(im_inv.data, im.w, im.h, im.w * im.c, im_res.data,
                       im_res.w, im_res.h, im_res.w * im.c, im.c);
    FreeImage(im_inv);
  } else {
    error("Unsupported FRGB resize!");
  }
}

void Image::ImRectangle(image &im, VecBox &boxes, bool console_show) {
  for (int b = 0; b < boxes.size(); ++b) {
    if (boxes[b].class_index == -1)
      continue;

    Box box = boxes[b];
    int x1 = static_cast<int>(constrain(0, im.w - 1, box.x));
    int y1 = static_cast<int>(constrain(0, im.h - 1, box.y));
    int x2 = static_cast<int>(constrain(x1, im.w - 1, x1 + box.w));
    int y2 = static_cast<int>(constrain(y1, im.h - 1, y1 + box.h));

    Scalar scalar;
    if (box.class_index == 0)
      scalar = {0, 255, 0};
    else
      scalar = {0, 0, 255};

    for (int i = x1; i <= x2; ++i) {
      int offset = (im.w * y1 + i) * im.c;
      im.data[offset + 0] = scalar.r;
      im.data[offset + 1] = scalar.g;
      im.data[offset + 2] = scalar.b;
      offset = (im.w * y2 + i) * im.c;
      im.data[offset + 0] = scalar.r;
      im.data[offset + 1] = scalar.g;
      im.data[offset + 2] = scalar.b;
    }
    for (int i = y1; i <= y2; ++i) {
      int offset = (im.w * i + x1) * im.c;
      im.data[offset + 0] = scalar.r;
      im.data[offset + 1] = scalar.g;
      im.data[offset + 2] = scalar.b;
      offset = (im.w * i + x2) * im.c;
      im.data[offset + 0] = scalar.r;
      im.data[offset + 1] = scalar.g;
      im.data[offset + 2] = scalar.b;
    }
    if (console_show) {
      std::cout << "x = " << box.x << ", y = " << box.y << ", w = " << box.w
                << ", h = " << box.h << ", score = " << box.score << std::endl;
    }
  }
}

void Image::ImFlatten(image im, image &im_fl) {
  FlattenImage(im.data, im.order, im_fl);
}

void Image::ImInverse(image im, image &im_inv) {
  InverseImage(im.data, im.order, im_inv);
}

void Image::GenBatchData(image im, float *batch_data) {
  int step = im.w * im.c;
  int count = 0;
  for (int c = 0; c < im.c; ++c) {
    for (int h = 0; h < im.h; ++h) {
      for (int w = 0; w < im.w; ++w) {
        batch_data[count++] = im.data[h * step + w * im.c + c];
      }
    }
  }
}

float Im2ColGetPixel(float *image, int in_h, int in_w, int im_row, int im_col,
                     int channel, int pad) {
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

void Image::Pooling(float *in_data, int batch, int in_c, int in_h, int in_w,
                    int ksize, int stride, int out_h, int out_w, int mode,
                    float *out_data) {
  int h_offset = ((in_h - ksize) % stride) / 2;
  int w_offset = ((in_w - ksize) % stride) / 2;

  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < in_c; ++c) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          int out_index = w + out_w * (h + out_h * (c + in_c * b));
          float max = -10000.0f;
          float sum = 0.f;
          for (int ki = 0; ki < ksize; ++ki) {
            for (int kj = 0; kj < ksize; ++kj) {
              int cur_h = h_offset + h * stride + ki;
              int cur_w = w_offset + w * stride + kj;
              int index = cur_w + in_w * (cur_h + in_h * (c + b * in_c));
              bool valid =
                  (cur_h >= 0 && cur_h < in_h && cur_w >= 0 && cur_w < in_w);
              float value = valid ? in_data[index] : -10000.0f;
              max = (value > max) ? value : max;
              sum += valid ? in_data[index] : 0.f;
            }
          }
          if (mode == 0)
            out_data[out_index] = max;
          else
            out_data[out_index] = sum / (ksize * ksize);
        }
      }
    }
  }
}
