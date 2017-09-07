#define CL_KERNEL_LOOP(globalid, count)  \
  const int globalid = get_global_id(0); \
  if (globalid >= count) return;

__kernel void ChannelMax(int num, int channels, int spatial_dim,
                         __global float *data, __global float *val_max) {
  CL_KERNEL_LOOP(globalid, num * spatial_dim)

  int n = globalid / spatial_dim;
  int s = globalid % spatial_dim;
  float maxval = -FLT_MAX;
  for (int c = 0; c < channels; ++c) {
    maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
  }
  val_max[globalid] = maxval;
}

__kernel void ChannelSub(int count, int num, int channels, int spatial_dim,
                         __global float *val_sub, __global float *data) {
  CL_KERNEL_LOOP(globalid, count)

  int n = globalid / channels / spatial_dim;
  int s = globalid % spatial_dim;
  data[globalid] -= val_sub[n * spatial_dim + s];
}

__kernel void ChannelSum(int num, int channels, int spatial_dim,
                         __global float *data, __global float *val_sum) {
  CL_KERNEL_LOOP(globalid, num * spatial_dim)

  int n = globalid / spatial_dim;
  int s = globalid % spatial_dim;
  float sum = 0;
  for (int c = 0; c < channels; ++c) {
    sum += data[(n * channels + c) * spatial_dim + s];
  }
  val_sum[globalid] = sum;
}

__kernel void ChannelDiv(int count, int num, int channels, int spatial_dim,
                         __global float *val_div, __global float *data) {
  CL_KERNEL_LOOP(globalid, count)

  int n = globalid / channels / spatial_dim;
  int s = globalid % spatial_dim;
  data[globalid] /= val_div[n * spatial_dim + s];
}

__kernel void Set(int n, float val, __global float *y, int offy) {
  CL_KERNEL_LOOP(globalid, n);

  y[offy + globalid] = val;
}

__kernel void AddScalar(int n, float val, __global float *y, int offy) {
  CL_KERNEL_LOOP(globalid, n);

  y[offy + globalid] += val;
}

#define BLAS_BINARY_FUNC(name, operation)                                   \
  __kernel void name(int n, __global float *a, int offa, __global float *b, \
                     int offb, __global float *y, int offy) {               \
    CL_KERNEL_LOOP(i, n);                                                   \
    a += offa, b += offb, y += offy;                                        \
    operation;                                                              \
  }

#define BLAS_UNARY_FUNC(name, operation)                                    \
  __kernel void name(int n, __global float *a, int offa, __global float *y, \
                     int offy) {                                            \
    CL_KERNEL_LOOP(i, n);                                                   \
    a += offa, y += offy;                                                   \
    operation;                                                              \
  }

BLAS_BINARY_FUNC(Add, y[i] = a[i] + b[i]);
BLAS_BINARY_FUNC(Sub, y[i] = a[i] - b[i]);
BLAS_BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
BLAS_BINARY_FUNC(Div, y[i] = a[i] / b[i]);
BLAS_BINARY_FUNC(Max, y[i] = fmax(a[i], b[i]));
BLAS_BINARY_FUNC(Min, y[i] = fmin(a[i], b[i]));

BLAS_UNARY_FUNC(Abs, y[i] = fabs(a[i]));
BLAS_UNARY_FUNC(Square, y[i] = a[i] * a[i]);
BLAS_UNARY_FUNC(Sqrt, y[i] = sqrt(a[i]));
BLAS_UNARY_FUNC(Log, y[i] = log(a[i]));
BLAS_UNARY_FUNC(Exp, y[i] = exp(a[i]));
BLAS_UNARY_FUNC(Sin, y[i] = sin(a[i]));
BLAS_UNARY_FUNC(Cos, y[i] = cos(a[i]));
BLAS_UNARY_FUNC(Tan, y[i] = tan(a[i]));
BLAS_UNARY_FUNC(Asin, y[i] = asin(a[i]));
BLAS_UNARY_FUNC(Acos, y[i] = acos(a[i]));
BLAS_UNARY_FUNC(Atan, y[i] = atan(a[i]));
BLAS_UNARY_FUNC(Floor, y[i] = floor(a[i]));
BLAS_UNARY_FUNC(Ceil, y[i] = ceil(a[i]));

__kernel void Pow(int n, __global float *a, int offa, float alpha,
                  __global float *y, int offy) {
  CL_KERNEL_LOOP(globalid, n);

  y[offy + globalid] = pow(a[offa + globalid], alpha);
}
