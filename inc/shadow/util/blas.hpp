#ifndef SHADOW_UTIL_BLAS_HPP
#define SHADOW_UTIL_BLAS_HPP

namespace Blas {

template <typename T>
void ChannelMax(int num, int channels, int spatial_dim, const T *data,
                T *val_max);
template <typename T>
void ChannelSub(int count, int num, int channels, int spatial_dim,
                const T *val_sub, T *data);
template <typename T>
void ChannelSum(int num, int channels, int spatial_dim, const T *data,
                T *val_sum);
template <typename T>
void ChannelDiv(int count, int num, int channels, int spatial_dim,
                const T *val_div, T *data);

template <typename T>
void Set(int n, float val, T *y, int offy);

template <typename T>
void Add(int n, const T *a, int offa, const T *b, int offb, T *y, int offy);
template <typename T>
void Sub(int n, const T *a, int offa, const T *b, int offb, T *y, int offy);
template <typename T>
void Mul(int n, const T *a, int offa, const T *b, int offb, T *y, int offy);
template <typename T>
void Div(int n, const T *a, int offa, const T *b, int offb, T *y, int offy);

template <typename T>
void Sqr(int n, const T *a, int offa, T *y, int offy);
template <typename T>
void Exp(int n, const T *a, int offa, T *y, int offy);
template <typename T>
void Log(int n, const T *a, int offa, T *y, int offy);
template <typename T>
void Abs(int n, const T *a, int offa, T *y, int offy);

template <typename T>
void Pow(int n, const T *a, int offa, float alpha, T *y, int offy);

template <typename T>
void Scale(int n, float alpha, const T *x, int offx, T *y, int offy);

// Level 1
template <typename T>
void BlasSscal(int n, float alpha, T *x, int offx);

template <typename T>
void BlasScopy(int n, const T *x, int offx, T *y, int offy);

template <typename T>
void BlasSaxpy(int n, float alpha, const T *x, int offx, T *y, int offy);

template <typename T>
void BlasSasum(int n, const T *x, int offx, float *y);

// Level 2
template <typename T>
void BlasSgemv(int TA, int M, int N, float alpha, const T *A, int offA,
               const T *x, int offx, float beta, T *y, int offy);

// Level 3
template <typename T>
void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha, const T *A,
               int offA, const T *B, int offB, float beta, T *C, int offC);

}  // namespace Blas

#endif  // SHADOW_UTIL_BLAS_HPP
