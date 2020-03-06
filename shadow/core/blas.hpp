#ifndef SHADOW_CORE_BLAS_HPP
#define SHADOW_CORE_BLAS_HPP

#include "context.hpp"

namespace Shadow {

namespace Blas {

template <typename T>
void Set(int n, float val, T *y, int offy, Context *context);

#define DECLARE_BLAS_BINARY_FUNC(name)                                         \
  template <typename T>                                                        \
  void name(int n, const T *a, int offa, const T *b, int offb, T *y, int offy, \
            Context *context);                                                 \
  template <typename T>                                                        \
  void name(int n, const T *a, int offa, float alpha, T *y, int offy,          \
            Context *context);

DECLARE_BLAS_BINARY_FUNC(Add);
DECLARE_BLAS_BINARY_FUNC(Sub);
DECLARE_BLAS_BINARY_FUNC(Mul);
DECLARE_BLAS_BINARY_FUNC(Div);
DECLARE_BLAS_BINARY_FUNC(Pow);
DECLARE_BLAS_BINARY_FUNC(Max);
DECLARE_BLAS_BINARY_FUNC(Min);
#undef DECLARE_BLAS_BINARY_FUNC

#define DECLARE_BLAS_UNARY_FUNC(name) \
  template <typename T>               \
  void name(int n, const T *a, int offa, T *y, int offy, Context *context);

DECLARE_BLAS_UNARY_FUNC(Abs);
DECLARE_BLAS_UNARY_FUNC(Square);
DECLARE_BLAS_UNARY_FUNC(Sqrt);
DECLARE_BLAS_UNARY_FUNC(Log);
DECLARE_BLAS_UNARY_FUNC(Exp);
DECLARE_BLAS_UNARY_FUNC(Sin);
DECLARE_BLAS_UNARY_FUNC(Cos);
DECLARE_BLAS_UNARY_FUNC(Tan);
DECLARE_BLAS_UNARY_FUNC(Asin);
DECLARE_BLAS_UNARY_FUNC(Acos);
DECLARE_BLAS_UNARY_FUNC(Atan);
DECLARE_BLAS_UNARY_FUNC(Floor);
DECLARE_BLAS_UNARY_FUNC(Ceil);
DECLARE_BLAS_UNARY_FUNC(Neg);
DECLARE_BLAS_UNARY_FUNC(Reciprocal);
#undef DECLARE_BLAS_UNARY_FUNC

// Level 1
template <typename T>
void BlasSscal(int n, float alpha, T *x, int offx, Context *context);

template <typename T>
void BlasScopy(int n, const T *x, int offx, T *y, int offy, Context *context);

template <typename T>
void BlasSaxpy(int n, float alpha, const T *x, int offx, T *y, int offy,
               Context *context);

template <typename T>
void BlasSasum(int n, const T *x, int offx, float *y, Context *context);

// Level 2
template <typename T>
void BlasSgemv(int TA, int M, int N, float alpha, const T *A, int offA,
               const T *x, int offx, float beta, T *y, int offy,
               Context *context);

// Level 3
template <typename T>
void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha, const T *A,
               int offA, const T *B, int offB, float beta, T *C, int offC,
               Context *context);

}  // namespace Blas

}  // namespace Shadow

#endif  // SHADOW_CORE_BLAS_HPP
