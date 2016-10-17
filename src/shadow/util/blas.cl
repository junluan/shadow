__kernel void Set(int n, float val, __global float *y) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  y[globalid] = val;
}

__kernel void SetRepeat(int n, __global float *val, int val_size,
                        __global float *y, int offy) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  int val_index = globalid / (n / val_size);
  y[offy + globalid] = val[val_index];
}

__kernel void Add(int n, __global float *a, __global float *b,
                  __global float *y) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  y[globalid] = a[globalid] + b[globalid];
}

__kernel void Sub(int n, __global float *a, __global float *b,
                  __global float *y) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  y[globalid] = a[globalid] - b[globalid];
}

__kernel void Mul(int n, __global float *a, __global float *b,
                  __global float *y) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  y[globalid] = a[globalid] * b[globalid];
}

__kernel void Div(int n, __global float *a, __global float *b,
                  __global float *y) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  y[globalid] = a[globalid] / b[globalid];
}

__kernel void Pow(int n, __global float *a, float alpha, __global float *y) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  y[globalid] = pow(a[globalid], alpha);
}
