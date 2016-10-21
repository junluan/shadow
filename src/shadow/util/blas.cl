__kernel void Set(int n, float val, __global float *y, int offy) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  y[offy + globalid] = val;
}

__kernel void Add(int n, __global float *a, int offa, __global float *b,
                  int offb, __global float *y, int offy) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  y[offy + globalid] = a[offa + globalid] + b[offb + globalid];
}

__kernel void Sub(int n, __global float *a, int offa, __global float *b,
                  int offb, __global float *y, int offy) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  y[offy + globalid] = a[offa + globalid] - b[offb + globalid];
}

__kernel void Mul(int n, __global float *a, int offa, __global float *b,
                  int offb, __global float *y, int offy) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  y[offy + globalid] = a[offa + globalid] * b[offb + globalid];
}

__kernel void Div(int n, __global float *a, int offa, __global float *b,
                  int offb, __global float *y, int offy) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  y[offy + globalid] = a[offa + globalid] / b[offb + globalid];
}

__kernel void Square(int n, __global float *a, int offa, __global float *y,
                     int offy) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  y[offy + globalid] = a[offa + globalid] * a[offa + globalid];
}

__kernel void Pow(int n, __global float *a, int offa, float alpha,
                  __global float *y, int offy) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  y[offy + globalid] = pow(a[offa + globalid], alpha);
}
