__kernel void SetArray(__global float *y, int n, float value) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  y[globalid] = value;
}

__kernel void SetArrayRepeat(__global float *y, int offy, int n, int value_size,
                             __global float *value) {
  const int globalid = get_global_id(0);
  if (globalid >= n * value_size) return;

  int value_index = globalid / n;
  y[offy + globalid] = value[value_index];
}

__kernel void PowArray(__global float *x, int n, float alpha,
                       __global float *y) {
  const int globalid = get_global_id(0);
  if (globalid >= n) return;

  y[globalid] = pow(x[globalid], alpha);
}
