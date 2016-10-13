__kernel void SetArray(__global float *data, int count, float value) {
  const int globalid = get_global_id(0);
  if (globalid >= count) return;

  data[globalid] = value;
}

__kernel void SetArrayRepeat(__global float *data, int offset, int N,
                             int value_size, __global float *value) {
  const int globalid = get_global_id(0);
  if (globalid >= N * value_size) return;

  int value_index = globalid / N;
  data[offset + globalid] = value[value_index];
}
