file(GLOB_RECURSE tmp *.cpp *hpp)
set(shadow_cpu_src ${shadow_cpu_src} ${tmp})

file(GLOB_RECURSE tmp *.cu)
set(shadow_gpu_src ${shadow_gpu_src} ${tmp})

set(shadow_cpu_src ${shadow_cpu_src} PARENT_SCOPE)
set(shadow_gpu_src ${shadow_gpu_src} PARENT_SCOPE)
