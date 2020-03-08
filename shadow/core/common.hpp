#ifndef SHADOW_CORE_COMMON_HPP
#define SHADOW_CORE_COMMON_HPP

#define SHADOW_STRINGIFY_IMPL(s) #s
#define SHADOW_STRINGIFY(s) SHADOW_STRINGIFY_IMPL(s)

#define SHADOW_CONCATENATE_IMPL(s1, s2) s1##s2
#define SHADOW_CONCATENATE(s1, s2) SHADOW_CONCATENATE_IMPL(s1, s2)

#ifdef __COUNTER__
#define SHADOW_ANONYMOUS_VARIABLE(s) SHADOW_CONCATENATE(s, __COUNTER__)
#else
#define SHADOW_ANONYMOUS_VARIABLE(s) SHADOW_CONCATENATE(s, __LINE__)
#endif

#define SHADOW_VERSION_MAJOR 0
#define SHADOW_VERSION_MINOR 1
#define SHADOW_VERSION_PATCH 0
#define SHADOW_VERSION_STRING \
  SHADOW_STRINGIFY(           \
      SHADOW_VERSION_MAJOR.SHADOW_VERSION_MINOR.SHADOW_VERSION_PATCH)

namespace Shadow {

#define DISABLE_COPY_AND_ASSIGN(classname) \
 private:                                  \
  classname(const classname&) = delete;    \
  classname& operator=(const classname&) = delete

}  // namespace Shadow

#endif  // SHADOW_CORE_COMMON_HPP
