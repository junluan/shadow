#ifndef SHADOW_CORE_COMMON_HPP_
#define SHADOW_CORE_COMMON_HPP_

#define SHADOW_STRINGIFY_IMPL(s) #s
#define SHADOW_STRINGIFY(s) SHADOW_STRINGIFY_IMPL(s)

#define SHADOW_CONCATENATE_IMPL(s1, s2) s1##s2
#define SHADOW_CONCATENATE(s1, s2) SHADOW_CONCATENATE_IMPL(s1, s2)

#ifdef __COUNTER__
#define SHADOW_ANONYMOUS_VARIABLE(s) SHADOW_CONCATENATE(s, __COUNTER__)
#else
#define SHADOW_ANONYMOUS_VARIABLE(s) SHADOW_CONCATENATE(s, __LINE__)
#endif

namespace Shadow {

#define DISABLE_COPY_AND_ASSIGN(classname) \
 private:                                  \
  classname(const classname&) = delete;    \
  classname& operator=(const classname&) = delete

}  // namespace Shadow

#endif  // SHADOW_CORE_COMMON_HPP_
