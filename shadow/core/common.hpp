#ifndef SHADOW_CORE_COMMON_HPP
#define SHADOW_CORE_COMMON_HPP

#define SHADOW_VERSION_MAJOR 0
#define SHADOW_VERSION_MINOR 1
#define SHADOW_VERSION_PATCH 0
#define SHADOW_VERSION                                         \
  (SHADOW_VERSION_MAJOR * 10000 + SHADOW_VERSION_MINOR * 100 + \
   SHADOW_VERSION_PATCH)

namespace Shadow {

#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(classname) \
 private:                                  \
  classname(const classname &) = delete;   \
  classname &operator=(const classname &) = delete
#endif

}  // namespace Shadow

#endif  // SHADOW_CORE_COMMON_HPP
