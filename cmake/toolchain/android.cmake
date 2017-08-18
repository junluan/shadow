#   ANDROID_NDK (REQUIRED) - NDK root directory
#
#   ANDROID_ABI - architecture
#     Default: armeabi-v7a
#     Posible values are:
#       armeabi
#       armeabi-v7a
#       armeabi-v7a with NEON
#       arm64-v8a
#       mips
#       mips64
#       x86
#       x86_64
# 
#   ANDROID_NATIVE_API_LEVEL - Android API version
#     Default: 9 (for 32bit) and 21 (for 64bit)
#     Posible values are independent on NDK version
#
#   ANDROID_TOOLCHAIN - toolchain name
#     Default: lastest gcc toolchain, e.g. arm-linux-androideabi-4.9
#     Posible values are independent on NDK version
#
#   ANDROID_STL - specify the runtime to use
#     Default: system
#     Posible values are:
#       system
#       gabi++_{static|shared}
#       gnustl_{static|shared}
#       stlport_{static|shared}


# Get values from environment variables
foreach (name ANDROID_NDK ANDROID_NATIVE_API_LEVEL ANDROID_ABI ANDROID_TOOLCHAIN ANDROID_STL)
  if (DEFINED ENV{${name}} AND NOT ${name})
    set(${name} $ENV{${name}})
  endif ()
endforeach ()

# ANDROID_NDK
if (NOT ANDROID_NDK)
  message(FATAL_ERROR "Please set ANDROID_NDK variable to ndk root directory")
endif ()

# ANDROID_ABI
if (NOT ANDROID_ABI)
  set(ANDROID_ABI "armeabi-v7a")
endif ()

# ANDROID_NATIVE_API_LEVEL
file(GLOB ANDROID_NATIVE_API_LEVEL_SUPPORTED RELATIVE ${ANDROID_NDK}/platforms "${ANDROID_NDK}/platforms/android-*")
message(STATUS "Available ANDROID_NATIVE_API_LEVEL: ${ANDROID_NATIVE_API_LEVEL_SUPPORTED}")
if (NOT ANDROID_NATIVE_API_LEVEL)
  if (ANDROID_ABI MATCHES "^(arm64-v8a|mips64|x86_64)$")
    set(ANDROID_NATIVE_API_LEVEL "21")
  else ()
    set(ANDROID_NATIVE_API_LEVEL "9")
  endif ()
  message(STATUS "No ANDROID_NATIVE_API_LEVEL is set, default is ${ANDROID_NATIVE_API_LEVEL}")
  list(FIND ANDROID_NATIVE_API_LEVEL_SUPPORTED "android-${ANDROID_NATIVE_API_LEVEL}" ANDROID_API_FOUND)
  if (ANDROID_API_FOUND EQUAL -1)
    list(GET ANDROID_NATIVE_API_LEVEL_SUPPORTED -1 ANDROID_API_LATEST)
    message(WARNING "ANDROID_NATIVE_API_LEVEL (${ANDROID_NATIVE_API_LEVEL}) is not supported (${ANDROID_NATIVE_API_LEVEL_SUPPORTED}). Use the latest one (${ANDROID_API_LATEST})")
    string(SUBSTRING ${ANDROID_API_LATEST} 8 -1 ANDROID_NATIVE_API_LEVEL)
  endif ()
endif ()
set(ANDROID_API_ROOT ${ANDROID_NDK}/platforms/android-${ANDROID_NATIVE_API_LEVEL})

# ANDROID_ABI
if (ANDROID_ABI STREQUAL "armeabi")
  set(NDK_ABI "armeabi")
  set(NDK_PROCESSOR "arm")
  set(NDK_C_FLAGS "-march=armv5te")
  # set(NDK_C_FLAGS "-march=armv5te -mtune=xscale -msoft-float")
  set(NDK_LLVM_TRIPLE "armv5te-none-linux-androideabi")
elseif (ANDROID_ABI STREQUAL "armeabi-v7a")
  set(NDK_ABI "armeabi-v7a")
  set(NDK_PROCESSOR "arm")
  set(NDK_C_FLAGS "-march=armv7-a")
  # set(NDK_C_FLAGS "-march=armv7-a -mfloat-abi=softfp -mfpu=neon -ftree-vectorize -ffast-math")
  set(NDK_LLVM_TRIPLE "armv7-none-linux-androideabi")
elseif (ANDROID_ABI STREQUAL "armeabi-v7a with NEON")
  set(NDK_ABI "armeabi-v7a")
  set(NDK_PROCESSOR "arm")
  set(NDK_C_FLAGS "-march=armv7-a -mfpu=neon")
  # set(NDK_C_FLAGS "-march=armv7-a -mfloat-abi=softfp -mfpu=neon -ftree-vectorize -ffast-math")
  set(NDK_LLVM_TRIPLE "armv7-none-linux-androideabi")
elseif (ANDROID_ABI STREQUAL "arm64-v8a")
  set(NDK_ABI "arm64-v8a")
  set(NDK_PROCESSOR "arm64")
  set(NDK_C_FLAGS "-march=armv8-a")
  set(NDK_LLVM_TRIPLE "aarch64-none-linux-androideabi")
elseif (ANDROID_ABI STREQUAL "x86")
  set(NDK_ABI "x86")
  set(NDK_PROCESSOR "x86")
  set(NDK_C_FLAGS "-m32")
  set(NDK_LLVM_TRIPLE "x86-none-linux-androideabi")
elseif (ANDROID_ABI STREQUAL "x86_64")
  set(NDK_ABI "x86_64")
  set(NDK_PROCESSOR "x86_64")
  set(NDK_C_FLAGS "-m64")
  set(NDK_LLVM_TRIPLE "x86_64-none-linux-androideabi")
else ()
  message(FATAL_ERROR "Unsupported ANDROID_ABI: ${ANDROID_ABI}")
endif()
message(STATUS "ANDROID_ABI: ${ANDROID_ABI}")
message(STATUS "NDK_ABI: ${NDK_ABI}")
message(STATUS "NDK_PROCESSOR: ${NDK_PROCESSOR}")
set(_lib_root "${ANDROID_NDK}/sources/cxx-stl/stlport/libs")
file(GLOB NDK_ABI_SUPPORTED RELATIVE "${_lib_root}" "${_lib_root}/*")
list(FIND NDK_ABI_SUPPORTED ${NDK_ABI} NDK_ABI_FOUND)
if (NDK_ABI_FOUND EQUAL -1)
  message(WARNING "NDK_ABI (${NDK_ABI}) is not supported (${NDK_ABI_SUPPORTED}). Rollback to 'armeabi'")
  set(NDK_ABI "armeabi")
endif ()
unset(_lib_root)

# system info
# set(CMAKE_SYSTEM_NAME Android) # Not work in CMake 2.8.12
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION ${ANDROID_NATIVE_API_LEVEL})
set(CMAKE_SYSTEM_PROCESSOR ${NDK_PROCESSOR})

# For convenience
set(UNIX TRUE)
set(ANDROID TRUE)

# CMAKE_SYSROOT - in Android this in function of Android API and architecture
set(CMAKE_SYSROOT "${ANDROID_API_ROOT}/arch-${NDK_PROCESSOR}")

# ANDROID_TOOLCHAIN
if (NDK_PROCESSOR STREQUAL "arm64")
  set(ANDROID_TOOLCHAIN_PREFIX "aarch64")
elseif (NDK_PROCESSOR MATCHES "mips")
  set(ANDROID_TOOLCHAIN_PREFIX "${NDK_PROCESSOR}el")
else ()
  set(ANDROID_TOOLCHAIN_PREFIX ${NDK_PROCESSOR})
endif ()
file(GLOB ANDROID_TOOLCHAIN_SUPPORTED RELATIVE "${ANDROID_NDK}/toolchains" "${ANDROID_NDK}/toolchains/${ANDROID_TOOLCHAIN_PREFIX}-*")
message(STATUS "Available ANDROID_TOOLCHAIN: ${ANDROID_TOOLCHAIN_SUPPORTED}")
list(SORT ANDROID_TOOLCHAIN_SUPPORTED)
list(REVERSE ANDROID_TOOLCHAIN_SUPPORTED)
if (NOT ANDROID_TOOLCHAIN)
  foreach (_TC ${ANDROID_TOOLCHAIN_SUPPORTED})
    if (NOT _TC MATCHES "(llvm|clang)") # skip the llvm/clang
      set(ANDROID_TOOLCHAIN ${_TC})
      break()
    endif ()
  endforeach ()
  message(STATUS "No ANDROID_TOOLCHAIN is set. Use the latest gcc toolchain '${ANDROID_TOOLCHAIN}'")
endif ()
file(GLOB ANDROID_TOOLCHAIN_ROOT "${ANDROID_NDK}/toolchains/${ANDROID_TOOLCHAIN}/prebuilt/*")

# get gcc version
string(REGEX MATCH "([.0-9]+)$" NDK_COMPILER_VERSION "${ANDROID_TOOLCHAIN}")

# ANDROID_STL
set(ANDROID_STL_SUPPORTED "system;gabi++_static;gabi++_shared;gnustl_static;gnustl_shared;stlport_static;stlport_shared")
message(STATUS "Available ANDROID_STL: ${ANDROID_STL_SUPPORTED}")
if (NOT ANDROID_STL)
  message(STATUS "No ANDROID_STL is set, default is 'system'")
  set(ANDROID_STL "system")
else ()
  message(STATUS "ANDROID_STL: ${ANDROID_STL}")
endif ()
set(ANDROID_STL_ROOT "${ANDROID_NDK}/sources/cxx-stl")
if (ANDROID_STL STREQUAL "system")
  set(NDK_RTTI       OFF)
  set(NDK_EXCEPTIONS OFF)
  set(ANDROID_STL_ROOT         "${ANDROID_STL_ROOT}/system")
  set(ANDROID_STL_INCLUDE_DIRS "${ANDROID_STL_ROOT}/include")
elseif (ANDROID_STL MATCHES "^gabi\\+\\+_(static|shared)$")
  set(NDK_RTTI       ON)
  set(NDK_EXCEPTIONS ON)
  set(ANDROID_STL_ROOT         "${ANDROID_STL_ROOT}/gabi++")
  set(ANDROID_STL_INCLUDE_DIRS "${ANDROID_STL_ROOT}/include")
  set(ANDROID_STL_LDFLAGS      "-L${ANDROID_STL_ROOT}/libs/${NDK_ABI}")
  set(ANDROID_STL_LIB          "-l${ANDROID_STL}")
elseif (ANDROID_STL MATCHES "^stlport_(static|shared)$")
  set(NDK_RTTI       ON)
  set(NDK_EXCEPTIONS ON)
  set(ANDROID_STL_ROOT         "${ANDROID_STL_ROOT}/stlport")
  set(ANDROID_STL_INCLUDE_DIRS "${ANDROID_STL_ROOT}/stlport")
  set(ANDROID_STL_LDFLAGS      "-L${ANDROID_STL_ROOT}/libs/${NDK_ABI}")
  set(ANDROID_STL_LIB          "-l${ANDROID_STL}")
elseif (ANDROID_STL MATCHES "^gnustl_(static|shared)$")
  set(NDK_RTTI       ON)
  set(NDK_EXCEPTIONS ON)
  set(ANDROID_STL_ROOT         "${ANDROID_STL_ROOT}/gnu-libstdc++/${NDK_COMPILER_VERSION}")
  set(ANDROID_STL_INCLUDE_DIRS "${ANDROID_STL_ROOT}/include" "${ANDROID_STL_ROOT}/libs/${NDK_ABI}/include")
  set(ANDROID_STL_LDFLAGS      "-L${ANDROID_STL_ROOT}/libs/${NDK_ABI}")
  set(ANDROID_STL_LIB          "-lsupc++ -l${ANDROID_STL}")
else ()
  message(FATAL_ERROR "Unknown ANDROID_STL: ${ANDROID_STL}")
endif ()
# NOTE: set -fno-exceptions -fno-rtti when use system
if (NOT NDK_RTTI)
  set(NDK_CXX_FLAGS "-fno-rtti")
endif ()
if (NOT NDK_EXCEPTIONS)
  set(NDK_CXX_FLAGS "${NDK_CXX_FLAGS} -fno-exceptions")
endif ()

# search paths
set(CMAKE_FIND_ROOT_PATH "${ANDROID_TOOLCHAIN_ROOT}/bin" "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# find clang
file(GLOB ANDROID_TOOLCHAIN_CLANG_ROOT "${ANDROID_NDK}/toolchains/llvm/prebuilt/*")
# NOTE: clang is slower than gcc in ndk-r11b for OT, DISABLE it
unset(ANDROID_TOOLCHAIN_CLANG_ROOT)
if (ANDROID_TOOLCHAIN_CLANG_ROOT)
  message(STATUS "Use clang: ${ANDROID_TOOLCHAIN_CLANG_ROOT}")
endif ()

# compilers (set CMAKE_C_COMPILER_ID and CMAKE_CXX_COMPILER automatically)
find_program(CMAKE_C_COMPILER
             NAMES clang
                   arm-linux-androideabi-gcc
                   aarch64-linux-android-gcc
                   mipsel-linux-android-gcc
                   mips64el-linux-android-gcc
                   i686-linux-android-gcc
                   x86_64-linux-android-gcc
             PATHS ${ANDROID_TOOLCHAIN_CLANG_ROOT} ${ANDROID_TOOLCHAIN_ROOT}
             PATH_SUFFIXES bin
             NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER
             NAMES clang++
                   arm-linux-androideabi-g++
                   aarch64-linux-android-g++
                   mipsel-linux-android-g++
                   mips64el-linux-android-g++
                   i686-linux-android-g++
                   x86_64-linux-android-g++
             PATHS ${ANDROID_TOOLCHAIN_CLANG_ROOT} ${ANDROID_TOOLCHAIN_ROOT}
             PATH_SUFFIXES bin
             NO_DEFAULT_PATH)

# global includes and link directories
include_directories(SYSTEM ${ANDROID_STL_INCLUDE_DIRS})

# cflags, cppflags, ldflags
# NOTE: -nostdlib causes link error when compiling 'viv': hidden symbol `__dso_handle'
if (ANDROID_TOOLCHAIN_CLANG_ROOT)
  set(NDK_C_FLAGS "-target ${NDK_LLVM_TRIPLE} -Qunused-arguments -gcc-toolchain ${ANDROID_TOOLCHAIN_ROOT} ${NDK_C_FLAGS}")
endif ()
# set sysroot manually for low version cmake
if (CMAKE_VERSION VERSION_LESS "3.0")
  set(NDK_C_FLAGS "--sysroot=${CMAKE_SYSROOT} ${NDK_C_FLAGS}")
endif ()
set(NDK_C_FLAGS "${NDK_C_FLAGS} -fno-short-enums")

# find path of libgcc.a
find_program(NDK_GCC_COMPILER
             NAMES arm-linux-androideabi-gcc
                   aarch64-linux-android-gcc
                   mipsel-linux-android-gcc
                   mips64el-linux-android-gcc
                   i686-linux-android-gcc
                   x86_64-linux-android-gcc
             PATHS ${ANDROID_TOOLCHAIN_ROOT}
             PATH_SUFFIXES bin
             NO_DEFAULT_PATH)
execute_process(COMMAND "${NDK_GCC_COMPILER} -print-libgcc-file-name ${CMAKE_C_FLAGS} ${NDK_C_FLAGS}"
                OUTPUT_VARIABLE NDK_LIBGCC)
# message("NDK_GCC_COMPILER: ${NDK_GCC_COMPILER}")
# message("NDK_LIBGCC: ${NDK_LIBGCC}")

# Linker flags
# set(NDK_LINKER_FLAGS "${NDK_LIBGCC} ${ANDROID_STL_LDFLAGS} -lc -lm -lstdc++ -ldl -llog")
set(NDK_LINKER_FLAGS "${NDK_LIBGCC} ${ANDROID_STL_LDFLAGS} -lc -lstdc++ -ldl -llog") # maybe linked with libm_hard.a

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${NDK_C_FLAGS}" CACHE STRING "C flags" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NDK_C_FLAGS} ${NDK_CXX_FLAGS}" CACHE STRING "C++ flags" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${NDK_LINKER_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${NDK_LINKER_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${NDK_LINKER_FLAGS}" CACHE STRING "" FORCE)

# Support automatic link of
set(CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CMAKE_CXX_CREATE_SHARED_MODULE  "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CMAKE_CXX_LINK_EXECUTABLE       "<CMAKE_CXX_COMPILER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
if (DEFINED ANDROID_STL_LIB)
  set(CMAKE_CXX_CREATE_SHARED_LIBRARY "${CMAKE_CXX_CREATE_SHARED_LIBRARY} ${ANDROID_STL_LIB}")
  set(CMAKE_CXX_CREATE_SHARED_MODULE  "${CMAKE_CXX_CREATE_SHARED_MODULE} ${ANDROID_STL_LIB}")
  set(CMAKE_CXX_LINK_EXECUTABLE       "${CMAKE_CXX_LINK_EXECUTABLE} ${ANDROID_STL_LIB}")
endif ()
