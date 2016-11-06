find_package(Protobuf REQUIRED)

if(EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
  message(STATUS "Found Protobuf Compiler: ${PROTOBUF_PROTOC_EXECUTABLE}")
else()
  message(FATAL_ERROR "Could not find Protobuf Compiler")
endif()

set(copy_dir "${PROJECT_SOURCE_DIR}/tools")

file(GLOB proto_files "${PROJECT_SOURCE_DIR}/src/shadow/proto/shadow.proto"
                      "${PROJECT_SOURCE_DIR}/tools/caffe.proto")

foreach (fil ${proto_files})
  get_filename_component(abs_fil ${fil} ABSOLUTE)
  get_filename_component(fil_we ${fil} NAME_WE)
  get_filename_component(fil_dir ${fil} DIRECTORY)

  list(APPEND proto_srcs "${fil_dir}/${fil_we}.pb.cc")
  list(APPEND proto_hdrs "${fil_dir}/${fil_we}.pb.h")

  add_custom_command(
    OUTPUT "${fil_dir}/${fil_we}.pb.cc"
           "${fil_dir}/${fil_we}.pb.h"
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --proto_path=${fil_dir} --cpp_out=${fil_dir} ${abs_fil}
    DEPENDS ${abs_fil}
    COMMENT "Running C++ protocol buffer compiler on ${fil}" VERBATIM )
endforeach ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

add_library(proto STATIC ${proto_srcs} ${proto_hdrs})
target_link_libraries(proto protobuf)
