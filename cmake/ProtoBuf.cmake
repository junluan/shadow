find_package(Protobuf REQUIRED QUIET)
if (Protobuf_FOUND)
  include_directories(SYSTEM ${Protobuf_INCLUDE_DIRS})
  message(STATUS "Found Protobuf include: ${Protobuf_INCLUDE_DIRS} (found version ${Protobuf_VERSION})")
  message(STATUS "Found Protobuf libraries: ${Protobuf_LIBRARIES}")
endif ()

if (EXISTS ${Protobuf_PROTOC_EXECUTABLE})
  message(STATUS "Found Protobuf protoc: ${Protobuf_PROTOC_EXECUTABLE}")
else ()
  message(FATAL_ERROR "Could not find Protobuf Compiler")
endif ()

file(GLOB proto_files "${PROJECT_SOURCE_DIR}/shadow/proto/*.proto")

foreach (fil ${proto_files})
  get_filename_component(abs_fil ${fil} ABSOLUTE)
  get_filename_component(fil_we ${fil} NAME_WE)
  get_filename_component(fil_dir ${fil} DIRECTORY)

  list(APPEND proto_srcs "${fil_dir}/${fil_we}.pb.cc")
  list(APPEND proto_hdrs "${fil_dir}/${fil_we}.pb.h")

  add_custom_command(
    OUTPUT "${fil_dir}/${fil_we}.pb.cc"
           "${fil_dir}/${fil_we}.pb.h"
           "${PROJECT_SOURCE_DIR}/shadow/python/shadow/${fil_we}_pb2.py"
    COMMAND ${Protobuf_PROTOC_EXECUTABLE} --proto_path=${fil_dir} --cpp_out=${fil_dir} ${abs_fil}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE} --proto_path=${fil_dir} --python_out=${PROJECT_SOURCE_DIR}/shadow/python/shadow ${abs_fil}
    DEPENDS ${abs_fil}
    COMMENT "Running C++/Python protocol buffer compiler on ${fil}" VERBATIM)
endforeach ()

add_library(proto STATIC ${proto_srcs} ${proto_hdrs})
target_link_libraries(proto ${Protobuf_LIBRARIES})
install(FILES ${proto_hdrs} DESTINATION include)
