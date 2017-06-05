function (GenProtobuf proto_src proto_hdr)
  set(${proto_src})
  set(${proto_hdr})

  foreach (fil ${ARGN})
    get_filename_component(abs_fil ${fil} ABSOLUTE)
    get_filename_component(fil_we ${fil} NAME_WE)
    get_filename_component(fil_dir ${fil} DIRECTORY)

    list(APPEND ${proto_src} "${fil_dir}/${fil_we}.pb.cc")
    list(APPEND ${proto_hdr} "${fil_dir}/${fil_we}.pb.h")

    add_custom_command(
      OUTPUT "${fil_dir}/${fil_we}.pb.cc"
             "${fil_dir}/${fil_we}.pb.h"
             "${PROJECT_SOURCE_DIR}/shadow/python/shadow/${fil_we}_pb2.py"
      COMMAND ${Protoc_EXECUTABLE} --proto_path=${fil_dir} --cpp_out=${fil_dir} ${abs_fil}
      COMMAND ${Protoc_EXECUTABLE} --proto_path=${fil_dir} --python_out=${PROJECT_SOURCE_DIR}/shadow/python/shadow ${abs_fil}
      DEPENDS ${abs_fil}
      COMMENT "Running C++/Python protocol buffer compiler on ${fil}" VERBATIM)
  endforeach ()

  set_source_files_properties(${${proto_src}} ${${proto_hdr}} PROPERTIES GENERATED TRUE)
  set(${proto_src} ${${proto_src}} PARENT_SCOPE)
  set(${proto_hdr} ${${proto_hdr}} PARENT_SCOPE)
endfunction ()

function (GenGRPC proto_src proto_hdr)
  set(${proto_src})
  set(${proto_hdr})

  foreach (fil ${ARGN})
    get_filename_component(abs_fil ${fil} ABSOLUTE)
    get_filename_component(fil_we ${fil} NAME_WE)
    get_filename_component(fil_dir ${fil} DIRECTORY)

    list(APPEND ${proto_src} "${fil_dir}/${fil_we}.pb.cc")
    list(APPEND ${proto_src} "${fil_dir}/${fil_we}.grpc.pb.cc")
    list(APPEND ${proto_hdr} "${fil_dir}/${fil_we}.pb.h")
    list(APPEND ${proto_hdr} "${fil_dir}/${fil_we}.grpc.pb.h")

    add_custom_command(
      OUTPUT "${fil_dir}/${fil_we}.pb.cc"
             "${fil_dir}/${fil_we}.grpc.pb.cc"
             "${fil_dir}/${fil_we}.pb.h"
             "${fil_dir}/${fil_we}.grpc.pb.h"
      COMMAND ${Protoc_EXECUTABLE} --proto_path=${fil_dir} --cpp_out=${fil_dir} ${abs_fil}
      COMMAND ${Protoc_EXECUTABLE} --proto_path=${fil_dir} --grpc_out=${fil_dir} --plugin=protoc-gen-grpc=/usr/local/bin/grpc_cpp_plugin ${abs_fil}
      DEPENDS ${abs_fil}
      COMMENT "Running C++ protocol buffer compiler on ${fil}" VERBATIM)
  endforeach ()

  set_source_files_properties(${${proto_src}} ${${proto_hdr}} PROPERTIES GENERATED TRUE)
  set(${proto_src} ${${proto_src}} PARENT_SCOPE)
  set(${proto_hdr} ${${proto_hdr}} PARENT_SCOPE)
endfunction ()
