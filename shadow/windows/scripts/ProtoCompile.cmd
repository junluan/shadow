set SOLUTION_DIR=%~1%
set PROTOC_DIR=%~2%

set PROTO_DIR=%SOLUTION_DIR%..\proto
set PROTO_PY_DIR=%SOLUTION_DIR%..\python\shadow
set PROTO_TEMP_DIR=%PROTO_DIR%\temp

echo ProtoCompile.cmd : Create proto temp directory "%PROTO_TEMP_DIR%"
mkdir "%PROTO_TEMP_DIR%"

echo ProtoCompile.cmd : Generating .cc and .h files for "%PROTO_DIR%\shadow.proto" and "%PROTO_DIR%\caffe.proto"
"%PROTOC_DIR%\protoc" --proto_path="%PROTO_DIR%" --cpp_out="%PROTO_TEMP_DIR%" "%PROTO_DIR%\shadow.proto"
"%PROTOC_DIR%\protoc" --proto_path="%PROTO_DIR%" --cpp_out="%PROTO_TEMP_DIR%" "%PROTO_DIR%\caffe.proto"

echo ProtoCompile.cmd : Generating shadow_pb2.py and caffe_pb2.py files for "%PROTO_DIR%\shadow.proto" and "%PROTO_DIR%\caffe.proto"
"%PROTOC_DIR%\protoc" --proto_path="%PROTO_DIR%" --python_out="%PROTO_TEMP_DIR%" "%PROTO_DIR%\shadow.proto"
"%PROTOC_DIR%\protoc" --proto_path="%PROTO_DIR%" --python_out="%PROTO_TEMP_DIR%" "%PROTO_DIR%\caffe.proto"

echo ProtoCompile.cmd : Compare newly compiled shadow.ph.h with existing one
fc /b "%PROTO_TEMP_DIR%\shadow.pb.h" "%PROTO_DIR%\shadow.pb.h" > NUL
if errorlevel 1 (
  robocopy /NS /NC /NFL /NDL /NP /NJH /NJS "%PROTO_TEMP_DIR%" %PROTO_DIR% shadow.pb.cc shadow.pb.h
  set errorlevel=0
)

echo ProtoCompile.cmd : Compare newly compiled caffe.pb.h with existing one
fc /b "%PROTO_TEMP_DIR%\caffe.pb.h" "%PROTO_DIR%\caffe.pb.h" > NUL
if errorlevel 1 (
  robocopy /NS /NC /NFL /NDL /NP /NJH /NJS "%PROTO_TEMP_DIR%" %PROTO_DIR% caffe.pb.cc caffe.pb.h
  set errorlevel=0
)

echo ProtoCompile.cmd : Compare newly compiled shadow_pb2.py with existing one
fc /b "%PROTO_TEMP_DIR%\shadow_pb2.py" "%PROTO_PY_DIR%\shadow_pb2.py" > NUL
if errorlevel 1 (
  robocopy /NS /NC /NFL /NDL /NP /NJH /NJS "%PROTO_TEMP_DIR%" %PROTO_PY_DIR% shadow_pb2.py
  set errorlevel=0
)

echo ProtoCompile.cmd : Compare newly compiled caffe_pb2.py with existing one
fc /b "%PROTO_TEMP_DIR%\caffe_pb2.py" "%PROTO_PY_DIR%\caffe_pb2.py" > NUL
if errorlevel 1 (
  robocopy /NS /NC /NFL /NDL /NP /NJH /NJS "%PROTO_TEMP_DIR%" %PROTO_PY_DIR% caffe_pb2.py
  set errorlevel=0
)

rmdir /S /Q "%PROTO_TEMP_DIR%"

if errorlevel 1 (
  set errorlevel=0
)
