set SOLUTION_DIR=%~1%
set PROTOC_DIR=%~2%

set SHADOW_PROTO_DIR=%SOLUTION_DIR%..\src\shadow\proto
set CAFFE_PROTO_DIR=%SOLUTION_DIR%..\tools
set PROTO_TEMP_DIR=%CAFFE_PROTO_DIR%\temp

echo ProtoCompile.cmd : Create proto temp directory "%PROTO_TEMP_DIR%"
mkdir "%PROTO_TEMP_DIR%"

echo ProtoCompile.cmd : Generating .cc and .h files for "%SHADOW_PROTO_DIR%\shadow.proto" and "%CAFFE_PROTO_DIR%\caffe.proto" 
"%PROTOC_DIR%\protoc" --proto_path="%SHADOW_PROTO_DIR%" --cpp_out="%PROTO_TEMP_DIR%" "%SHADOW_PROTO_DIR%\shadow.proto"
"%PROTOC_DIR%\protoc" --proto_path="%CAFFE_PROTO_DIR%" --cpp_out="%PROTO_TEMP_DIR%" "%CAFFE_PROTO_DIR%\caffe.proto"

echo ProtoCompile.cmd : Compare newly compiled shadow.ph.h with existing one
fc /b "%PROTO_TEMP_DIR%\shadow.pb.h" "%SHADOW_PROTO_DIR%\shadow.pb.h" > NUL

if errorlevel 1 (
  robocopy /NS /NC /NFL /NDL /NP /NJH /NJS "%PROTO_TEMP_DIR%" %SHADOW_PROTO_DIR% shadow.pb.cc shadow.pb.h
)

echo ProtoCompile.cmd : Compare newly compiled caffe.pb.h with existing one
fc /b "%PROTO_TEMP_DIR%\caffe.pb.h" "%CAFFE_PROTO_DIR%\caffe.pb.h" > NUL

if errorlevel 1 (
  robocopy /NS /NC /NFL /NDL /NP /NJH /NJS "%PROTO_TEMP_DIR%" %CAFFE_PROTO_DIR% caffe.pb.cc caffe.pb.h
)

rmdir /S /Q "%PROTO_TEMP_DIR%"

if errorlevel 1 (
  set errorlevel=0
)
