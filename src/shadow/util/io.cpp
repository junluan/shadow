#include "shadow/util/io.hpp"
#include "shadow/util/util.hpp"

#if !defined(__linux)
#include <io.h>
#endif

#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace IO {

using google::protobuf::io::CodedInputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::TextFormat;

bool ReadProtoFromText(const std::string& proto_text, Message* proto) {
  return TextFormat::ParseFromString(proto_text, proto);
}

bool ReadProtoFromTextFile(const std::string& proto_file, Message* proto) {
  int fd = open(proto_file.c_str(), O_RDONLY);
  if (fd == -1) Fatal("File not found: " + proto_file);
  FileInputStream* input = new FileInputStream(fd);
  bool success = TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

bool ReadProtoFromBinaryFile(const std::string& proto_file, Message* proto) {
#if !defined(__linux)
  int fd = open(proto_file.c_str(), O_RDONLY | O_BINARY);
#else
  int fd = open(proto_file.c_str(), O_RDONLY);
#endif
  if (fd == -1) Fatal("File not found: " + proto_file);
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(INT_MAX, 536870912);
  bool success = proto->ParseFromCodedStream(coded_input);
  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToText(const Message& proto, std::string* proto_text) {
  if (!TextFormat::PrintToString(proto, proto_text)) {
    Fatal("Write proto to text error!");
  }
}

void WriteProtoToTextFile(const Message& proto, const std::string& proto_file) {
  int fd = open(proto_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* file = new FileOutputStream(fd);
  if (!TextFormat::Print(proto, file)) {
    Fatal("Write proto to text file error!");
  }
  delete file;
  close(fd);
}

void WriteProtoToBinaryFile(const Message& proto,
                            const std::string& proto_file) {
  std::ofstream file(proto_file,
                     std::ios::out | std::ios::trunc | std::ios::binary);
  if (!proto.SerializeToOstream(&file)) {
    Fatal("Write proto to binary file error!");
  }
}

}  // namespace IO
