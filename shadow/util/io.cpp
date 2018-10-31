#include "io.hpp"
#include "log.hpp"

#if defined(USE_Protobuf)
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#elif defined(_WIN32)
#pragma warning(disable : 4996)
#include <io.h>
#endif
#include <fcntl.h>
#include <fstream>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#if defined(SUPPORT_JSON)
#include <google/protobuf/util/json_util.h>
#endif

#else
#include "core/parser.hpp"
#include "util/util.hpp"
#endif

namespace Shadow {

namespace IO {

#if defined(USE_Protobuf)
using google::protobuf::TextFormat;
using google::protobuf::io::ArrayInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::ZeroCopyOutputStream;

bool ReadProtoFromText(const std::string& proto_text, Message* proto) {
  return TextFormat::ParseFromString(proto_text, proto);
}

bool ReadProtoFromTextFile(const std::string& proto_file, Message* proto) {
  int fd = open(proto_file.c_str(), O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << proto_file;
  auto* input = new FileInputStream(fd);
  bool success = TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

bool ReadProtoFromBinaryFile(const std::string& proto_file, Message* proto) {
#if defined(_WIN32)
  int fd = open(proto_file.c_str(), O_RDONLY | O_BINARY);
#else
  int fd = open(proto_file.c_str(), O_RDONLY);
#endif
  CHECK_NE(fd, -1) << "File not found: " << proto_file;
  auto* file_input = new FileInputStream(fd);
  auto* coded_input = new CodedInputStream(file_input);
  coded_input->SetTotalBytesLimit(INT_MAX, 1073741824);
  bool success = proto->ParseFromCodedStream(coded_input) &&
                 coded_input->ConsumedEntireMessage();
  delete coded_input;
  delete file_input;
  close(fd);
  return success;
}

bool ReadProtoFromArray(const void* proto_data, int proto_size,
                        Message* proto) {
  auto* array_input = new ArrayInputStream(proto_data, proto_size);
  auto* coded_input = new CodedInputStream(array_input);
  coded_input->SetTotalBytesLimit(INT_MAX, 1073741824);
  bool success = proto->ParseFromCodedStream(coded_input) &&
                 coded_input->ConsumedEntireMessage();
  delete coded_input;
  delete array_input;
  return success;
}

void WriteProtoToText(const Message& proto, std::string* proto_text) {
  CHECK(TextFormat::PrintToString(proto, proto_text))
      << "Write proto to text error!";
}

void WriteProtoToTextFile(const Message& proto, const std::string& proto_file) {
  int fd = open(proto_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  auto* file_output = new FileOutputStream(fd);
  CHECK(TextFormat::Print(proto, file_output))
      << "Write proto to text file error!";
  delete file_output;
  close(fd);
}

void WriteProtoToBinaryFile(const Message& proto,
                            const std::string& proto_file) {
  std::ofstream file(proto_file,
                     std::ios::out | std::ios::trunc | std::ios::binary);
  CHECK(proto.SerializeToOstream(&file)) << "Write proto to binary file error!";
}

void WriteProtoToArray(const Message& proto, void* proto_data) {
  int proto_size = proto.ByteSize();
  CHECK(proto.SerializeToArray(proto_data, proto_size))
      << "Write proto to array error!";
}

#if defined(SUPPORT_JSON)
using google::protobuf::util::JsonPrintOptions;
using google::protobuf::util::MessageToJsonString;
using google::protobuf::util::Status;

void WriteProtoToJsonText(const Message& proto, std::string* json_text,
                          bool compact) {
  JsonPrintOptions options;
  options.add_whitespace = !compact;
  options.preserve_proto_field_names = true;
  CHECK(MessageToJsonString(proto, json_text, options).ok())
      << "Write proto to json text error!";
}
#endif

#else
bool ReadProtoFromText(const std::string& proto_text, shadow::NetParam* proto) {
  Parser::ParseNet(proto_text, proto);
  return true;
}

bool ReadProtoFromTextFile(const std::string& proto_file,
                           shadow::NetParam* proto) {
  return ReadProtoFromText(Util::read_text_from_file(proto_file), proto);
}
#endif

}  // namespace IO

}  // namespace Shadow
