#ifndef SHADOW_PARSER_HPP
#define SHADOW_PARSER_HPP

#include "network.hpp"

#include <string>
#include <vector>

class Parser {
public:
  void ParseNetworkProto(Network *net, std::string prototxt_file, int batch);
  void LoadWeights(Network *net, std::string weight_file);

private:
  void ParseNet(Network *net);
  Layer *LayerFactory(shadow::LayerParameter layer_param,
                      shadow::BlobShape *shape);

  void LoadWeightsUpto(Network *net, std::string weight_file, int cut_off);
};

#endif // SHADOW_PARSER_HPP
