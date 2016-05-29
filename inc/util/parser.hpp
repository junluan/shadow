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
  Layer *ParseData(shadow::LayerParameter layer_param, SizeParams params);
  Layer *ParseConvolution(shadow::LayerParameter layer_param,
                          SizeParams params);
  Layer *ParsePooling(shadow::LayerParameter layer_param, SizeParams params);
  Layer *ParseConnected(shadow::LayerParameter layer_param, SizeParams params);
  Layer *ParseDropout(shadow::LayerParameter layer_param, SizeParams params);

  void LoadWeightsUpto(Network *net, std::string weightfile, int cutoff);
};

#endif // SHADOW_PARSER_HPP
