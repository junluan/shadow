# Convert To Shadow Model

## Notice

Clone the Shadow repository, and we'll call the directory that you cloned Shadow as ```${Shadow_ROOT}```.

```
git clone https://github.com/junluan/shadow.git
```

If you are going to convert a caffe model, please copy ```caffe.proto``` to ```${Shadow_ROOT}/shadow/python/proto``` directory and generate caffe protobuf's python source file.

```
cd ${Shadow_ROOT}/shadow/python/proto
protoc --python_out=./ caffe.proto
```

## Convert
 
The working directory for converting models is ```${Shadow_ROOT}/shadow/python```
 
```
cd ${Shadow_ROOT}/shadow/python
```

Any models must have a config function to return some params, please check and add your config functions in ```${Shadow_ROOT}/shadow/python/config/config_customs.py```. There are some example config functions in ```${Shadow_ROOT}/shadow/python/config/config_examples.py```.

```python
def get_config_custom():
    net_info = {
        'input_name': ['data'],  # net input blob names
        'input_shape': [[1, 3, 224, 224]],  # net input blob shapes
        'mean_value': [],  # data mean values
        'scale_value': [1],  # data scale values
        'arg': {  # net arguments, must end with one of 's_i, s_f, s_s, v_i, v_f, v_s'
            'num_classes_s_i': 1000,  # class numbers
            'out_blob_v_s': [],  # net output blob names
            'is_bgr_s_i': True  # other useful arguments
        }
    }

    meta_net_info = {
        'model_type': 'mxnet',  # model type: caffe or mxnet
        'model_name': ['squeezenet_v1.1'],  # model file name on disk
        'model_epoch': [0],  # only for mxnet model
        'save_name': 'squeezenet_v1.1',  # shadow model saved name
        'network': [net_info]  # networks
    }

    return meta_net_info
```

There are six types of arguments: ```s_i(single int)```, ```s_f(single float)```, ```s_s(single string)```, ```v_i(vector int)```, ```v_f(vector float)```, ```v_s(vector string)```. Any argument should have one of the above suffixes. For example, if the argument name is ```Arg_Name``` and the argument type is ```v_f```, then the config name should be ```Arg_Name_v_f```. 

Now you can convert a custom model after you finish the config function. The ```--config_name``` must be the same as config function's suffix name ```get_config_***```. You can either add ```--copy_params``` to copy weights or ```--merge_op``` to merge some operators.

```
python convert_to_shadow.py --model_root model_mxnet --config_name custom --save_root model_shadow
```

Python is snail! If you copy params, go and take a tea break.


