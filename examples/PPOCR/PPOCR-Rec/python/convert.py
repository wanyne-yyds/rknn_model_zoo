<<<<<<< HEAD
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
from tabnanny import verbose
from rknn.api import RKNN


DATASET_PATH = '../../../../datasets/PPOCR/imgs/dataset_20.txt'
DEFAULT_RKNN_PATH = '../model/ppocrv4_rec.rknn'
DEFAULT_QUANT = False

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]));
        print("       platform choose from [rk3562,rk3566,rk3568,rk3588]")
        print("       dtype choose from    [i8, fp]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type == 'i8':
            assert False, "i8 PPOCR-Rec got accuracy drop yet!"
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    return model_path, platform, do_quant, output_path

if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()

    model = RKNN(verbose=False)

    # Config
    model.config(
        target_platform=platform,
        op_target={'p2o.Add.235_shape4':'cpu', 'p2o.Add.245_shape4':'cpu', 'p2o.Add.255_shape4':'cpu',
                   'p2o.Add.265_shape4':'cpu', 'p2o.Add.275_shape4':'cpu'}
    )

    # Load ONNX model
    ret = model.load_onnx(model=model_path)
    assert ret == 0, "Load model failed!"

    # Build model
    ret = model.build(
        do_quantization=do_quant)
    assert ret == 0, "Build model failed!"

    # Init Runtime
    # ret = model.init_runtime()
    # assert ret == 0, "Init runtime environment failed!"

    # Export
    if not os.path.exists(os.path.dirname(output_path)):
        os.mkdir(os.path.dirname(output_path))

    ret = model.export_rknn(
        output_path)
    assert ret == 0, "Export rknn model failed!"
    print("Export OK!")
=======
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
from rknn.api import RKNN

DEFAULT_RKNN_PATH = '../model/ppocrv4_rec.rknn'
DEFAULT_QUANT = False
RKNPU1_PLATFORM = ['rk1808', 'rv1109', 'rv1126']

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]));
        print("       platform choose from [rk3562, rk3566, rk3568, rk3588, rk1808, rv1109, rv1126]")
        print("       dtype choose from    [fp] for [rk3562,rk3566,rk3568,rk3588]")
        print("       dtype choose from    [fp] for [rk1808,rv1109,rv1126]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    return model_path, platform, do_quant, output_path

if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    if platform in RKNPU1_PLATFORM:
        rknn.config(target_platform=platform)
    else:
        rknn.config(
            target_platform=platform,
            # In order to improve accuracy, these nodes need to fallback to CPU on RKNPU2 platform.
            op_target={'p2o.Add.235_shape4':'cpu', 'p2o.Add.245_shape4':'cpu', 'p2o.Add.255_shape4':'cpu',
                    'p2o.Add.265_shape4':'cpu', 'p2o.Add.275_shape4':'cpu'}
        )

    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
>>>>>>> 7a6984bc0e5527a9e384a0a98080fcbd55a9e1ad
