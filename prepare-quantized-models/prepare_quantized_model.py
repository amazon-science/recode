

# dependency: install from source following https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization


import torch
import os
from torch import nn
from transformers.modeling_utils import Conv1D
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import RobertaTokenizer, T5ForConditionalGeneration

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules

import copy
import json


def _conv1d_to_linear(module):
    in_size, out_size = module.weight.shape
    linear = torch.nn.Linear(in_size, out_size)
    linear.weight.data = module.weight.data.T.contiguous()
    linear.bias.data = module.bias.data
    return linear


def conv1d_to_linear(model):
    """in-place
    This is for Dynamic Quantization, as Conv1D is not recognized by PyTorch, convert it to nn.Linear
    """
    print("replace Conv1D with Linear")
    for name in list(model._modules):
        module = model._modules[name]
        if isinstance(module, Conv1D):
            linear = _conv1d_to_linear(module)
            model._modules[name] = linear
        else:
            conv1d_to_linear(module)


def copy_linear(module):
    in_size, out_size = module.weight.shape
    linear = torch.nn.Linear(in_size, out_size)
    linear.weight.data = module.weight.data
    linear.bias.data = module.bias.data
    return linear



def compute_amax(model, **kwargs):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")


import argparse

parser = argparse.ArgumentParser(description='My ML model')

parser.add_argument('--model_name', type=str, default='facebook/incoder-6B',
                    help='Name of the model (should match HF model name)')

parser.add_argument('--output_folder_path', type=str, default='./output',
                    help='Path to output folder')
                    
parser.add_argument('--num_calibration_samples', type=int, default=500,
                    help='Number of samples to use for calibration')

parser.add_argument('--input_length', type=int, default=512)

parser.add_argument('--per_column', action='store_true',
                    help='should we use per_column or per_tensor quantization')

parser.add_argument('--if_quantize_lm_head', action='store_true',
                    help='if we should quantize lm_head, we suggest false')

parser.add_argument('--calibration_method', type=str, help='choose between mse/entropy and percenile')

parser.add_argument('--data_file', type=str, help='jsonl file with some calibrationd data')

parser.add_argument('--quant_type', type=str, help='choose between static/dynamic')






def main():
    args = parser.parse_args()
    quant_type = args.quant_type
    w_bits = 8
    a_bits = 8
    file_name = args.data_file
    num_cali_samples = args.num_calibration_samples
    cali_input_length = args.input_length
    axis = 0 if args.per_tensor else None
    cali_method = args.calibration_method
    if cali_method is 'percentile':
        cali_method = 99

    output_model_name = args.model_name.replace('/', '_').replace('-','_') + '_w' + str(w_bits) + 'a' + str(a_bits) + '_cali' + str(num_cali_samples) + '_' + str(cali_method)
    if axis is not None:
        output_model_name = output_model_name + '_axis' + str(axis)

    output_model_name = output_model_name + '.pt'
    path_to_quantized = os.path.join(args.output_folder_path, quant_type, output_model_name)


    if 'codet5' in args.model_name:
        tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
        model_copy = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model_copy = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    if quant_type == 'static':

        device = 'cuda'
        quant_desc_input = QuantDescriptor(num_bits=a_bits, axis=None, calib_method='histogram')

        if axis is None:
            quant_desc_weight = QuantDescriptor(num_bits=w_bits, axis=None, calib_method='max')
        else:
            quant_desc_weight = QuantDescriptor(num_bits=w_bits, axis=(axis), calib_method='max')


        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)


        quant_modules.initialize()
        if 'codet5' in args.model_name:
            model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name)

        quant_modules.deactivate()
        # not quantize lm_head
        if not args.if_quantize_lm_head:
            model.lm_head = copy.deepcopy(model_copy.lm_head)

        del model_copy

        # Enable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        # collect stats
        cnt = 0
        model.to(device)
        model.eval()
        skip_lines = 5
        line_cnt = 0
        with open(file_name, 'r') as f:
            for line in f:
                line_cnt += 1
                if line_cnt % skip_lines != 0:
                    continue
                sample = json.loads(line)
                #print(sample['original_string'])
                inputs = tokenizer(sample['original_string'][:cali_input_length], return_tensors="pt").to(device)
                output = model.generate(input_ids=inputs.input_ids, max_new_tokens=max_new_tokens)
                #print(tokenizer.decode(output[0], skip_special_tokens=True))
                cnt += 1
                if cnt % 100 == 0:
                    print(str(cnt) + ' samples processed')
                if cnt >= num_cali_samples:
                    break

        # disable calib since stats is collected
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

        with torch.no_grad():
            if str(cali_method) in ('mse', 'entropy'):
                compute_amax(model, method=cali_method)
            else:
                compute_amax(model, method="percentile", percentile=cali_method)

    else:

        # this is the call that does the work
        model = torch.quantization.quantize_dynamic(
            model_copy, {nn.Linear}, dtype=torch.qint8)
        
    
    torch.save(model, path_to_quantized)
    

