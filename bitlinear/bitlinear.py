from collections.abc import Sequence
from math import ceil
import re
import torch
import torch.nn as nn

from kernels import TorchLinear
from measures import AbsMax, AbsMedian, AbsMean

def round_clamp(input, range):
    return (input.round().clamp(range[0], range[1]) - input).detach() + input

def scale(input, range, measure, keepdim, eps):
    return max(abs(k) for k in range) / measure(input.detach(), keepdim=keepdim).clamp_(min=eps)

def range_from_bits(bits):
    return (ceil(-2**(bits-1)), ceil(2**(bits-1)-1))

def sample(input, range):
    if range[0] != -1 or range[1] != 1:
        return round_clamp(input, range)
    rand = torch.rand(input.size(), device=input.device, dtype=input.dtype)
    result = input.sign() * torch.where(input.abs() < rand, 0, 1)
    return (result-input).detach() + input

class BitLinear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            eps=1e-5,
            weight_range=1.58,
            weight_measure="AbsMedian",
            activation_range=8,
            activation_measure="AbsMax",
            kernel="TorchLinear",
            strategy="round_clamp",
        ):
        super(BitLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.eps = eps
        self.weight_range = weight_range if isinstance(weight_range, Sequence) else range_from_bits(weight_range)
        self.weight_measure = eval(weight_measure)() if isinstance(weight_measure, str) else weight_measure
        self.activation_range = activation_range if isinstance(activation_range, Sequence) else range_from_bits(activation_range)
        self.activation_measure = eval(activation_measure)() if isinstance(activation_measure, str) else activation_measure
        self.kernel = eval(kernel)() if isinstance(kernel, str) else kernel
        self.strategy = eval(strategy) if isinstance(strategy, str) else strategy

    def __repr__(self):
        return f"BitLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, eps={self.eps}, weight_range={self.weight_range}, weight_measure={self.weight_measure}, activation_range={self.activation_range}, activation_measure={self.activation_measure}, kernel={self.kernel}, strategy={self.strategy})"

    def forward(self, x):
        if self.activation_measure is None:
            x_scale, x_quant = 1, x
        else:
            x_norm = torch.layer_norm(x, x.size()[1:])
            x_scale = scale(x_norm, self.activation_range, self.activation_measure, True, self.eps)
            x_quant = self.strategy(x_norm * x_scale, self.activation_range)
        if self.weight_measure is None:
            w_scale, w_quant = 1, self.weight
        else:
            w_scale = scale(self.weight, self.weight_range, self.weight_measure, False, self.eps)
            w_quant = self.strategy(self.weight * w_scale, self.weight_range)
        y_quant = self.kernel(x_quant, w_quant, self.bias)
        y = y_quant / (w_scale * x_scale)
        return y

class FrozenBitLinear(nn.Linear):
    def __init__(self, bitlinear, pack_weights):
        bias = bitlinear.bias is not None
        super(FrozenBitLinear, self).__init__(
            in_features=bitlinear.in_features,
            out_features=bitlinear.out_features,
            bias=bias,
            # device=bitlinear.device,
            #dtype=bitlinear.dtype,
        )
        self.eps = bitlinear.eps
        self.activation_range=bitlinear.activation_range
        self.activation_measure=bitlinear.activation_measure
        self.weight_measure=bitlinear.weight_measure
        self.strategy=bitlinear.strategy
        self.kernel=bitlinear.kernel
        self.pack_weights=pack_weights
        
        if bias:
            self.bias.data = bitlinear.bias.data
        if self.weight_measure is None:
            self.w_scale, self.w_quant = 1, bitlinear.weight
        else:
            self.w_scale = scale(bitlinear.weight, bitlinear.weight_range, bitlinear.weight_measure, False, self.eps)
            self.w_quant = self.strategy(bitlinear.weight * self.w_scale, bitlinear.weight_range)

        if self.pack_weights:
            self.w_quant = self.w_quant.to(dtype=torch.int8)
            self.w_quant, self.w_quant, self.padding_size = self.pack_matrix(self.w_quant)
             
        self.weight = None
            # if isinstance(self.w_quant, torch.nn.Parameter):
            #   self.weight = self.w_quant
            # else:
            #   self.weight = nn.Parameter(self.w_quant)
        
    def __repr__(self):
        return f"FrozenBitLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, kernel={self.kernel}), activation_range={self.activation_range}, activation_measure={self.activation_measure}"
    
    def forward(self, x):
        if self.activation_measure is None:
            x_scale, x_quant = 1, x
        else:
            x_norm = torch.layer_norm(x, x.size()[1:])
            x_scale = scale(x_norm, self.activation_range, self.activation_measure, True, self.eps)
            x_quant = self.strategy(x_norm * x_scale, self.activation_range)
        y_quant = self.kernel(x_quant, self.w_quant, self.bias)
        y = y_quant / (self.w_scale * x_scale)
        return y


    def pack_matrix(self, matrix):
        # Flatten the input matrix
        flattened = matrix.view(-1)

        padding_size = (4 - (flattened.size(0) % 4)) % 4 # Compute no. padding elements to make row multiple of 4. %4 ensures that we don't add anything (3when pad elements is multiple of 4 -> (4-0) % 4 = 0
        if padding_size > 0:
            flattened = torch.cat([flattened, torch.zeros(padding_size, dtype=torch.int8)])

        packed = torch.zeros((flattened.size(0) // 4,), dtype=torch.int8) # init to hold packed values

        # Pack values in groups of 4 2-bit values into int8
        # 0b11 -> 3 -> 11
        # << shift LSB bits into positions <<6 takes 2 LSB bits and shift them to two most significant
        for i in range(packed.size(0)):
            chunk = flattened[i * 4:(i + 1) * 4] 
            packed[i] = ((chunk[0] & 0b11) << 6) | ((chunk[1] & 0b11) << 4) | ((chunk[2] & 0b11) << 2) | (chunk[3] & 0b11)

        return packed, matrix.shape, padding_size


    def unpack_matrix(self, packed_matrix, original_shape, padding_size):
        # Initialize unpacked matrix (flattened)
        unpacked = torch.zeros(packed_matrix.size(0) * 4, dtype=torch.int8)

        # Unpack each int8 into four 2-bit values
        for i, packed_value in enumerate(packed_matrix):
            unpacked[i * 4 + 0] = (packed_value >> 6) & 0b11
            unpacked[i * 4 + 1] = (packed_value >> 4) & 0b11
            unpacked[i * 4 + 2] = (packed_value >> 2) & 0b11
            unpacked[i * 4 + 3] = packed_value & 0b11

        # Replace binary representation back into -1, 0, and 1
        unpacked[unpacked == 3] = -1  # Since 11 (binary) is -1 in 2-bit signed

        # Remove padding if it was added
        if padding_size > 0:
            unpacked = unpacked[:-padding_size]

        # Reshape back to the original shape
        return unpacked.view(original_shape)


def freeze(
    model, 
    pack_weights=False,
    ):
    """
        Parameters:
            model : torch.nn.Module
                Model with BitLinear modules to replace with FrozenBitLinear for faster inference
            kernel : str or Kernel
                Forward kernel to implement in the model
            weightMeasure : str
                str corresponding to a weight packing method
                options are 'AbsMean', 'AbsMax', and 'AbsMedian'
                default is 'AbsMean'
            eps : float
                value to clamp the weights to in the scaling step to avoid divide-by-zero
                default is 1e-5
            activation_measure : str
                str corresponding to the activation quantization method
                options are 'AbsMean', 'AbsMax', 'AbsMedian', and 'Fp16'
                'AbsMax'
            activation_range : int
                number of bits to represent the activations in
                default is 8
            device : Optional(torch.device)
            dtype : Optional(torch.dtype)

        Returns:
            None
        
        This function replaces all of the BitLinear instances in a model with FrozenBitLinear in order to
        speed up inference. The weights are packed corresponding to the kernel.
        """
    
    for name, module in model.named_children():
        if isinstance(module, BitLinear):
            frozen_module = FrozenBitLinear(module, pack_weights=pack_weights)
            setattr(model, name, frozen_module)
            
        else: # recursively iterate throughout the rest of the module
            freeze(module, pack_weights=pack_weights)


def replace_modules(model, old_class=nn.Linear, new_class=BitLinear, new_class_kwargs={}, match_name="", prefix=""):
    for name, module in model.named_children():
        qual_name = prefix + "." + name
        if isinstance(module, old_class) and re.search(match_name, qual_name) is not None:
            kwargs = dict(new_class_kwargs)
            kwargs["in_features"] = module.in_features
            kwargs["out_features"] = module.out_features
            bias = getattr(module, "bias", None) is not None
            kwargs["bias"] = bias
            new_module = new_class(**kwargs)
            new_module.weight.data = module.weight.data
            if bias:
                new_module.bias.data = module.bias.data
            setattr(model, name, new_module)
        else:
            replace_modules(module, old_class, new_class, new_class_kwargs, match_name, prefix=qual_name)

def bitlinearize(model, old_class=nn.Linear, new_class=BitLinear, replacements=[{}]):
    for replacement in replacements:
        replacement = dict(replacement)
        match_name = replacement.pop("match_name", "")
        replace_modules(
            model=model,
            old_class=old_class,
            new_class=new_class,
            new_class_kwargs=replacement,
            match_name=match_name,
        )
    return model