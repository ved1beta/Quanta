import json 
import shlex 
import subprocess

import torch

def outlier_hook(module , imput):
    assert isinstance(module , torch.nn.Linear)
    tracer = OutlierTracer.get_instance()
    hvalue = tracer.get_hvalue(module.weight)
    if hvalue not in tracer.hvalue2outlier_idx: 
        outlier_idx = find_outlier_dims(module.weight)
        tracer.outliers.append(outlier_idx)
        tracer.hvalues.append(hvalue)
        if len(tracer.outliers) > 1:
            # assign the current layer the outlier idx found from the weight
            # of the previous linear layer
            if tracer.outliers[-1].numel() > 0:
                assert tracer.outliers[-1].max() < module.weight.shape[1]
            tracer.hvalue2outlier_idx[hvalue] = tracer.outliers[-1]

        else:
            # first layer, we cannot use the weight for outlier detection
            # we follow a mixed approach:
            # (1) zscore test of std of hidden dimension
            # (2) magnitude > 6 test
            merged = input[0].view(-1, input[0].shape[-1])
            # (1) zscore test of std of hidden dimension
            outlier_idx = find_outlier_dims(merged, reduction_dim=1, zscore=3)
            # (2) magnitude > 6 test
            dims = (torch.abs(input[0]) > 6).sum(dim=list(range(len(input[0].shape) - 1)))
            outlier_idx2 = torch.where(dims > 0)[0]
            outlier_idx = torch.cat([outlier_idx, outlier_idx2]).unique()
            tracer.hvalue2outlier_idx[hvalue] = outlier_idx
    else:
        for hook in tracer.hooks:
            hook.remove()

class OutlierTracer:
    _instance = None 
    def __init__(self):
        raise RuntimeError("Call get_instance() instead")
    
    def initialize(self, model):
        self.last_w = None
        self.current_outlier_dims = None
        self.hvalues = []
        self.outliers = []
        self.hvalue2outlier_idx = {}
        self.initialized = True
        self.hooks = []

        for n , m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                self.hooks.append(m.register_forward_pre_hook(outlier_hook))
    def is_initialized(self):
        return getattr(self, "initialized", False)
    
    def get_hvalue(self , weight):
        return weight.data.storage().data_ptr()
    
    def get_outliers(self, weight):
        if not self.is_initialized():
            print("Outlier tracer is not initialized...")
            return None
        hvalue = self.get_hvalue(weight)
        if hvalue in self.hvalue2outlier_idx:
            return self.hvalue2outlier_idx[hvalue]
        else:
            return N