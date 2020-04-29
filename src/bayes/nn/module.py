import torch
import torch.nn.functional as F
import itertools
from collections import OrderedDict


class TensorModule:
    def __init__(self):
        self.training = True
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self._tmodules = OrderedDict()
        self._buffers = OrderedDict()

    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)
        return result

    def forward(self, *input):
        raise NotImplementedError

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        params = self.__dict__.get('_parameters')
        if name.startswith('p_') and isinstance(value, torch.Tensor):
            if params is None:
                raise AttributeError
            remove_from(self.__dict__, self._buffers, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, TensorModule):
                if modules is None:
                    raise AttributeError
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]

        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]

        if '_modules' in self.__dict__:
            _modules = self.__dict__['_modules']
            if name in _modules:
                return _modules[name]

    def register_parameter(self, name, param):
        # debug
        if name in self._parameters:
            prev_param = self._parameters[name]
        else:
            prev_param = None
        # print('%s: %s -> %s' % (name, str(prev_param), str(param)))

        if '.' in name:
            raise KeyError

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, torch.Tensor):
            raise TypeError
        else:
            self._parameters[name] = param

    def register_buffer(self, name, tensor):
        if '.' in name:
            raise KeyError
        elif tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError
        else:
            self._buffers[name] = tensor

    def named_modules(self, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [
            (prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=True
        )
        for elem in gen:
            yield elem

    def _load_from_state_dict(self, state_dict, prefix):
        local_name_params = itertools.chain(
            self._parameters.items(), self._buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                setattr(self, name, input_param)
                # with torch.no_grad():
                # param.copy_(input_param)

    def set_params(self, state_dict):
        def load(module, prefix=''):
            module._load_from_state_dict(state_dict, prefix)

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(self)
        load = None

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def children(self):
        for name, module in self.named_children():
            yield module

    def named_children(self):
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module


class TensorLinear(TensorModule):
    def __init__(self, in_features, out_features, bias=True):
        super(TensorLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        # self.weight = TensorParameter((out_features, in_features))
        self.p_weight = torch.zeros(
            (out_features, in_features), dtype=torch.float32)
        if bias:
            self.p_bias = torch.zeros((out_features, ), dtype=torch.float32)
        else:
            # original self.register_parameter('bias', None)
            self.p_bias = None

    def forward(self, input):
        return F.linear(input, self.p_weight, self.p_bias)
