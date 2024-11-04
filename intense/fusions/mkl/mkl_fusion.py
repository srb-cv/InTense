from typing import Union, Tuple, List
import logging
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import math

from .normalization_module import VectorWiseBatchNorm, Normalize3

class MROFusion(nn.Module):
    def __init__(self,
                 in_features_dict: dict[str, int],
                 out_features: int,
                 ) -> None:
        super().__init__()
        self.in_features = in_features_dict
        self.out_features = out_features
        self.norm_layers = nn.ModuleDict({
            key:nn.BatchNorm1d(value)
            for (key, value) in self.in_features.items()
        })
        self.fusion_layers = nn.ModuleDict({
            key:nn.Linear(value, self.out_features)
            for (key, value) in self.in_features.items()
        })
        
        
    def forward(self, tensors: dict[str, torch.Tensor], outs_pre_tf: dict[str, torch.Tensor]):
        outs = {key:self.norm_layers[key](value)
                for (key,value) in tensors.items()}
        outs = {key:self.fusion_layers[key](value)
                for (key,value) in outs.items()}        
        outs_uni = sum([value for (key, value) in outs.items() if len(key) == 1])
        outs_bi = sum([value for (key, value) in outs.items() if len(key) == 2])
        outs_tri = sum([value for (key, value) in outs.items() if len(key) == 3])
        return {'out_uni': outs_uni,
                'out_bi': outs_bi,
                'out_tri': outs_tri}

class MKLFusion(nn.Module):
    def __init__(
        self,
        # in_tensors: int,
        in_features: dict[str, int],
        out_features: int,
        bias: bool = True,
    ):
        """Initialize the MKLFusion module
        Args:
            in_features(list[int]): Each element of the list is the dimension of a
                                    modality's feature-representation, which is an input to
                                    the final fusion layer
            out_features(int): Final output which depends on the number if classes
                                in the dataset
        Returns:
            A nn.Module to fuse the mdalities
        """
        # TODO: add an option to pass the hyperparameter p here
        super().__init__()
        # self.in_tensors = in_tensors
        self.in_features = in_features
        self.out_features = out_features
        self.fusion_layer = nn.Linear(
            in_features=sum(in_features.values()),
            out_features=out_features,
            bias=bias
        )
        # self.dropout = nn.Dropout(p=0.5)

    @property
    def bias(self):
        return self.fusion_layer.bias

    @property
    def weight(self):
        return torch.split(self.fusion_layer.weight,
                           list(self.in_features.values()),
                           dim=1)

    def forward(self, tensors: Union[Tuple[torch.Tensor, ...], List[Tensor]]):
        tensors = torch.cat(tensors, dim=1)
        # tensors = self.dropout(tensors)
        return self.fusion_layer(tensors)

    def regularizer(self, p=1):
        q = 2 * p / (p + 1)
        return torch.sum(self.weight_norms() ** q) ** (2 / q)

    def scores(self, p=1):
        with torch.no_grad():
            norms = self.weight_norms()
            a = norms ** (2 / (p + 1))
            b = torch.sum(norms ** (2 * p / (p + 1))) ** (1 / p)
            scores = a / b
            scores = scores.numpy()
            return dict(zip(self.in_features.keys(), scores))

    def weight_norms(self):
        return torch.tensor(
            [torch.linalg.matrix_norm(tens) for tens in self.weight]
        )


# class MKLFusion(nn.Module):
#     def __init__(
#         self,
#         # in_tensors: int,
#         in_features: list[int],
#         out_features: int,
#         bias: bool = True,
#     ):
#         """Initialize the MKLFusion module
#         Args:
#             in_features(list[int]): Each element of the list is the dimension of a
#                                     modality's feature-representation, which is an input to
#                                     the final fusion layer
#             out_features(int): Final output which depends on the number if classes
#                                 in the dataset
#         Returns:
#             A nn.Module to fuse the mdalities
#         """
#         # TODO: add an option to pass the hyperparameter p here
#         super().__init__()
#         # self.in_tensors = in_tensors
#         self.in_features = in_features
#         self.out_features = out_features
#         self.fusion_layer = nn.Linear(
#             in_features=sum(in_features), out_features=out_features, bias=bias
#         )
#         # self.dropout = nn.Dropout(p=0.5)

#     @property
#     def bias(self):
#         return self.fusion_layer.bias

#     @property
#     def weight(self):
#         return torch.split(self.fusion_layer.weight, self.in_features, dim=1)

#     def forward(self, tensors: Union[Tuple[torch.Tensor, ...], List[Tensor]]):
#         tensors = torch.cat(tensors, dim=1)
#         # tensors = self.dropout(tensors)
#         return self.fusion_layer(tensors)

#     def regularizer(self, p=1):
#         q = 2 * p / (p + 1)
#         return torch.sum(self.weight_norms() ** q) ** (2 / q)

#     def scores(self, p=1):
#         with torch.no_grad():
#             norms = self.weight_norms()
#             a = norms ** (2 / (p + 1))
#             b = torch.sum(norms ** (2 * p / (p + 1))) ** (1 / p)
#             return a / b

#     def weight_norms(self):
#         return torch.tensor(
#             [torch.linalg.matrix_norm(tens) for tens in self.weight]
#         )  # shape: (in_tensors)


class MKLFusionBatchNorm(MKLFusion):
    def __init__(
        self,
        in_features_dict: dict[str, int],
        out_features: int,
        bias: bool = True,
        affine=False,
        activation=False,
    ):
        super().__init__(in_features_dict, out_features, bias)
        self.batch_norm = nn.BatchNorm1d(
            num_features=sum(in_features_dict.values()),
            affine=False)
        self.affine = affine
        self.activation = activation
        self.in_features = in_features_dict
        if self.activation:
            self.act = nn.ReLU()
        if self.affine:
            self.scale_factor = Parameter(torch.empty(1, dtype=torch.float))
            self.scale_bias = Parameter(torch.empty(1, dtype=torch.float))
        else:
            self.register_parameter("scale_factor", None)
            self.register_parameter("scale_bias", None)
        self.reset_parameters()

    def forward(self, tensors: dict[str, torch.Tensor], *args):
        try:
            tensors_list = list(tensors.values())
        except AttributeError:
            tensors_list = tensors
        out_tens: Tensor = torch.cat(tensors, dim=1)
        if self.activation:
            out_tens = self.act(out_tens)
        out_tens = self.batch_norm(out_tens)
        # list_tens: list[Tensor] = torch.split(out_tens, self.in_features, dim=1)
        # list_tens = [
        #     torch.div(list_tens[i], math.sqrt(self.in_features[i]))
        #     for i in range(len(list_tens))
        # ]
        # out: Tensor = torch.cat(list_tens, dim=1)
        if self.affine:
            out_tens = out_tens * self.scale_factor + self.scale_bias
        return self.fusion_layer(out_tens)

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.scale_factor)
            nn.init.zeros_(self.scale_bias)


class MKLFusionVectorwiseBatchNorm(MKLFusion):
    def __init__(
        self,
        in_features_dict: dict[str, int],
        out_features: int,
        bias: bool = True,
        affine=False,
        activation=False,
    ):
        in_features = in_features_dict
        super().__init__(in_features, out_features, bias)
        self.activation = activation
        self.affine = affine
        self.in_features_dict = in_features_dict
        self.vbn_modules = nn.ModuleDict(self._get_vbn_modules())
        self.two_fusion_indices = [
            idx for idx in in_features_dict.keys() if len(idx) <= 2
        ]
        self.three_fusion_indices = [
            idx for idx in in_features_dict.keys() if len(idx) == 3
        ]
        logging.info(f"two fusion indices: {self.two_fusion_indices}")
        logging.info(f"three fusion indices: {self.three_fusion_indices}")
        if activation:
            self.act = nn.ReLU()
        if self.affine:
            self.scale_factor = Parameter(torch.empty(1, dtype=torch.float))
            self.scale_bias = Parameter(torch.empty(1, dtype=torch.float))
        else:
            self.register_parameter("scale_factor", None)
            self.register_parameter("scale_bias", None)
        self.reset_parameters()

    def forward(
        self, tensors: dict[str, torch.Tensor], pre_fusion_tens: dict[str, torch.Tensor]
    ):
        if self.activation:
            tensors = {
                mod_idx: self.act(feature) for (mod_idx, feature) in tensors.items()
            }
        out_3 = [
            self.vbn_modules[mod_idx](tensors, pre_fusion_tens)
            for mod_idx in self.three_fusion_indices
        ]
        out = [
            self.vbn_modules[mod_idx](tensors[mod_idx])
            for mod_idx in self.two_fusion_indices
        ]
        out = torch.cat(out + out_3, dim=1)
        if self.affine:
            out = out * self.scale_factor + self.scale_bias
        return self.fusion_layer(out)

    def _get_vbn_modules(self) -> dict[str, nn.Module]:
        vbn_modules = {}
        for mod_idx, feature_dim in self.in_features_dict.items():
            if len(mod_idx) <= 2:
                module = VectorWiseBatchNorm(num_features=feature_dim)
            elif len(mod_idx) == 3:
                assert mod_idx == "123"
                module = Normalize3(
                    feature_dim_dict=self.in_features_dict, mod_index=mod_idx
                )
            else:
                raise NotImplementedError(
                    "Normalization not implemented for more than 3 in tensor fusion"
                )
            vbn_modules[mod_idx] = module
        return vbn_modules

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.scale_factor)
            nn.init.zeros_(self.scale_bias)


class MKLFusionVectorwiseBatchNorm_V1(MKLFusion):
    '''
    Depreacated version. This supports proper Normalization
    for only upto two modality interactions
    '''
    def __init__(
            self,
            in_features_dict: dict[str, int],
            out_features: int,
            bias: bool = True,
            affine=False,
            activation=False):
        self.in_features = in_features_dict
        print("In features: ", self.in_features)
        super().__init__(self.in_features, out_features, bias)
        self.activation = activation
        self.affine = affine
        self.vbn_modules = nn.ModuleList(self._get_vbn_modules())
        if activation:
            self.act = nn.ReLU()
        if self.affine:
            self.scale_factor = Parameter(torch.empty(1, dtype=torch.float))
            self.scale_bias = Parameter(torch.empty(1, dtype=torch.float))
        else:
            self.register_parameter("scale_factor", None)
            self.register_parameter("scale_bias", None)
        self.reset_parameters()

    def forward(
            self, tensors: dict[str, torch.Tensor],
            pre_fusion_tens: dict[str, torch.Tensor] | None = None
            ):
        """Forward pass for VectorWiseBatchNorm
        Args:
            tensors: a dict of all the inputs that will be normalized,
            it is passed as a dict here but treated as a list
            pre_fusion_tens: is not also being used here, but may be passed
            as an arghument in certain cases

        """
        try:
            tensors_list = list(tensors.values())
        except AttributeError:
            tensors_list = tensors
        if self.activation:
            tensors_list = [self.act(tensors_list[i]) for i in range(len(tensors_list))]
        out = [self.vbn_modules[i](tensors_list[i]) for i in range(len(tensors_list))]
        out = torch.cat(out, dim=1)
        if self.affine:
            out = out * self.scale_factor + self.scale_bias
        return self.fusion_layer(out)

    def _get_vbn_modules(self):
        vbn_modules = []
        for feature_dim in self.in_features.values():
            module = VectorWiseBatchNorm(num_features=feature_dim)
            vbn_modules.append(module)
        return vbn_modules

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.scale_factor)
            nn.init.zeros_(self.scale_bias)
