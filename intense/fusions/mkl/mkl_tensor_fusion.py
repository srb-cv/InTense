import torch
import torch.nn as nn


class TensorFusionModel(nn.Module):
    def __init__(self, modality_indices: list[str],
                 input_dim: int = 16) -> None:
        super().__init__()
        self.modality_indices = modality_indices
        self.input_dim = input_dim
        # self.out_layer = nn.Linear(input_dim**2, out_dim)

    def forward(self, x: dict[str, torch.Tensor]):
        tens_list_x = [x[index] for index in self.modality_indices]
        t: torch.Tensor = self.compute_tensor_product(tens_list_x)
        t = torch.flatten(t, start_dim=1)
        # t = self.out_layer(t)
        return t

    def compute_tensor_product(self, inp: list[torch.Tensor]) -> torch.Tensor:
        if len(inp) == 2:
            return torch.einsum("bi,bj->bij", inp)
        elif len(inp) == 3:
            return torch.einsum("bi,bj,bk->bijk", inp)
        elif len(inp) == 4:
            return torch.einsum("bi,bj,bk,bl->bijkl", inp)
        else:
            raise ValueError('Tensor product is only supported for 2 to 4\
                 batch vectors.')


class PreTensorFusionBN(nn.Module):
    def __init__(self, input_dim, out_dim) -> None:
        super().__init__()
        self.linear_map = nn.Linear(input_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        # self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.linear_map(x)
        # x = self.act(x)
        x = self.bn(x)
        return x
