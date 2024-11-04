import torch.nn as nn
import torch


class Unimodal(nn.Module):
    """Implements MMDL classifier."""

    def __init__(self, encoders, fusion, head, index, has_padding=False):
        """Instantiate MMDL Module

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            fusion (nn.Module): Fusion module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
            p: The p norm applied while training the network
        """
        super(Unimodal, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.is_fusion_mkl = 'MKL' in type(self.fuse).__name__
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []
        self.modality_index = index

    def forward(self, inputs):
        """Implement Unimodal model with minmal changes
        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        i = self.modality_index
        outs = []
        if self.has_padding:
            outs.append(self.encoders[i](
                    [inputs[0][i], inputs[1][i]]))
        else:
            outs.append(self.encoders[i](inputs[i]))
        self.reps = outs
        if self.has_padding:
            if isinstance(outs[0], torch.Tensor):
                out = self.fuse(outs)
            else:
                out = self.fuse([i[0] for i in outs])
        else:
            out = self.fuse(outs)
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        if self.has_padding and not isinstance(outs[0], torch.Tensor):
            return self.head([out, inputs[1][0]])
        return self.head(out)
