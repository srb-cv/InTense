import torch.nn as nn
import torch
import logging 


class MMDL(nn.Module):
    """Implements MMDL classifier."""

    def __init__(self, encoders, fusion, head, has_padding=False):
        """Instantiate MMDL Module

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            fusion (nn.Module): Fusion module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
            p: The p norm applied while training the network
        """
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.is_fusion_mkl = "MKL" in type(self.fuse).__name__
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def forward(self, inputs):
        """Apply MMDL to Layer Input.

        Args:
            inputs : Layer Input. list of tensor or a list of list of tensor

        Returns:
            torch.Tensor: Layer Output
        """
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i]([inputs[0][i], inputs[1][i]]))
        else:
            for i in range(len(inputs)):
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

    def modality_scores(self, p) -> dict[str, float]:
        return self.fuse.scores(p)


class MMTF(nn.Module):
    def __init__(
        self,
        encoders,
        tf_encoders,
        pre_tf_encoders,
        tf_modality_indices,
        fusion,
        head,
        has_padding=False
    ):
        """Instantiate MMTF Module

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            tf_encoders: List of nn.Module encoders for the fusion modalities
            fusion (nn.Module): Fusion module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
            p: The p norm applied while training the network
        """
        super(MMTF, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.tf_encoders = nn.ModuleDict(tf_encoders)
        self.pre_tf_encoders = nn.ModuleDict(pre_tf_encoders)
        self.fuse = fusion
        self.is_fusion_mkl = "MKL" in type(self.fuse).__name__
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []
        self.tf_modality_indices = tf_modality_indices
        self.involved_modality_indices = self._get_involved_modalities()

    def _get_involved_modalities(self):
        all_modalities = "".join(self.tf_modality_indices)
        return sorted(set(all_modalities))

    def forward(self, inputs):  # sourcery skip: aug-assign
        """Apply MMTF to Layer Input.

        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        outs = []
        outs_tf = []
        if self.has_padding:
            outs.extend(
                self.encoders[i]([inputs[0][i], inputs[1][i]])
                for i in range(len(inputs[0]))
            )
        else:
            outs.extend(self.encoders[i](inputs[i]) for i in range(len(inputs)))
        self.reps = outs
        # TODO: change the keys to orginal modailites instead of indices
        outs_dict = {str(i+1): v for i, v in enumerate(outs)}
        # outs_pre_tf = {mod_index: self.pre_tf_encoders[mod_index](outs[int(mod_index)-1].detach())
        #                 for mod_index in self.involved_modality_indices}
        outs_pre_tf = {
            mod_index: self.pre_tf_encoders[mod_index](outs[int(mod_index) - 1])
            for mod_index in self.involved_modality_indices
        }
        outs_tf = {
                indices: self.tf_encoders[indices](outs_pre_tf) for indices in self.tf_modality_indices
        }
        if (
            self.has_padding
            and isinstance(outs[0], torch.Tensor)
            or not self.has_padding
        ):
            outs = dict(outs_dict, **outs_tf)
            # outs = [outs_pre_tf['1']]+[outs_pre_tf['2']]+[outs_pre_tf['3']] + outs_tf
            out = self.fuse(outs, outs_pre_tf)
        else:
            logging.warning("WARNING:: Probably an unwanted condtion")
            out = self.fuse([i[0] for i in outs])
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        if type(out) is not dict and self.has_padding and not isinstance(out[0], torch.Tensor):
            return self.head([out, inputs[1][0]])
        return self.head(out)

    def modality_scores(self, p) -> dict[str, float]:
        return self.fuse.scores(p)


class Unimodal(nn.Module):
    """Implements Unimodal classifier."""

    def __init__(self, encoder, head, has_padding=False):
        """Instantiate Unimodal Module

        Args:
            encoder (nn.Module): Encoder module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not.
                                          Defaults to False.
        """
        super(Unimodal, self).__init__()
        self.encoder: nn.Module = encoder
        self.head: nn.Module = head
        self.has_padding: bool = has_padding
        self.reps = []

    def forward(self, input):
        """Apply forward pass to a modality.

        Args:
            input : Layer Input. a batch on tensors

        Returns:
            torch.Tensor: Layer Output
        """
        outs = []
        if self.has_padding:
            raise NotImplementedError("forward for has_padding True is yet to"
                                      "be defined")
        else:
            out = self.encoder(input)
        self.reps = outs
        return self.head(out)


class MMRO(nn.Module):
    def __init__(
        self,
        encoders,
        tf_encoders,
        pre_tf_encoders,
        tf_modality_indices,
        fusion,
        head,
        has_padding=False
    ):
        """Instantiate parallel MRO Module. Should return an array of outputs,
        corresponding to uni, bi, tri up to n-modal interaction

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            tf_encoders: Dict of nn.Module encoders for the fusion modalities
            pre_tf_encoders: Dict of nn.modules to reduce the latent dimension
            fusion (nn.Module): Fusion module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
        """
        super(MMTF, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.tf_encoders = nn.ModuleDict(tf_encoders)
        self.pre_tf_encoders = nn.ModuleDict(pre_tf_encoders)
        self.is_fusion_mkl = "MKL" in type(self.fuse).__name__
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []
        self.tf_modality_indices = tf_modality_indices
        self.involved_modality_indices = self._get_involved_modalities()

    def _get_involved_modalities(self):
        all_modalities = "".join(self.tf_modality_indices)
        return sorted(set(all_modalities))

    def forward(self, inputs):  # sourcery skip: aug-assign
        """Apply MRO to Layer Input.

        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        outs = []
        outs_tf = []
        if self.has_padding:
            outs.extend(
                self.encoders[i]([inputs[0][i], inputs[1][i]])
                for i in range(len(inputs[0]))
            )
        else:
            outs.extend(self.encoders[i](inputs[i]) for i in range(len(inputs)))
        self.reps = outs
        # TODO: change the keys to orginal modailites instead of indices
        outs_dict = {str(i+1): v for i, v in enumerate(outs)} # here we get unimodal outs
        outs_pre_tf = {
            mod_index: self.pre_tf_encoders[mod_index](outs[int(mod_index) - 1].detach())
            for mod_index in self.involved_modality_indices
        }
        # outs_pre_tf = {
        #     mod_index: self.pre_tf_encoders[mod_index](outs[int(mod_index) - 1])
        #     for mod_index in self.involved_modality_indices
        # }
        outs_tf = {
                indices: self.tf_encoders[indices](outs_pre_tf) for indices in self.tf_modality_indices
        } # here we get the bimodal and trimodal outputs
        if (
            self.has_padding
            and isinstance(outs[0], torch.Tensor)
            or not self.has_padding
        ):
            outs = dict(outs_dict, **outs_tf)
            outs_dict = self.fuse(outs)
        else:
            logging.warning("WARNING:: Probably an unwanted condtion")
            exit(0)
        return outs_dict
        
        