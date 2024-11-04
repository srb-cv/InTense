from intense.fusions.mkl.mkl_tensor_fusion import (TensorFusionModel,
                                                        PreTensorFusionBN)


def _build_tensor_fusion_models(tf_modalities: list[str], input_dim):
    submodels_tensor_fusion = {}
    for modality_indices in tf_modalities: 
        modality_indices_list = list(modality_indices)
        model = TensorFusionModel(modality_indices=modality_indices_list,
                                  input_dim=input_dim)
        submodels_tensor_fusion[modality_indices] = model
    return submodels_tensor_fusion


def _build_pre_tf_models(tf_modality_indices: list[str], latent_dim_dict: dict,
                         out_dim: int):
    pre_tf_model = {}
    all_modalities = "".join(tf_modality_indices)
    involved_modality_indices = sorted(set(all_modalities))
    for modality in involved_modality_indices:
        model = PreTensorFusionBN(input_dim=latent_dim_dict[modality],
                                  out_dim=out_dim)
        pre_tf_model[modality] = model
    return pre_tf_model
