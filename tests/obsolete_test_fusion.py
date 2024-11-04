import torch


@torch.no_grad()
def test_mkl_regularizer_one_output():
    from intensfusion.fusions.mkl.mkl_fusion import MKLFusion
    import torch
    fusion = MKLFusion(in_features=[1, 2, 2],
                       out_features=1)
    assert fusion.fusion_layer.weight.shape == (1, 5)
    custom_weights = torch.Tensor([1, 2, 3, 1, 2]).unsqueeze(0)
    assert custom_weights.shape == (1, 5)
    fusion.fusion_layer.weight.copy_(custom_weights)
    modality_norms = [torch.linalg.matrix_norm(torch.Tensor([1]).unsqueeze(0)),
                      torch.linalg.matrix_norm(torch.Tensor([2, 3]).unsqueeze(0)),
                      torch.linalg.matrix_norm(torch.Tensor([1, 2]).unsqueeze(0))]
    regularizer = torch.sum(torch.Tensor(modality_norms)) ** 2
    print(regularizer)
    print(fusion.regularizer())
    assert regularizer == fusion.regularizer()


@torch.no_grad()
def test_mkl_scores_one_output():
    from intensfusion.fusions.mkl.mkl_fusion import MKLFusion
    import torch
    fusion = MKLFusion(in_features=[1, 2, 2],
                       out_features=1)
    custom_weights = torch.Tensor([1, 2, 3, 1, 2]).unsqueeze(0)
    fusion.fusion_layer.weight.copy_(custom_weights)
    modality_norms = [torch.linalg.matrix_norm(torch.Tensor([1]).unsqueeze(0)),
                      torch.linalg.matrix_norm(torch.Tensor([2, 3]).unsqueeze(0)),
                      torch.linalg.matrix_norm(torch.Tensor([1, 2]).unsqueeze(0))]
    sum_norms = torch.sum(torch.Tensor(modality_norms))
    scores = torch.Tensor([norm_ / sum_norms for norm_ in modality_norms])
    print(scores)
    print(fusion.scores())
    assert torch.equal(scores, fusion.scores())


@torch.no_grad()
def test_mkl_scores_multiple_outputs():
    from intensfusion.fusions.mkl.mkl_fusion import MKLFusion
    import torch
    fusion = MKLFusion(in_features=[1, 2, 2],
                       out_features=2)
    custom_weights = torch.Tensor([[1, 2, 3, 1, 2],
                                   [2, 1, 1, 4, 1]])
    fusion.fusion_layer.weight.copy_(custom_weights)
    modality_norms = [torch.linalg.matrix_norm(torch.Tensor([1, 2]).unsqueeze(0)),
                      torch.linalg.matrix_norm(torch.Tensor([2, 3, 1, 1]).unsqueeze(0)),
                      torch.linalg.matrix_norm(torch.Tensor([1, 2, 4, 1]).unsqueeze(0))]
    sum_norms = torch.sum(torch.Tensor(modality_norms))
    scores = torch.Tensor([norm_ / sum_norms for norm_ in modality_norms])
    print(scores)
    print(fusion.scores())
    assert torch.equal(scores, fusion.scores())





    
 
