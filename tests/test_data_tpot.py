def test_tpot_data():
    import pickle
    import torch
    path = "/data/varshneya/datasets/tpot/PANAM_constructs_test_0.pickle"
    tensor_data = torch.load(path)
    print(tensor_data)