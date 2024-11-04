import pandas as pd


def test_synthetic_multiclass_data():
    data_csv_path = "data/synthetic_data/data_c2_m10_v3/data.csv"
    df = pd.read_csv(data_csv_path,  index_col=0)
    print("The Keys of the data are:", df.columns)
    assert len(df.columns) == 11
    assert 'label' in df.columns
    assert len(df.iloc[0].loc['mod_1_0.30']) == 100
    assert set(df['label'].iloc[1:10]) == {-1, 1}


def test_synthetic_xor_data():
    data_csv_path = "data/synthetic_data/data_xor_c2_m3_v3/xor_data.csv"
    df = pd.read_csv(data_csv_path, index_col=0)
    print("The Keys of the data are:", df.columns)
    assert len(df.columns) == 4
    assert 'label' in df.columns
    assert set(df['label'].iloc[1:10]) == {0, 1}
    assert len(df['mod_1'].iloc[0]) == 100


def test_synthetic_multiclass_loader():
    import torch
    from intensfusion.datasets.synthetic.get_data import (
        get_dataloader,
        MutltiModalCharDataset)
    data_csv_path = "data/synthetic_data/data_c2_m10_v5/data.csv"
    dataset = MutltiModalCharDataset(data_csv_path=data_csv_path)
    _, _, test_loader = get_dataloader(dataset=dataset, batch_size=32)
    features, labels = next(iter(test_loader))
    assert len(features) == 10  # num_modalities
    assert torch.stack(features).shape == (10, 32, 4, 100)
    assert labels.shape == (32,)
    assert set(labels.numpy()) == {0, 1}


def test_synthetic_multiclass_unimodal_loader():
    """
    for the scenario where a unimodal model has to
    be trained on a single modality of the data, which
    is multimodal 
    """
    from intensfusion.datasets.synthetic.get_data import (
        get_dataloader,
        MutltiModalCharDataset)
    data_csv_path = "data/synthetic_data/data_c2_m10_v5/data.csv"
    dataset = MutltiModalCharDataset(data_csv_path=data_csv_path)
    _, _, test_loader = get_dataloader(dataset=dataset, batch_size=32)
    features, labels = next(iter(test_loader))
    assert len(features) == 10  #num_modalities
    # pick a modality index from {,..,9}
    idx = 4
    assert features[idx].shape == (32, 4, 100)
    assert labels.shape == (32,)
    assert set(labels.numpy()) == {0, 1}


def test_synthetic_xor_dataloader():
    import torch
    from intensfusion.datasets.synthetic.get_data import (
        get_dataloader,
        MutltiModalCharDataset)
    data_csv_path = "data/synthetic_data/data_xor_c2_m3_v3/xor_data.csv"
    dataset = MutltiModalCharDataset(data_csv_path=data_csv_path)
    _, _, test_loader = get_dataloader(dataset=dataset, batch_size=32)
    features, labels = next(iter(test_loader))
    assert len(features) == 3  # num_modalities
    assert torch.stack(features).shape == (3, 32, 4, 100)
    assert labels.shape == (32,)
    assert set(labels.numpy()) == {0, 1}