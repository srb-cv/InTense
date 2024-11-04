def test_multibench_data_mosei_senti():
    import pickle
    with open("/work/ML/varshneya/mosei/mosei_senti_data.pkl", 'rb') as f:
        data = pickle.load(f)
    assert list(data.keys()) == ["train", "test", "valid"]
    test_data = data['test']
    expected_keys = ["vision", "audio", "text", "labels", "id"]
    print(test_data.keys())
    assert list(test_data.keys()) == expected_keys
    assert test_data['vision'].shape == (4643, 50, 35)
    assert test_data['audio'].shape == (4643, 50, 74)
    assert test_data['text'].shape == (4643, 50, 300)


def test_multibench_dataloader_mosei():
    '''
    Testing with maxpad as TRUE
    all the sequences are padded to a length of 50
    '''
    from intensfusion.datasets.affect.get_data import get_dataloader
    _, _, testloader = get_dataloader(
            "data/mosei/mosei_senti_data.pkl",
            robust_test=False,
            data_type="mosi",
            save_path='./',
            batch_size=8,
            z_norm=False,
            label_norm=False,
            num_workers=0,
            max_pad=True,
            max_seq_len=50
        )
    batch_features = next(iter(testloader))
    assert len(batch_features) == 4
    assert batch_features[0].shape == (8, 50, 35)  # vision
    assert batch_features[1].shape == (8, 50, 74)  # audio
    assert batch_features[2].shape == (8, 50, 300)  # text
    assert batch_features[3].shape == (8, 1)  # labels


def test_multibench_dataloader_mosei_v2():
    '''
    Testing with maxpad as FALSE
    all the sequences are padded to a max length of the batch
    suitable for learnig with RNN based models
    '''
    import torch
    from intensfusion.datasets.affect.get_data import get_dataloader
    _, _, testloader = get_dataloader(
            "data/mosei/mosei_senti_data.pkl",
            robust_test=False,
            data_type="mosi",
            save_path='./',
            batch_size=8,
            z_norm=False,
            label_norm=False,
            num_workers=0,
        )
    features, feature_lengths, indices, labels = next(iter(testloader))
    num_modalities = 3  # vision, audio, text
    assert len(features) == num_modalities
    assert len(feature_lengths) == num_modalities
    max_length = int(torch.max(feature_lengths[0]))
    assert features[0].shape == (8, max_length, 35)  # vision
    assert features[1].shape == (8, max_length, 74)  # audio
    assert features[2].shape == (8, max_length, 300)  # text
    assert labels.shape == (8, 1)


def test_multibench_data_mosei():
    import pickle
    with open("data/mosei/mosei_raw.pkl", 'rb') as f:
        data = pickle.load(f)
    assert sorted(list(data.keys())) == ["test", "train", "valid"]
    test_data = data['test']
    expected_keys = ["vision", "audio", "text", "labels", "id"]
    print(test_data.keys())
    assert list(test_data.keys()) == expected_keys
    assert test_data['vision'].shape == (4662, 50, 713)
    assert test_data['audio'].shape == (4662, 50, 74)
    assert test_data['text'].shape == (4662, 50, 300)
