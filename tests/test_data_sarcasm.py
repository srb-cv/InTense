def test_sarcasm_data():
    import pickle
    with open("/data/varshneya/datasets/sarcasm.pkl", 'rb') as f:
        data = pickle.load(f)
    print("The Keys of the data are:", data.keys())
    assert sorted(list(data.keys())) == ["test", "train", "valid"]
    test_data = data['test']
    print("The Keys for the test data are: ", test_data.keys())
    expected_keys = sorted(["vision", "audio", "text", "labels", "id"])
    assert sorted(test_data.keys()) == expected_keys
    assert test_data['audio'].shape == (138, 50, 81)
    assert test_data['vision'].shape == (138, 50, 371)
    assert test_data['text'].shape == (138, 50, 300)


def test_sarcasm_dataloader_v1():
    '''
    Testing with maxpad as true
    all the sequences are padded to a length of 50
    '''
    from intense.datasets.affect.get_data import get_dataloader
    _, _, testloader = get_dataloader(
            "/data/varshneya/datasets/sarcasm.pkl",
            robust_test=False,
            data_type="sarcasm",
            save_path='./',
            batch_size=8,
            z_norm=False,
            label_norm=False,
            num_workers=0,
            max_pad=True,
            max_seq_len=50,
            task="classification"
        )
    batch_features = next(iter(testloader))
    assert len(batch_features) == 4
    assert batch_features[0].shape == (8, 50, 371)  # vision
    assert batch_features[1].shape == (8, 50, 81)  # audio
    assert batch_features[2].shape == (8, 50, 300)  # text
    labels = batch_features[3]
    assert labels.shape == (8, 1)  # labels
    assert set(labels.numpy().squeeze()) == {0, 1}


def test_sarcasm_dataloader_mosei_v2():
    '''
    Testing with maxpad as FALSE
    all the sequences are padded to a max length of the batch
    suitable for learnig with RNN based models
    '''
    import torch
    from intense.datasets.affect.get_data import get_dataloader
    _, _, testloader = get_dataloader(
            "/data/varshneya/datasets/sarcasm.pkl",
            robust_test=False,
            data_type="sarcasm",
            save_path='./',
            batch_size=8,
            z_norm=False,
            label_norm=False,
            num_workers=0,
            task="classification"
        )
    features, feature_lengths, indices, labels = next(iter(testloader))
    num_modalities = 3  # vision, audio, text
    assert len(features) == num_modalities
    assert len(feature_lengths) == num_modalities
    max_length = int(torch.max(feature_lengths[0]))
    assert features[0].shape == (8, max_length, 371)  # vision
    assert features[1].shape == (8, max_length, 81)  # audio
    assert features[2].shape == (8, max_length, 300)  # text
    assert labels.shape == (8, 1)
    assert set(labels.numpy().squeeze()) == {0, 1}


def test_labels_v1():
    from intense.datasets.affect.get_data import get_dataloader
    _, _, testloader = get_dataloader(
            "/data/varshneya/datasets/sarcasm.pkl",
            robust_test=False,
            data_type="sarcasm",
            save_path='./',
            batch_size=32,
            z_norm=False,
            label_norm=False,
            num_workers=0,
            max_pad=True,
            max_seq_len=50
        )
    batch_features = next(iter(testloader))
    labels = batch_features[3].numpy().squeeze()
    assert set(labels) == {-1.0, 1.0}
