def test_multibench_data_mosi_senti():
    import pickle
    with open("/work/ML/varshneya/mosi/mosi_senti_data.pkl", 'rb') as f:
        data = pickle.load(f)
    assert list(data.keys()).sort() == ["train", "test", "valid"].sort()
    test_data = data['test']
    expected_keys = ["vision", "audio", "text", "labels", "id"].sort()
    print(test_data.keys())
    assert list(test_data.keys()).sort() == expected_keys
    assert test_data['vision'].shape == (686, 50, 20)
    assert test_data['audio'].shape == (686, 50, 5)
    assert test_data['text'].shape == (686, 50, 300)
