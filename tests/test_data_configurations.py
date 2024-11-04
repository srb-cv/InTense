def test_mosi_configurations():
    import tomli
    with open('intensfusion/datasets/datasets.toml', mode='rb') as fp:
        config = tomli.load(fp)
        assert config["mosi"]["modalities"] == ["video", "audio", "text"]
