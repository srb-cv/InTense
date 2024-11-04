def test_unimodal_syn_model():
    from intensfusion.models.mm_container import Unimodal
    from intensfusion.models.common_models import MLP
    from intensfusion.models.encoders_syn import CharModelBatchNorm

    # load data
    from intensfusion.datasets.synthetic.get_data import (
        get_dataloader,
        MutltiModalCharDataset)
    data_csv_path = "data/synthetic_data/data_c2_m10_v5/data.csv"
    dataset = MutltiModalCharDataset(data_csv_path=data_csv_path)
    _, _, test_loader = get_dataloader(dataset=dataset, batch_size=32)
    features, labels = next(iter(test_loader))

    # define the depndency objects and model
    encoder = CharModelBatchNorm(output_dim=32)
    head = MLP(32, 64, 2)
    model = Unimodal(encoder=encoder, head=head, has_padding=False)

    idx = 4  # pick a modality index
    out = model(features[idx])
    print(out.shape)
    assert out.shape == (32, 2)
