def test_avmnist_data():
    import numpy as np
    data_dir="data/avmnist"
    train_data = [np.load(data_dir+"/image/train_data.npy"),
                  np.load(data_dir +"/audio/train_data.npy"),
                  np.load(data_dir+"/train_labels.npy")]
    assert len(train_data) == 3
    assert train_data[0].shape == (60000, 784)
    assert train_data[1].shape == (60000, 112, 112)
    assert train_data[2].shape == (60000,)
    
def test_avmnist_dataloader():
    import torch
    from intense.datasets.avmnist.get_data import get_dataloader
    _, _, testloader = get_dataloader(
        data_dir="data/avmnist",
        batch_size=8,
        num_workers=0,
    )
    imgs, audio, labels = next(iter(testloader))
    assert imgs.shape == (8, 1, 28, 28)
    assert audio.shape == (8, 1, 112, 112)
    assert labels.shape == (8,)
    

def test_encoders():
    import torch
    from intense.datasets.avmnist.get_data import get_dataloader
    from intense.models.common_models import LeNet
    _, _, testloader = get_dataloader(
        data_dir="data/avmnist",
        batch_size=8,
        num_workers=0,
    )
    imgs, audio, labels = next(iter(testloader))
    channels = 6
    encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
    out_img = encoders[0](imgs.float().cuda())
    out_audio = encoders[1](audio.float().cuda())
    assert out_img.shape == (8,48)
    assert out_audio.shape == (8,192)



if __name__=="__main__":
    test_avmnist_data()