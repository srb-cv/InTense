def test_enrico_dataset():
    from intense.datasets.enrico.get_data import EnricoDataset
    dataset = EnricoDataset(
        data_dir='data/enrico',
        mode='test'
    )
    image, wireframe, label = dataset[0]
    assert image.shape==(3,256,128)
    assert wireframe.shape==(3,256,128)
    assert type(label)==int
    

def test_enrico_dataloader():
    from intense.datasets.enrico.get_data import get_dataloader
    data_loaders, class_weights = get_dataloader(
        data_dir='data/enrico',
        batch_size=8,
    ) 
    train_loader,val_loader,test_loader = data_loaders
    train_batch = next(iter(train_loader))
    images, wireframes, labels = train_batch
    assert images.shape==(8,3,256,128)
    assert wireframes.shape==(8,3,256,128)
    assert labels.shape==(8,)
    
    test_batch = next(iter(test_loader))
    test_images, test_wireframes, test_labels = test_batch
    assert test_images.shape==(8,3,256,128)
    assert test_wireframes.shape==(8,3,256,128)
    assert test_labels.shape==(8,)

    
        
    
    
    