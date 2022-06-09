import torch

from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose

from finetune_pretrained_model import get_model_instance_segmentation
from trackrcnn_kitty.kitti_seg_track_dataset import KITTISegTrackDataset
from trackrcnn_kitty.resnet_101_temporally_extended import get_resnet101_backbone
from trackrcnn_kitty.track_rcnn_model import TrackRCNN
from references.detection.engine import train_one_epoch
from references.detection.utils import collate_fn

from copy import  deepcopy


def get_transform(train):
    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))

    return Compose(transforms)


if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has three classes - background, car and person
    num_classes = 3
    batch_size = 4

    # use our dataset and defined transformations
    dataset = KITTISegTrackDataset("D://Robert//KITTITrackSegDataset", get_transform(True))
    dataset_test = deepcopy(dataset)
    dataset_test.transforms = get_transform(False)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-1000])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-1000:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )

    # data_loader_test = torch.utils.data.DataLoader(
    #    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    #    collate_fn=collate_fn
    # )

    model = TrackRCNN(num_classes=num_classes, device=device)
    # model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it fo 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

    checkpoint = {
        "epoch": num_epochs,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }

    torch.save(checkpoint, "mask_rcnn_kitty.pth")

    print("That's it! Training is complete!")
