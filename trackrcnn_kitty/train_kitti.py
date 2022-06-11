import torch

from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose

from trackrcnn_kitty.kitti_seg_track_dataset import KITTISegTrackDataset
from trackrcnn_kitty.track_rcnn_model import TrackRCNN
from references.detection.engine import train_one_epoch
from references.detection.utils import collate_fn


def get_transform(train, do_h_flip=False):
    transforms = [ToTensor()]
    if train and do_h_flip:
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

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )

    model = TrackRCNN(num_classes=num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0000005)

    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.1)

    # let's train it for epochs as in the paper
    num_epochs = 40

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        # lr_scheduler.step()

    checkpoint = {
        "epoch": num_epochs,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }

    torch.save(checkpoint, "mask_trackrcnn_kitty.pth")

    print("That's it! Training is complete!")
