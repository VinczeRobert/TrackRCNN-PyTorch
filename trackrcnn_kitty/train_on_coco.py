import torch
from references.pytorch_detection.engine import train_one_epoch
from references.pytorch_detection.utils import collate_fn
from references.pytorch_maskrcnn_coco.coco import CocoConfig
from trackrcnn_kitty.creators.backbone_with_fpn_creator import BackboneWithFPNCreator
from trackrcnn_kitty.datasets.coco_dataset import CocoDataset, MainCOCODataset
from trackrcnn_kitty.models.mask_rcnn import CustomMaskRCNN

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    config = CocoConfig()

    dataset_train = CocoDataset()
    dataset_train.load_coco("D:\\Robert\\train2017", "train", year="2017")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CocoDataset()
    dataset_val.load_coco("D:\Robert\\train2017", "val", year="2017")
    dataset_val.prepare()

    train_set = MainCOCODataset(dataset_train, config, augment=True)
    train_generator = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_set = MainCOCODataset(dataset_val, config, augment=True)
    val_generator = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)

    backbone = BackboneWithFPNCreator(use_resnet_101=True,
                                      pretrain_backbone=False,
                                      trainable_backbone_layers=5).get_instance()

    maskrcnn_params = {
        "anchor_sizes": [8, 16, 32, 64, 128],
        "aspect_ratios": [0.5, 1.0, 2.0],
        "rpn_batch_size_per_image": 32
    }

    model = CustomMaskRCNN(num_classes=91,
                           backbone=backbone,
                           pretrain_only_backbone=False,
                           maskrcnn_params=maskrcnn_params,
                           fixed_size=(1024, 309))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.06, momentum=0, weight_decay=0)

    for epoch in range(40):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_generator, device, epoch, print_freq=10)

    checkpoint = {
        "epoch": 40,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }

    torch.save(checkpoint, "custom_coco.pth")

    print("Training complete.")
