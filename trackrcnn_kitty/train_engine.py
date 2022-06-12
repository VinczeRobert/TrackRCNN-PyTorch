import torch
from torch.hub import load_state_dict_from_url
from torchvision.models.detection._utils import overwrite_eps

from references.detection.engine import train_one_epoch, evaluate
from references.detection.utils import collate_fn
from trackrcnn_kitty.creators.backbone_with_fpn_creator import BackboneWithFPNCreator
from trackrcnn_kitty.datasets.dataset_factory import get_dataset
from trackrcnn_kitty.json_config import JSONConfig
from trackrcnn_kitty.models.track_rcnn_model import TrackRCNN
from trackrcnn_kitty.datasets.transforms import get_transforms


model_urls = {
    "maskrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
}


class TrainEngine:
    def __init__(self, config_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.cuda.empty_cache()
        self.config = JSONConfig.get_instance(config_path)

        transforms = get_transforms(self.config.transforms_list)
        self.dataset = get_dataset(self.config.dataset, self.config.dataset_path, transforms)
        # self.dataset_test = get_dataset(self.config.dataset, self.config.dataset_path, transforms)
        #
        # indices = torch.randperm(len(self.dataset)).tolist()
        # self.dataset = torch.utils.data.Subset(self.dataset, indices[:-50])
        # self.dataset.num_classes = 2
        # self.dataset_test = torch.utils.data.Subset(self.dataset_test, indices[-50:])
        #
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=4,
            collate_fn=collate_fn
        )
        #
        # self.data_loader_test = torch.utils.data.DataLoader(
        #     self.dataset_test,
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=4,
        #     collate_fn=collate_fn
        # )

        backbone = BackboneWithFPNCreator(train_last_layer=self.config.train_last_layer,
                                          use_resnet_101=self.config.use_resnet_101).get_instance()
        self.model = TrackRCNN(num_classes=self.dataset.num_classes,
                               backbone=backbone,
                               do_tracking=self.config.add_associations,
                               batch_size=self.config.batch_size)
        self.model.to(self.device)

    def run_training(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay,
                                    momentum=self.config.momentum)

        if self.config.add_lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                           step_size=3,
                                                           gamma=0.1)

        for epoch in range(self.config.num_epochs):
            # train for one epoch, printing every 10 iterations
            # train_one_epoch(self.model, optimizer, self.data_loader, self.device, epoch, print_freq=10)
            evaluate(self.model, self.data_loader, device=self.device)

            if self.config.add_associations:
                lr_scheduler.step()

        checkpoint = {
            "epoch": self.config.num_epochs,
            "model_state": self.model.state_dict(),
            "optim_state": optimizer.state_dict()
        }

        torch.save(checkpoint, self.config.weights_path)

        print("Training complete.")
