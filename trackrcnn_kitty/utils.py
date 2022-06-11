import torch


def as_numpy(tensor):
    return tensor.cpu().detach().numpy()


def check_for_degenerate_boxes(targets):
    # For a valid box the height and width have to be positive numbers
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb = boxes[bb_idx].tolist()
                raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                )


def validate_and_build_stacked_boxes(targets, is_training=True):
    # the stacked boxes are needed for the association head
    stacked_boxes = []
    if is_training:
        assert targets is not None
        for target in targets:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
            else:
                raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

            stacked_boxes.extend(boxes)

    stacked_boxes = torch.stack(stacked_boxes)
    return stacked_boxes
