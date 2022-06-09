def as_numpy(tensor):
    return tensor.cpu().detach().numpy()