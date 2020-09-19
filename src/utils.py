from torch.utils.tensorboard import SummaryWriter


def get_writer(path, stem, current_time, params):
    log_dir = path + stem
    for key, value in params.items():
        log_dir += f'_{key}{value}'
    log_dir += f'_{current_time}'
    return SummaryWriter(log_dir=log_dir)


def get_model_path(path, stem, current_time, params, f1=0.0):
    path += stem
    if f1 is not 0.0:
        path += '_{:1.5f}'.format(f1)
    for key, value in params.items():
        path += f'_{key}{value}'
    path += f'_{current_time}.pt'
    return path
