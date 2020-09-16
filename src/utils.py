from torch.utils.tensorboard import SummaryWriter


def get_writer(path, category, current_time, params):
    log_dir = path + category
    for key, value in params.items():
        log_dir += f'_{key}{value}'
    log_dir += f'_{current_time}'
    return SummaryWriter(log_dir=log_dir)
