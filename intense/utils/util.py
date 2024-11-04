import os
import uuid


def get_save_path(mode='train', version=None, subdir=None):
    from datetime import date
    week = date.today().isocalendar()[1]
    weekday = date.today().isocalendar()[2]
    exp_dir = f'week_{week}_{weekday}'
    save_path = f'logs/{exp_dir}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    curr_version = int(str(uuid.uuid4().fields[-1])[:7])
    if mode == 'train' and version is None:
        curr_version += 1
    if subdir is not None:
        return os.path.join(exp_dir,
                            f'v_{curr_version}_{os.uname().nodename}',
                            subdir)
    save_dir_path = os.path.join(exp_dir,
                                 f'v_{curr_version}_{os.uname().nodename}')
    return save_dir_path


def get_split_len(split_fraction: int | float, len_dataset: int):
    if isinstance(split_fraction, int):
        return split_fraction
    elif isinstance(split_fraction, float):
        return int(split_fraction * len_dataset)
    else:
        raise ValueError(f"Unsupported type {type(split_fraction)}"
                         "for split fraction.")


def get_split_lengths(split_fractions: list[int | float], len_dataset: int):
    split_lengths = [get_split_len(fraction, len_dataset)
                     for fraction in split_fractions]
    splits_sum = sum(split_lengths)
    if splits_sum > len_dataset:
        raise ValueError(f"Dataset length is smaller than sum of "
                         f"specified splits ({split_lengths}): {len_dataset} < {splits_sum}.")
    split_lengths.insert(0, len_dataset - splits_sum)
    return split_lengths
