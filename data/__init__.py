import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from .datasets import AVLip


def get_bal_sampler(dataset):
    targets = dataset.targets

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights)
    )
    return sampler


def create_dataloader(opt):
    class_bal = getattr(opt, "class_bal", False)
    shuffle = not opt.serial_batches if (opt.isTrain and not class_bal) else False
    dataset = AVLip(opt)

    sampler = get_bal_sampler(dataset) if class_bal else None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(opt.num_threads),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(opt.num_threads) > 0,
    )
    return data_loader
