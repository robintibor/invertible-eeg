import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import TensorDataset


def restrict_to_classes(loader, i_classes, remap_labels):
    subset = restrict_dataset_to_classes(loader.dataset, i_classes, remap_labels=remap_labels)
    assert loader.sampler.__class__.__name__ in ['RandomSampler', 'SequentialSampler']
    shuffle = loader.sampler.__class__.__name__ == 'RandomSampler'
    return th.utils.data.DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )


def restrict_dataset_to_classes(dataset, i_classes, remap_labels):
    found_key = None
    keys = ['train_labels', 'test_labels', 'targets', 'labels', 'tensors']
    for key in keys:
        if hasattr(dataset, key):
            found_key = key
            if key != 'tensors':
                labels = getattr(dataset, key)
            else:
                labels = dataset.tensors[1].argmax(dim=1).detach().cpu().numpy()
            indices = [np.flatnonzero(np.array(labels) == i_class) for i_class in i_classes]
            indices = np.sort(np.concatenate(indices))

    if found_key is None:
        assert hasattr(dataset, 'tensors')

    if remap_labels:
        resubtract = dict([(i_old_cls, i_old_cls - i_new_cls)
                           for i_new_cls, i_old_cls in
                           enumerate(i_classes)])
        labels = [l - resubtract[int(l)] if int(l) in resubtract else l for l in
                  labels]

    if found_key == 'tensor':
        new_labels = F.one_hot(
            th.tensor(labels), num_classes=dataset.tensors[1].shape[1]
        ).type_as(dataset.tensors[1])[indices].clone()
        subset = TensorDataset(dataset.tensors[0][indices].clone(),
                               new_labels)
    else:
        dataset.__dict__[found_key] = labels
        subset = th.utils.data.Subset(dataset, indices)
    return subset
