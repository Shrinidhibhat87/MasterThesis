import os
from typing import List, Union

import torch


def list_sum(x: list) -> any:
    if len(x):
        return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])
    else:
        return 0


def list_mean(x: list) -> any:
    return list_sum(x) / len(x)


def get_dist_size() -> int:
    return int(os.environ['WORLD_SIZE'])


def sync_tensor(
    tensor: Union[torch.Tensor, float], reduce='mean'
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.Tensor(1).fill_(tensor).cuda()
    tensor_list = [torch.empty_like(tensor) for _ in range(get_dist_size())]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == 'mean':
        return list_mean(tensor_list)
    elif reduce == 'sum':
        return list_sum(tensor_list)
    elif reduce == 'cat':
        return torch.cat(tensor_list, dim=0)
    elif reduce == 'root':
        return tensor_list[0]
    else:
        return tensor_list


class AverageMeterRelative:
    def __init__(self, is_distributed=False):
        """
        for loss and reset_bn
        """
        self.is_distributed = is_distributed
        self.sum = 0
        self.count = 0

    def _sync(self, val: Union[torch.Tensor, int, float]) -> Union[torch.Tensor, int, float]:
        return sync_tensor(val, reduce='sum') if self.is_distributed else val

    def update(self, val: Union[torch.Tensor, int, float], delta_n=1):
        self.count += self._sync(delta_n)
        self.sum += self._sync(val * delta_n)

    def get_count(self) -> Union[torch.Tensor, int, float]:
        return (
            self.count.item()
            if isinstance(self.count, torch.Tensor) and self.count.numel() == 1
            else self.count
        )

    @property
    def avg(self):
        avg = -1 if self.count == 0 else self.sum / self.count
        return avg.item() if isinstance(avg, torch.Tensor) and avg.numel() == 1 else avg


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, is_distributed=False):
        self.is_distributed = is_distributed
        self.sum = 0
        self.aux_sum = 0
        self.count = 0

    def _sync(self, val: Union[torch.Tensor, int, float]) -> Union[torch.Tensor, int, float]:
        return sync_tensor(val, reduce='sum') if self.is_distributed else val

    def update(self, val: Union[torch.Tensor, int, float], delta_n=1):
        self.sum += self._sync(val[0])
        self.aux_sum += self._sync(val[1])

    def get_count(self) -> Union[torch.Tensor, int, float]:
        return (
            self.count.item()
            if isinstance(self.count, torch.Tensor) and self.count.numel() == 1
            else self.count
        )

    @property
    def avg(self):
        avg = (self.sum / self.aux_sum).mean()
        return avg.item() if isinstance(avg, torch.Tensor) and avg.numel() == 1 else avg

    @property
    def individual_miou(self):
        ind_mIoU = self.sum / self.aux_sum
        return ind_mIoU if isinstance(ind_mIoU, torch.Tensor) else None
