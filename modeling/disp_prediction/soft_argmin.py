import torch
import torch.nn.functional as F


def soft_argmin(cost_volume, max_disp, start_disp=0, dilation=1, normalize=True, temperature=1.0):
    # type: (torch.Tensor, int, int, int, bool, [float, int]) -> torch.Tensor
    r"""Implementation of soft argmin proposed by GC-Net.
    Args:
        max_disp, (int): under the scale of feature used, often equals to (end disp - start disp + 1), the max searching range of disparity
        start_disp (int): the start searching disparity index, usually be 0
        dilation (int): the step between near disparity index
        normalize (bool): whether apply softmax on cost_volume, default True
        temperature (float, int): a temperature factor will times with cost_volume
                    details can refer to: https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/

    Inputs:
        cost_volume (Tensor): the matching cost after regularization, in [B, disp_sample_number, W, H] layout

    Returns:
        disp_map (Tensor): a disparity map regressed from cost volume, in [B, 1, W, H] layout
    """
    if cost_volume.dim() != 4:
        raise ValueError(
            'expected 4D input (got {}D input)'.format(cost_volume.dim())
        )
    end_disp = start_disp + max_disp - 1
    disp_sample_number = (max_disp + dilation - 1) // dilation

    # grab cost volume shape
    N, D, H, W = cost_volume.shape

    assert disp_sample_number == D, \
        "The number of disparity samples should be same with the size of cost volume Channel dimension!"

    # generate disparity indexes
    disp_index = torch.linspace(start_disp, end_disp, disp_sample_number).to(cost_volume.device)
    disp_index = disp_index.repeat(N, H, W, 1).permute(0, 3, 1, 2).contiguous()

    # compute probability volume
    # prob_volume: (BatchSize, disp_sample_number, Height, Width)
    cost_volume = cost_volume * temperature
    if normalize:
        prob_volume = F.softmax(cost_volume, dim=1)
    else:
        prob_volume = cost_volume

    # compute disparity: (BatchSize, 1, Height, Width)
    disp_map = torch.sum(prob_volume * disp_index, dim=1, keepdim=True)

    return disp_map
