import torch


def cat_fms(reference_fm, target_fm, max_disp, start_disp=0, dilation=1):
    """
    Based on reference feature, shift target feature within [start disp, end disp] to form the cost volume
    Details please refer to GC-Net, generate disparity [start disp, end disp]
    Args:
        max_disp, (int): under the scale of feature used, often equals to (end disp - start disp + 1), the max searching range of disparity
        start_disp (int): the start searching disparity index, usually be 0
        dilation (int): the step between near disparity index

    Inputs:
        reference_fm, (Tensor): reference feature, i.e. left image feature, in [BatchSize, Channel, Height, Width] layout
        target_fm, (Tensor): target feature, i.e. right image feature, in [BatchSize, Channel, Height, Width] layout

    Output:
        concat_fm, (Tensor): the formed cost volume, in [BatchSize, Channel*2, disp_sample_number, Height, Width] layout
    """
    end_disp = start_disp + max_disp - 1
    disp_sample_number = (max_disp+dilation-1)//dilation

    device = reference_fm.device
    N, C, H, W = reference_fm.shape
    concat_fm = torch.zeros(N, C * 2, disp_sample_number, H, W).to(device)

    # PSMNet cost-volume construction method
    idx = 0
    for i in range(start_disp, end_disp+1, dilation):
        if i > 0:
            concat_fm[:, :C, idx, :, i:] = reference_fm[:, :, :, i:]
            concat_fm[:, C:, idx, :, i:] = target_fm[:, :, :, :-i]
        elif i==0:
            concat_fm[:, :C, idx, :, :] = reference_fm
            concat_fm[:, C:, idx, :, :] = target_fm
        else:
            concat_fm[:, :C, idx, :, :i] = reference_fm[:, :, :, :i]
            concat_fm[:, C:, idx, :, :i] = target_fm[:, :, :, abs(i):]
        idx = idx + 1

    concat_fm = concat_fm.contiguous()
    return concat_fm
