import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import PsmBb
from .cost_computation import cat_fms
from .cost_aggregation import AcfCost
from .conf_measure import ConfidenceEstimation
from .disp_prediction import faster_soft_argmin
from .loss import StereoFocalLoss, ConfidenceNllLoss, DispSmoothL1Loss
from dmb.data.datasets.evaluation.stereo.eval import remove_padding, do_evaluation, do_occlusion_evaluation

class AcfNet(nn.Module):
    """
    Args:
        cfg, (dict): the configuration of model
    Inputs:
        batch, (dict): the input batch of the model,
                keywords must contains 'leftImage', 'rightImage', 'original size'
                optionally contains 'leftDisp', 'rightDisp'
                Image or disparity map are in torch.Tensor type
                original size is the size of original unprocessed image, i.e. (H,W), e.g. (540, 960) in SceneFlow
    Outputs:
        result, (dict): the result grouped in a dict.
    """

    def __init__(self, cfg):
        super(AcfNet, self).__init__()
        self.cfg = cfg.copy()

        self.batchNorm = cfg.model.batchNorm

        self.max_disp = cfg.model.max_disp

        # image feature extraction
        self.backbone_in_planes = cfg.model.backbone.in_planes
        self.backbone = PsmBb(in_planes=self.backbone_in_planes, batchNorm=self.batchNorm)

        # matching cost computation
        self.cost_computation = cat_fms

        # matching cost aggregation
        self.cost_aggregation_in_planes = cfg.model.cost_aggregation.in_planes
        self.cost_aggregation = AcfCost(max_disp=self.max_disp,
                                        in_planes=self.cost_aggregation_in_planes,
                                        batchNorm=self.batchNorm)

        # confidence learning
        self.conf_est_net = nn.ModuleList([
            ConfidenceEstimation(in_planes=self.max_disp, batchNorm=self.batchNorm) for i in range(3)])
        self.confidence_coefficient = cfg.model.confidence.coefficient
        self.confidence_init_value = cfg.model.confidence.init_value

        # calculate loss
        self.weights = cfg.model.loss.weights
        self.sparse = cfg.data.sparse

        # focal loss
        self.focal_coefficient = cfg.model.loss.focal_coefficient
        self.loss_variance = cfg.model.loss.variance
        self.focal_loss_evaluator = \
            StereoFocalLoss(max_disp=self.max_disp, weights=self.weights,
                            focal_coefficient=self.focal_coefficient, sparse=self.sparse)

        # smooth l1 loss
        self.l1_loss_evaluator = DispSmoothL1Loss(self.max_disp, weights=self.weights,
                                                  sparse=self.sparse)
        self.l1_loss_weight = cfg.model.loss.l1_loss_weight

        # nll loss
        self.conf_loss_evaluator = \
            ConfidenceNllLoss(max_disp=self.max_disp, weights=self.weights, sparse=self.sparse)
        self.conf_loss_weight = cfg.model.loss.conf_loss_weight

        # disparity regression
        # Attention: faster soft argmin contains a nn.Conv3d with fixed value
        # and cannot be initialized with other initialization method, e.g. Xavier, Kaiming initialization
        self.sa_temperature = cfg.model.disparity_prediction.sa_temperature
        self.disp_predictor = faster_soft_argmin(self.max_disp)


    def forward(self, batch):
        ref_image, target_image = batch['leftImage'], batch['rightImage']
        target = batch['leftDisp'] if 'leftDisp' in batch else None

        ref_fm, target_fm = self.backbone(ref_image, target_image)

        raw_cost = self.cost_computation(ref_fm, target_fm, int(self.max_disp // 4))
        costs = self.cost_aggregation(raw_cost)

        confidence_costs = [cen(c) for c, cen in zip(costs, self.conf_est_net)]
        confidences = [torch.sigmoid(c) for c in confidence_costs]

        variances = [self.confidence_coefficient * (1 - conf) + self.confidence_init_value for conf in confidences]

        disps = [self.disp_predictor(cost, temperature=self.sa_temperature) for cost in costs]


        if self.training:
            assert target is not None, "Ground truth disparity map should be given"
            losses = {}
            focal_losses = self.focal_loss_evaluator(costs, target, variances)
            losses.update(focal_losses)

            l1_losses = self.l1_loss_evaluator(disps, target)
            l1_losses = {k: v * self.l1_loss_weight for k, v in zip(l1_losses.keys(), l1_losses.values())}
            losses.update(l1_losses)

            nll_losses = self.conf_loss_evaluator(confidence_costs, target)
            nll_losses = {k : v * self.conf_loss_weight for k, v in zip(nll_losses.keys(), nll_losses.values())}
            losses.update(nll_losses)

            return losses
        else:
            confidences = remove_padding(confidences, batch['original_size'])
            disps = remove_padding(disps, batch['original_size'])

            error_dict = {}
            if target is not None:
                target = remove_padding(target, batch['original_size'])
                error_dict = do_evaluation(disps[0], target,
                                           self.cfg.model.eval.lower_bound,
                                           self.cfg.model.eval.upper_bound)

            if self.cfg.model.eval.eval_occlusion and 'leftDisp' in batch and 'rightDisp' in batch:
                batch['leftDisp'] = remove_padding(batch['leftDisp'], batch['original_size'])
                batch['rightDisp'] = remove_padding(batch['rightDisp'], batch['original_size'])
                occ_error_dict = do_occlusion_evaluation(disps[0], batch['leftDisp'], batch['rightDisp'],
                                        self.cfg.model.eval.lower_bound,
                                        self.cfg.model.eval.upper_bound)
                error_dict.update(occ_error_dict)


            result = {'Disparity': disps,
                      'GroundTruth': target,
                      'Confidence': confidences,
                      'Error': error_dict,
                      }

            if self.cfg.model.eval.is_cost_return:
                if self.cfg.model.eval.is_cost_to_cpu:
                    costs = [cost.cpu() for cost in costs]
                result['Cost'] = costs

            return result
