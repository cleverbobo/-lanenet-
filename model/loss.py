from torch.nn.modules.loss import _Loss
import torch
from torch.autograd import Variable
from torch.functional import F
import numpy as np


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义Hnet损失函数（不一定能用上）
class HNetLoss(_Loss):
    """
    HNet Loss
    """

    def __init__(self, gt_pts, transformation_coefficient, name, usegpu=True):
        """

        :param gt_pts: [x, y, 1]
        :param transformation_coeffcient: [[a, b, c], [0, d, e], [0, f, 1]]
        :param name:
        :return: 
        """
        super(HNetLoss, self).__init__()

        self.gt_pts = gt_pts

        self.transformation_coefficient = transformation_coefficient
        self.name = name
        self.usegpu = usegpu

    def _hnet_loss(self):
        """

        :return:
        """
        H, preds = self._hnet()
        x_transformation_back = torch.matmul(torch.inverse(H), preds)
        loss = torch.mean(torch.pow(self.gt_pts.t()[0, :] - x_transformation_back[0, :], 2))

        return loss

    def _hnet(self):
        """

        :return:
        """
        self.transformation_coefficient = torch.cat((self.transformation_coefficient, torch.tensor([1.0])),
                                                    dim=0)
        H_indices = torch.tensor([0, 1, 2, 4, 5, 7, 8])
        H_shape = 9
        H = torch.zeros(H_shape)
        H.scatter_(dim=0, index=H_indices, src=self.transformation_coefficient)
        H = H.view((3, 3))

        pts_projects = torch.matmul(H, self.gt_pts.t())

        Y = pts_projects[1, :]
        X = pts_projects[0, :]
        Y_One = torch.ones(Y.size())
        Y_stack = torch.stack((torch.pow(Y, 3), torch.pow(Y, 2), Y, Y_One), dim=1).squeeze()
        w = torch.matmul(torch.matmul(torch.inverse(torch.matmul(Y_stack.t(), Y_stack)),
                                      Y_stack.t()),
                         X.view(-1, 1))

        x_preds = torch.matmul(Y_stack, w)
        preds = torch.stack((x_preds.squeeze(), Y, Y_One), dim=1).t()
        return (H, preds)

    def _hnet_transformation(self):
        """
        """
        H, preds = self._hnet()
        x_transformation_back = torch.matmul(torch.inverse(H), preds)

        return x_transformation_back

    def forward(self, input, target, n_clusters):
        return self._hnet_loss(input, target)

def calculate_means(pred, gt, n_objects, max_n_objects, usegpu):
    """pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances"""

    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    pred_repeated = pred.unsqueeze(2).expand(
        bs, n_loc, n_instances, n_filters)  # bs, n_loc, n_instances, n_filters
    # bs, n_loc, n_instances, 1
    gt_expanded = gt.unsqueeze(3)

    pred_masked = pred_repeated * gt_expanded

    means = []
    for i in range(bs):
        _n_objects_sample = n_objects[i]
        # n_loc, n_objects, n_filters
        _pred_masked_sample = pred_masked[i, :, : _n_objects_sample]
        # n_loc, n_objects, 1
        _gt_expanded_sample = gt_expanded[i, :, : _n_objects_sample]

        _mean_sample = _pred_masked_sample.sum(
            0) / _gt_expanded_sample.sum(0)  # n_objects, n_filters
        if (max_n_objects - _n_objects_sample) != 0:
            n_fill_objects = int(max_n_objects - _n_objects_sample)
            _fill_sample = torch.zeros(n_fill_objects, n_filters)
            if usegpu:
                _fill_sample = _fill_sample.cuda()
            _fill_sample = Variable(_fill_sample)
            _mean_sample = torch.cat((_mean_sample, _fill_sample), dim=0)
        means.append(_mean_sample)

    means = torch.stack(means)

    # means = pred_masked.sum(1) / gt_expanded.sum(1)
    # # bs, n_instances, n_filters

    return means


def calculate_variance_term(pred, gt, means, n_objects, delta_v, norm=2):
    """pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances
       means: bs, n_instances, n_filters"""

    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    # bs, n_loc, n_instances, n_filters
    means = means.unsqueeze(1).expand(bs, n_loc, n_instances, n_filters)
    # bs, n_loc, n_instances, n_filters
    pred = pred.unsqueeze(2).expand(bs, n_loc, n_instances, n_filters)
    # bs, n_loc, n_instances, n_filters
    gt = gt.unsqueeze(3).expand(bs, n_loc, n_instances, n_filters)

    _var = (torch.clamp(torch.norm((pred - means), norm, 3) -
                        delta_v, min=0.0) ** 2) * gt[:, :, :, 0]

    var_term = 0.0
    for i in range(bs):
        _var_sample = _var[i, :, :n_objects[i]]  # n_loc, n_objects
        _gt_sample = gt[i, :, :n_objects[i], 0]  # n_loc, n_objects

        var_term += torch.sum(_var_sample) / torch.sum(_gt_sample)
    var_term = var_term / bs

    return var_term


def calculate_distance_term(means, n_objects, delta_d, norm=2, usegpu=True):
    """means: bs, n_instances, n_filters"""

    bs, n_instances, n_filters = means.size()

    dist_term = 0.0
    for i in range(bs):
        _n_objects_sample = int(n_objects[i])

        if _n_objects_sample <= 1:
            continue

        _mean_sample = means[i, : _n_objects_sample, :]  # n_objects, n_filters
        means_1 = _mean_sample.unsqueeze(1).expand(
            _n_objects_sample, _n_objects_sample, n_filters)
        means_2 = means_1.permute(1, 0, 2)

        diff = means_1 - means_2  # n_objects, n_objects, n_filters

        _norm = torch.norm(diff, norm, 2)

        margin = 2 * delta_d * (1.0 - torch.eye(_n_objects_sample))
        if usegpu:
            margin = margin.cuda()
        margin = Variable(margin)

        _dist_term_sample = torch.sum(
            torch.clamp(margin - _norm, min=0.0) ** 2)
        _dist_term_sample = _dist_term_sample / \
            (_n_objects_sample * (_n_objects_sample - 1))
        dist_term += _dist_term_sample

    dist_term = dist_term / bs

    return dist_term


def calculate_regularization_term(means, n_objects, norm):
    """means: bs, n_instances, n_filters"""

    bs, n_instances, n_filters = means.size()

    reg_term = 0.0
    for i in range(bs):
        _mean_sample = means[i, : n_objects[i], :]  # n_objects, n_filters
        _norm = torch.norm(_mean_sample, norm, 1)
        reg_term += torch.mean(_norm)
    reg_term = reg_term / bs

    return reg_term


def discriminative_loss(input, target, n_objects,
                        max_n_objects, delta_v, delta_d, norm, usegpu):
    """input: bs, n_filters, fmap, fmap
       target: bs, n_instances, fmap, fmap
       n_objects: bs"""

    alpha = beta = 1.0
    gamma = 0.001

    bs, n_filters, height, width = input.size()
    n_instances = target.size(1)

    input = input.permute(0, 2, 3, 1).contiguous().view(
        bs, height * width, n_filters)
    target = target.permute(0, 2, 3, 1).contiguous().view(
        bs, height * width, n_instances)

    cluster_means = calculate_means(
        input, target, n_objects, max_n_objects, usegpu)

    var_term = calculate_variance_term(
        input, target, cluster_means, n_objects, delta_v, norm)
    dist_term = calculate_distance_term(
        cluster_means, n_objects, delta_d, norm, usegpu)
    reg_term = calculate_regularization_term(cluster_means, n_objects, norm)

    loss = alpha * var_term + beta * dist_term + gamma * reg_term

    return loss


class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var, delta_dist, norm,
                 size_average=True, usegpu=True):
        super(DiscriminativeLoss, self).__init__(size_average)
        self.delta_var = float(delta_var)
        self.delta_dist = float(delta_dist)
        # 范数
        self.norm = int(norm)
        self.usegpu = usegpu

        assert self.norm in [1, 2]

    def forward(self, input, target, n_objects, max_n_objects):
        return discriminative_loss(input, target, n_objects, max_n_objects,
                                   self.delta_var, self.delta_dist, self.norm,
                                   self.usegpu)
