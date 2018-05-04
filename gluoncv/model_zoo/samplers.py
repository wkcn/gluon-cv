# pylint: disable=arguments-differ
"""Samplers for positive/negative/ignore sample selections.
This module is used to select samples during training.
Based on different strategies, we would like to choose different number of
samples as positive, negative or ignore(don't care). The purpose is to alleviate
unbalanced training target in some circumstances.
The output of sampler is an NDArray of the same shape as the matching results.
Note: 1 for positive, -1 for negative, 0 for ignore.
"""
from __future__ import absolute_import
import numpy as np
from mxnet import gluon
from mxnet import nd
import mxnet as mx


class NaiveSampler(gluon.HybridBlock):
    """A naive sampler that take all existing matching results.
    There is no ignored sample in this case.
    """
    def __init__(self):
        super(NaiveSampler, self).__init__()

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        marker = F.ones_like(x)
        y = F.where(x >= 0, marker, marker * -1)
        return y

class OHEMSamplerOP(mx.operator.CustomOp):
    def __init__(self, ratio, min_samples, thresh):
        super(OHEMSamplerOP, self).__init__()
        self._ratio = ratio
        self._min_samples = min_samples
        self._thresh = thresh
    def forward(self, is_train, req, in_data, out_data, aux):
        """Forward"""
        x, logits, ious = in_data
        F = nd
        num_positive = F.sum(x > -1, axis=1)
        num_negative = self._ratio * num_positive
        num_total = x.shape[1]  # scalar
        num_negative = F.minimum(F.maximum(self._min_samples, num_negative),
                                 num_total - num_positive)
        positive = logits.slice_axis(axis=2, begin=1, end=-1)
        background = logits.slice_axis(axis=2, begin=0, end=1).reshape((0, -1))
        maxval = positive.max(axis=2)
        esum = F.exp(logits - maxval.reshape((0, 0, 1))).sum(axis=2)
        score = -F.log(F.exp(background - maxval) / esum)
        mask = F.ones_like(score) * -1
        score = F.where(x < 0, score, mask)  # mask out positive samples
        if len(ious.shape) == 3:
            ious = F.max(ious, axis=2)
        score = F.where(ious < self._thresh, score, mask)  # mask out if iou is large
        argmaxs = F.argsort(score, axis=1, is_ascend=False)

        # neg number is different in each batch, using dynamic numpy operations.
        y = np.zeros(x.shape)
        y[np.where(x.asnumpy() >= 0)] = 1  # assign positive samples
        argmaxs = argmaxs.asnumpy()
        for i, num_neg in zip(range(x.shape[0]), num_negative.asnumpy().astype(np.int32)):
            indices = argmaxs[i, :num_neg]
            y[i, indices.astype(np.int32)] = -1  # assign negative samples
        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)

@mx.operator.register('OHEMSamplerOP')
class OHEMSamplerProp(mx.operator.CustomOpProp):
    def __init__(self, ratio, min_samples, thresh):
        super(OHEMSamplerProp, self).__init__(need_top_grad = False)
        self._ratio = float(ratio)
        self._min_samples = int(min_samples)
        self._thresh = float(thresh) 
    def list_arguments(self):
        return ['x', 'logits', 'ious']
    def list_outputs(self):
        return ['y']
    def infer_shape(self, in_shape):
        oshape = in_shape[0]
        return in_shape, [oshape]
    def create_operator(self, ctx, shapes, dtypes):
        return OHEMSamplerOP(self._ratio, self._min_samples, self._thresh)

class OHEMSampler(gluon.HybridBlock):
    """A sampler implementing Online Hard-negative mining.
    As described in paper https://arxiv.org/abs/1604.03540.

    Parameters
    ----------
    ratio : float
        Ratio of negative vs. positive samples. Values >= 1.0 is recommended.
    min_samples : int, default 0
        Minimum samples to be selected regardless of positive samples.
        For example, if positive samples is 0, we sometimes still want some num_negative
        samples to be selected.
    thresh : float, default 0.5
        IOU overlap threshold of selected negative samples. IOU must not exceed
        this threshold such that good matching anchors won't be selected as
        negative samples.

    """
    def __init__(self, ratio, min_samples=0, thresh=0.5):
        super(OHEMSampler, self).__init__()
        assert ratio > 0, "OHEMSampler ratio must > 0, {} given".format(ratio)
        self._ratio = ratio
        self._min_samples = min_samples
        self._thresh = thresh

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, logits, ious):
        return F.Custom(x, logits, ious, op_type = 'OHEMSamplerOP', ratio = self._ratio, min_samples = self._min_samples, thresh = self._thresh)
