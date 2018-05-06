"""SSD training target generator."""
from __future__ import absolute_import

from mxnet import nd
from mxnet.gluon import HybridBlock
from ..matchers import CompositeMatcher, BipartiteMatcher, MaximumMatcher
from ..samplers import OHEMSampler
from ..coders import MultiClassEncoder, NormalizedBoxCenterEncoder
from ..bbox import BBoxCenterToCorner
import mxnet as mx

class box_iou_sym(mx.operator.CustomOp):
    def __init__(self):
        super(box_iou_sym, self).__init__()
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], mx.nd.contrib.box_iou(in_data[0], in_data[1]))
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)

@mx.operator.register('box_iou_sym')
class box_iou_prop(mx.operator.CustomOpProp):
    def __init__(self):
        super(box_iou_prop, self).__init__(need_top_grad = False)
    def list_arguments(self):
        return ['lhs', 'rhs']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        assert in_shape[0][-1] == 4
        assert in_shape[1][-1] == 4
        return in_shape, [in_shape[0][:-1] + in_shape[1][:-1]]
    def create_operator(self, ctx, shapes, dtypes):
        return box_iou_sym()

class SSDTargetGenerator(HybridBlock):
    """Training targets generator for Single-shot Object Detection.

    Parameters
    ----------
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    neg_thresh : float
        IOU overlap threshold for negative mining, default is 0.5.
    negative_mining_ratio : float
        Ratio of hard vs positive for negative mining.
    stds : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.
    """
    def __init__(self, iou_thresh=0.5, neg_thresh=0.5, negative_mining_ratio=3,
                 stds=(0.1, 0.1, 0.2, 0.2), **kwargs):
        super(SSDTargetGenerator, self).__init__(**kwargs)
        self._matcher = CompositeMatcher([BipartiteMatcher(), MaximumMatcher(iou_thresh)])
        self._sampler = OHEMSampler(negative_mining_ratio, thresh=neg_thresh)
        self._cls_encoder = MultiClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoder(stds=stds)
        self._center_to_corner = BBoxCenterToCorner(split=False)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, anchors, cls_preds, gt_boxes, gt_ids):
        anchors = self._center_to_corner(anchors.reshape((-1, 4)))
        ious = F.transpose(F.Custom(anchors, gt_boxes, op_type = 'box_iou_sym'), (1, 0, 2))
        matches = self._matcher(ious)
        samples = self._sampler(matches, cls_preds, ious)
        cls_targets = self._cls_encoder(samples, matches, gt_ids)
        box_targets, box_masks = self._box_encoder(samples, matches, anchors, gt_boxes)
        return cls_targets, box_targets, box_masks
