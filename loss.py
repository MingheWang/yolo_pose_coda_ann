# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, kpt_label=False):
        super(ComputeLoss, self).__init__()
        self.kpt_label = kpt_label
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))    # BCEWithLogitsLoss就是把Sigmoid-BCELoss合成一步
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCE_kptv = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'nkpt':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        #------------------------#
        #   predictions:
        #       B, 3, 80, 80, 57
        #       B, 3, 40, 40, 57
        #       B, 3, 20, 20, 57
        #       B, 3, 10, 10, 57
        #   targets:
        #       (nt,40)
        #------------------------#
        device = targets.device
        lcls, lbox, lobj, lkpt, lkptv = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89], device=device) / 10.0
        
        #######************########
        #----------------------------------------------------------------------------------------------------------#
        #   tcls:   [cls_1,cls_2,cls_3,cls_4]   cls_i.shape=(ni,)   每个待回归的锚框对应的目标类别
        #   tbox:   [box_1,box_2,box_3,box_4]   box_i.shape=(ni,4)  每个待回归的锚框(dx,dy,w,h)相对于网格点的回归量
        #   tkpt:   [kpt_1,kpt_2,kpt_3,kpt_4]   kpt_i.shape=(ni,34) 每个待回归的人体关键点(d_kx,d_ky)相对于网格点的回归量
        #   indices:    每个对回归的锚框对应的image_id, anchor_id, 以及该锚框在特征图上的网格点坐标(grid_y,grid_x)
        #   [anch_1,anch_2,anch_3,anch_4]   anch_i.shape=(ni,2) 每个待回归的目标框的宽高值
        #----------------------------------------------------------------------------------------------------------#
        #######************########
        tcls, tbox, tkpt, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj        # shape=(b,3,h,w)

            n = b.shape[0]  # number of targets         # 正样本的数目
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets   # 从预测特征图中挑选出正样本的预测特征向量 shape=(pos_n, 57)

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5       # -0.5 ~ 1.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]      # (0 ~ 2)^2 ---> (0 ~ 4)
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                if self.kpt_label:
                    #Direct kpt prediction
                    pkpt_x = ps[:, 6::3] * 2. - 0.5
                    pkpt_y = ps[:, 7::3] * 2. - 0.5
                    pkpt_score = ps[:, 8::3]
                    #mask
                    kpt_mask = (tkpt[i][:, 0::2] != 0)
                    lkptv += self.BCEcls(pkpt_score, kpt_mask.float()) 
                    #l2 distance based loss
                    #lkpt += (((pkpt-tkpt[i])*kpt_mask)**2).mean()  #Try to make this loss based on distance instead of ordinary difference
                    #oks based loss
                    d = (pkpt_x-tkpt[i][:,0::2])**2 + (pkpt_y-tkpt[i][:,1::2])**2
                    s = torch.prod(tbox[i][:,-2:], dim=1, keepdim=True)
                    kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0))/torch.sum(kpt_mask != 0)
                    lkpt += kpt_loss_factor*((1 - torch.exp(-d/(s*(4*sigmas**2)+1e-9)))*kpt_mask).mean()
                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lkptv *= self.hyp['cls']
        lkpt *= self.hyp['kpt']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lkpt + lkptv
        return loss * bs, torch.cat((lbox, lobj, lcls, lkpt, lkptv, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        #------------------------#
        #   predictions:
        #       B, 3, 80, 80, 57
        #       B, 3, 40, 40, 57
        #       B, 3, 20, 20, 57
        #       B, 3, 10, 10, 57
        #   targets:
        #       (nt,40)
        #------------------------#
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, tkpt, indices, anch = [], [], [], [], []
        if self.kpt_label:
            gain = torch.ones(41, device=targets.device)  # normalized to gridspace gain        
        else:
            gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        # ai : ---->  [0,1,2] --> [[0],[1],[2]] --> [[0,0,..],[1,1,..],[2,2,..]]    shape=(3,nt)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # (3,nt,40) + (3,nt,1) --> (3,nt,41)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m                # 左测、上侧、右侧、下侧
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            if self.kpt_label:
                gain[2:40] = torch.tensor(p[i].shape)[19*[3, 2]]  # xyxy gain       # 特征图大小
            else:
                gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain          # (3, nt, 41) * (41,)       # 乘以对应的特征图尺度后，将所有标签坐标对应到特征图坐标
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio     # (3,nt,2)/(3,1,2)
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare with 4.0   # shape=(3,nt)
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter  # 剔除相对锚框回归的高宽比(或宽高比)大于4.0的框     # shape=(n,41)      # 选出适合该特征图适合回归的目标框

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse     # gxy表示中心xy相对于"特征图"原点的坐标，gxi表示中心xy相对于"特征图"右下角的坐标
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T  # 中心点距离左上网格点更近
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T  # 中心点距离右下网格点更近
                j = torch.stack((torch.ones_like(j), j, k, l, m))       # shape=(5,n)
                t = t.repeat((5, 1, 1))[j]      # (5,n,41) --> (m,41)    
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]       # (1,n,2) + (5,1,2) = (5,n,2) ---[j]---> (n,2)每个中心坐标的偏置量，-0.5~0.5
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()            # 该回归框中心在"特征图"上的整数坐标(特征图上的网格点)     shape=(n,2)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, -1].long()  # anchor indices
            # indices:  每个对回归的锚框对应的image_id, anchor_id, 以及该锚框在特征图上的网格点坐标(grid_y,grid_x)
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid_(j_i) indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            if self.kpt_label:
                for kpt in range(self.nkpt):        # 将每点在特征图上的坐标转换为相对于(其目标框中心对应的)网格点的偏移量
                    t[:, 6+2*kpt: 6+2*(kpt+1)][t[:,6+2*kpt: 6+2*(kpt+1)] !=0] -= gij[t[:,6+2*kpt: 6+2*(kpt+1)] !=0]
                tkpt.append(t[:, 6:-1])
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        #-------------------------------------#
        #   tcls:   [cls_1,cls_2,cls_3,cls_4]   cls_i.shape=(ni,)   每个待回归的锚框对应的目标类别
        #   tbox:   [box_1,box_2,box_3,box_4]   box_i.shape=(ni,4)  每个待回归的锚框(dx,dy,w,h)相对于网格点的回归量
        #   tkpt:   [kpt_1,kpt_2,kpt_3,kpt_4]   kpt_i.shape=(ni,34) 每个待回归的人体关键点(d_kx,d_ky)相对于网格点的回归量
        #   indices:    每个对回归的锚框对应的image_id, anchor_id, 以及该锚框在特征图上的网格点坐标(grid_y,grid_x)
        #   [anch_1,anch_2,anch_3,anch_4]   anch_i.shape=(ni,2) 每个待回归的目标框的宽高值
        #-------------------------------------#
        return tcls, tbox, tkpt, indices, anch
