import torch
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F 
import logging


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """


    def __init__(self, polyLossesWithCoeffs, classLossesWithCoeffs):
        """Creates the matcher

        Params:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            
        """
        super().__init__()
        self.polyLossesWithCoeffs = polyLossesWithCoeffs
        self.classLossesWithCoeffs = classLossesWithCoeffs

    @torch.no_grad()
    def forward(self, pred_polys, class_logits, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        B = class_logits.shape[0]
        out = []
        for i in range(B):
            tgt = targets[i]
            
            C = torch.zeros((tgt.shape[0], pred_polys[i].shape[0]))
            for name, loss_coeff in self.polyLossesWithCoeffs.items():
                loss, coeff = loss_coeff
                if tgt.numel() == 0:
                    continue
                loss_n = loss(pred_polys[i, :, :].unsqueeze(0), tgt.unsqueeze(1)).detach().cpu()
                if(torch.isnan(loss_n).any()):
                    logging.error(f"Loss has nan at {loss_n}")
                    logging.error(f"pred_polys has nan at {pred_polys[i, :, :].unsqueeze(0)}")
                    logging.error(f"tgt has nan at {tgt.unsqueeze(1)}")
                if(loss_n.shape != C.shape):
                    loss_n = loss_n.squeeze()
                C += coeff*loss_n
            
            for name, loss_coeff in self.classLossesWithCoeffs.items():
                loss, coeff = loss_coeff
                loss_n = F.binary_cross_entropy_with_logits(class_logits[i].unsqueeze(0), torch.ones_like(class_logits[i]).unsqueeze(0), reduction="none").detach().cpu()
                if(torch.isnan(loss_n).any()):
                    logging.error(f"Loss has nan at {loss_n}")
                    logging.error(f"pred_polys has nan at {class_logits[i].unsqueeze(0)}")
                    logging.error(f"tgt has nan at {tgt.unsqueeze(1)}")
                if(loss_n.shape != C.shape):
                    loss_n = loss_n.squeeze()
                C += coeff*loss_n
            row_ind, col_ind = linear_sum_assignment(C.T)
            out.append((row_ind, col_ind))
        return out
