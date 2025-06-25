import sys
sys.path.insert(0, "/home/r4hul/Projects/row_detection/")
from models.row_column_former.resnet50fpn import ResNet50FPN
from models.row_column_former.row_column_decoder import RowColumnAttentionDecoder
from models.row_column_former.row_column_encoder import RowColumnAttentionEncoder
from mmengine.model import BaseModel
from mmengine.registry import MODELS
import torch
from models.loss_manager_row_col import LossManager
@MODELS.register_module(force=True)
class RowColumnFormer(BaseModel):
    def __init__(self, data_preprocessor=None, M=6, d_model=256, nhead=8, T=6, N=72, H=384, W=640, layers=4, polylossWithCoeffs=None, classLossesWithCoeffs=None, viz=False, atten_viz=False):
        super().__init__(data_preprocessor=data_preprocessor)
        self.backbone = ResNet50FPN(out_channels=d_model)
        dummy_input = torch.randn(1, 3, H, W)
        dummy_input = self.backbone(dummy_input)
        B, _, H, W = dummy_input.size()
        self.decoder = RowColumnAttentionDecoder(embed_dim=d_model, num_layers=T, num_queries=M, num_heads=nhead)
        self.encoder = RowColumnAttentionEncoder(embed_dim=d_model, H=H, W=W, num_layers=T, num_heads=nhead)
        self.ffns = torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Linear(d_model, d_model), torch.nn.ReLU()) for _ in range(layers)])
        self.head = torch.nn.Sequential(torch.nn.Linear(d_model, N+3))
        self.loss = LossManager(polylossWithCoeffs, classLossesWithCoeffs)
        
    def train_step(self, data, optim_wrapper):
        inputs_dict = self.data_preprocessor(data, True)
        with torch.set_grad_enabled(True):

            pred_polys, conf = self._forward(inputs_dict["images"])
            loss, loss_dict = self.loss(pred_polys, conf, inputs_dict["targets"])
            optim_wrapper.update_params(loss)

        return loss_dict # For Reporting

    def val_step(self, data):
        inputs_dict = self.data_preprocessor(data, True)
        report = {}
        self.eval()
        with torch.no_grad():
            points, conf = self.forward(inputs_dict["images"], mode="predict")
            _, loss_dict = self.loss(points, conf, inputs_dict["targets"])
        self.train()
        report.update(loss_dict)
        for key, value in report.items():
            report[key] = value.item()
        report["predictions"] = [points.cpu().detach().clone(), conf.sigmoid().cpu().detach().clone()]
        return report
      
    def test_step(self, data):
        return self.val_step(data)

    def _forward(self, x):

        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.ffns(x)
        ret = self.head(x)
        
        return ret[:, :, :-1], ret[:, :, -1]
    
    def forward(self, images, targets=None, mode="loss",**kwargs):

        if mode == "loss":

            outs = self._forward(images)

            pnts, conf = outs
            loss, loss_dict = self.loss(pnts, conf, targets)
            return loss_dict
        else:
            outs = self._forward(images)
            polys, conf = outs
            return polys, conf.sigmoid()
        
if __name__ == "__main__":
    model = RowColumnFormer()
    input_imgs = torch.randn(1, 3, 384, 640)
    print(model._forward(input_imgs))
