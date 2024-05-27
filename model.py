from torch import nn

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

ENCODER = config['ENCODER']
WEIGHTS = config['WEIGHTS']

class SegmentationModel(nn.Module):

  def __init__(self):
    super(SegmentationModel,self).__init__()

    self.backbone = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None
    )

  def forward(self,images,masks=None):

    logits = self.backbone(images)
    if masks !=None:
      return logits,DiceLoss(mode='binary')(logits,masks) + nn.BCEWithLogitsLoss()(logits,masks)

    return logits

