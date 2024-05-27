import torch
from model import SegmentationModel
from dataset_processing import validset
import matplotlib.pyplot as plt 


def show_image(image, mask, pred_image=None):


    if pred_image is None:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.set_title('IMAGE')
        ax1.imshow(image.permute(1, 2, 0).squeeze(), cmap='gray')
        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask.permute(1, 2, 0).squeeze(), cmap='gray')
    else:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
        ax1.set_title('IMAGE')
        ax1.imshow(image.permute(1, 2, 0).squeeze(), cmap='gray')
        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask.permute(1, 2, 0).squeeze(), cmap='gray')
        ax3.set_title('MODEL OUTPUT')
        ax3.imshow(pred_image.permute(1, 2, 0).squeeze(), cmap='gray')
    
    plt.show()


idx = 20
DEVICE = 'cuda'

model = SegmentationModel()
model.to(DEVICE)

model.load_state_dict(torch.load('best-model.pt'))
image,mask = validset[idx]

logits_mask = model(image.to(DEVICE).unsqueeze(0))
pred_mask = torch.sigmoid(logits_mask)
pred_mask = (pred_mask>0.5)*1.0


show_image(image,mask,pred_mask.detach().cpu().squeeze(0))