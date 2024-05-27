from tqdm import tqdm
from model import SegmentationModel
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset_processing import trainset,validset
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)



LR = config['LR']
EPOCHS = config['EPOCHS']
BATCH_SIZE = config['BATCH_SIZE']
DEVICE = config['DEVICE']

trainloader = DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True)
validloader = DataLoader(validset,batch_size=BATCH_SIZE)

model = SegmentationModel()
model.to(DEVICE)

def train(dataloader,model,optimizer):

  model.train()
  total_loss =0.0

  for images,masks in tqdm(dataloader):

    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    optimizer.zero_grad()
    logits,loss = model(images,masks)
    loss.backward()
    optimizer.step()

    total_loss +=loss.item()

  return total_loss/len(dataloader)

def eval(dataloader,model):

  model.eval()
  total_loss =0.0
  with torch.no_grad():
    for images,masks in tqdm(dataloader):

      images = images.to(DEVICE)
      masks = masks.to(DEVICE)

      logits,loss = model(images,masks)
      total_loss +=loss.item()

    return total_loss/len(dataloader)

optimizer = torch.optim.Adam(model.parameters(),lr=LR)
best_loss = np.Inf

for i in range(EPOCHS):
  train_loss = train(trainloader,model,optimizer)
  valid_loss = eval(validloader,model)

  if valid_loss < best_loss:
    torch.save(model.state_dict(),'best-model.pt')
    print("SAVED-MODEL")
    best_loss = valid_loss

  print(f'Epoch:{i+1} Train Loss : {train_loss} Valid Loss :{valid_loss}')