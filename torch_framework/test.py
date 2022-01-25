import numpy as np
import torch
import torch.nn.functional as F

# for the regression task
def predict_reg(x,model,device):
  x_in = torch.tensor(x).float().unsqueeze(-1)
  model.eval()
  with torch.no_grad():
    y_hat = model(x_in.to(device)).squeeze().cpu().numpy()
  return y_hat


# for each different type of picture predict classes
def sample_random_cl(label,dataset):
  mask = (dataset.targets == label).cpu().numpy()
  idx = np.where(mask == True)[0]
  rnd_idx = np.random.choice(idx,1)[0]
  return dataset[rnd_idx][0], dataset[rnd_idx][1]


def predict_soft(image,model,device):
  model.eval()
  image = image.unsqueeze(0)
  with torch.no_grad():
    y_hat = model(image.to(device))
    y_hat = F.softmax(y_hat,dim = 1).cpu().numpy()
  return y_hat

def predict_cl(test_loader,model,device):
  y_hat = np.array([],dtype = "int")
  y_true = np.array([],dtype = "int")]
  model.to(device)
  model.eval()
  with torch.no_grad():
    for x,y in test_loader:
      pred = model(x.to(device)).cpu().numpy()
      pred = np.argmax(pred,axis = 1)
      y_hat = np.concatenate((y_hat,pred))
      y_true = np.concatenate((y_true,y))
  return y_true, y_hat

