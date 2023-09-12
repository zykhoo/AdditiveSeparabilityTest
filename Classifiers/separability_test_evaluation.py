import numpy as np 
import glob
import pandas as pd
import torch
import os
import time
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
import random
from torch.func import hessian


x = np.linspace(-3.,3.,30)
y = np.linspace(-3.,3.,30)
z = np.linspace(-3.,3.,30)

functions = [lambda x : x, lambda x: x**2, lambda x: (x/3)**3, lambda x: 1/(x+4), lambda x: np.sin(x), lambda x: np.cos(x), lambda x: np.sin(x)**2, lambda x: np.cos(x)**2,
				lambda x: np.exp(x), lambda x: np.log(x+4), lambda x: np.sqrt(np.abs(x)), lambda x: np.cbrt(x)]


loc = os.getcwd()

# define model
def softplus(x):
    return torch.log(torch.exp(x)+1)

def dsoftplus(x):
    return torch.exp(x)/ (torch.exp(x)+1)


from sklearn.model_selection import train_test_split


# PINN
class Net(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Net , self).__init__()
        self.hidden_layer_1 = torch.nn.Linear( input_size, hidden_size, bias=True)
        self.hidden_layer_2 = torch.nn.Linear( hidden_size, hidden_size, bias=True)
        self.output_layer = torch.nn.Linear( hidden_size, output_size , bias=True)
        
    def forward(self, x):
        x = softplus(self.hidden_layer_1(x)) # F.relu(self.hidden_layer_1(x)) # 
        x = softplus(self.hidden_layer_2(x)) # F.relu(self.hidden_layer_2(x)) # 
        x = self.output_layer(x)

        return x

class dNet(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(dNet , self).__init__()
        self.hidden_layer_1 = torch.nn.Linear( input_size, hidden_size, bias=True)
        self.hidden_layer_2 = torch.nn.Linear( hidden_size, hidden_size, bias=True)
        self.output_layer = torch.nn.Linear( hidden_size, output_size , bias=True)
        
    def forward(self, x):
        x = softplus(self.hidden_layer_1(x)) # F.relu(self.hidden_layer_1(x)) # 
        print(x.shape)
        print(seflf.hidden_layer_1.weight)
        x = softplus(self.hidden_layer_2(x)) # F.relu(self.hidden_layer_2(x)) # 
        x = self.output_layer(x)

        return x

def dsoftplus(x):
    return torch.exp(x)/ (torch.exp(x)+1)


class dNet(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(dNet , self).__init__()
        self.hidden_layer_1 = torch.nn.Linear( input_size, hidden_size, bias=True)
        self.hidden_layer_2 = torch.nn.Linear( hidden_size, hidden_size, bias=True)
        self.output_layer = torch.nn.Linear( hidden_size, output_size , bias=True)
        
    def forward(self, x0):
        x1 = softplus(self.hidden_layer_1(x0)) # F.relu(self.hidden_layer_1(x)) # 
        dx1 = dsoftplus(self.hidden_layer_1(x0)) * self.hidden_layer_1.weight[:,0] # torch.einsum('bi,ik->bik', dsoftplus(self.hidden_layer_1(x0)), self.hidden_layer_1.weight)
        x2 = softplus(self.hidden_layer_2(x1)) # F.relu(self.hidden_layer_2(x)) # 
        dx2 = dsoftplus(self.hidden_layer_2(x1)) * (self.hidden_layer_2.weight @ dx1.T).T # torch.einsum('bk,ibk->ibk', dsoftplus(self.hidden_layer_2(x1)),torch.einsum('ibk,kj->ibj', dx1, self.hidden_layer_2.weight))
        dx3 = dx2 @ self.output_layer.weight.T  # torch.squeeze(torch.einsum('ibk,kj->ibj', dx2, self.output_layer.weight.T).permute(1,2,0))
        return dx3

def ddsoftplus(x):
    return torch.exp(x)/ ((torch.exp(x)+1)**2)

class ddNet(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ddNet , self).__init__()
        self.hidden_layer_1 = torch.nn.Linear( input_size, hidden_size, bias=True)
        self.hidden_layer_2 = torch.nn.Linear( hidden_size, hidden_size, bias=True)
        self.output_layer = torch.nn.Linear( hidden_size, output_size , bias=True)
        
    def forward(self, x0):
        x1 = softplus(self.hidden_layer_1(x0))  
        dx1 = dsoftplus(self.hidden_layer_1(x0)) * self.hidden_layer_1.weight[:,0]
        dx12 = dsoftplus(self.hidden_layer_1(x0)) * self.hidden_layer_1.weight[:,-1]
        ddx1 = ddsoftplus(self.hidden_layer_1(x0)) * self.hidden_layer_1.weight[:,0] * self.hidden_layer_1.weight[:,-1]
        x2 = softplus(self.hidden_layer_2(x1))  
        dx2 = dsoftplus(self.hidden_layer_2(x1)) * (self.hidden_layer_2.weight @ dx1.T).T 
        ddx2 = ddsoftplus(self.hidden_layer_2(x1)) * (self.hidden_layer_2.weight @ dx1.T).T * (self.hidden_layer_2.weight @ dx12.T).T + dsoftplus(self.hidden_layer_2(x1)) * (self.hidden_layer_2.weight @ ddx1.T).T 
        x3 = self.output_layer(x2)
        dx3 = ddx2 @ self.output_layer.weight.T  
        return dx3

np.random.seed(1)
xv, yv, zv = (np.random.rand(30,30,30)-0.5)*6, (np.random.rand(30,30,30)-0.5)*6, (np.random.rand(30,30,30)-0.5)*6
z3 = torch.tensor(np.vstack((np.ravel(xv), np.ravel(yv), np.ravel(zv)))).T
xv, yv = (np.random.rand(30,30)-0.5)*6, (np.random.rand(30,30)-0.5)*6
z2 = torch.tensor(np.vstack((np.ravel(xv), np.ravel(yv)))).T

if torch.cuda.is_available():
  device=torch.device('cuda:0')
else:
  device=torch.device('cpu')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)



def Method7(model, inputs,device):
      start = time.time()
      inputs=torch.autograd.Variable(inputs.clone().detach()).requires_grad_(True).to(device)
      out = 0
      for i in range(inputs.shape[0]):
        out += torch.abs(torch.squeeze(hessian(model)(inputs[i,:].float())[0])[0][-1]).detach().cpu().numpy()
      if inputs.shape[1] == 3: 
        return [out/27000, time.time()-start]
      if inputs.shape[1] == 2: 
        return [out/900, time.time()-start]


def Method8(model, inputs,device):
      start = time.time()
      inputs=torch.autograd.Variable(inputs.clone().detach()).requires_grad_(True).to(device)
      out=torch.mean(torch.abs(model(inputs.float()))).detach().cpu().numpy()
      return [out, time.time()-start]


def Method5(model, inputs,device):
      start = time.time()
      inputs=torch.autograd.Variable(inputs.clone().detach()).requires_grad_(True).to(device)
      out=model(inputs.float())
      dH=torch.autograd.grad(out, inputs, grad_outputs=out.data.new(out.shape).fill_(1),create_graph=True)[0]
      secondderiv=torch.autograd.grad(dH[:,-1], inputs, grad_outputs=dH[:,-1].data.new(dH[:,-1].shape).fill_(1),create_graph=True)[0]
      d2H1 = torch.mean(torch.abs(secondderiv[:,0]))
      return [torch.nan_to_num(d2H1,nan=np.nan).detach().cpu().numpy(), time.time()-start]

def Method6(model, inputs,device):
      start = time.time()
      inputs=torch.autograd.Variable(inputs.clone().detach()).requires_grad_(True).to(device)
      out=model(inputs.float())
      dH=torch.autograd.grad(out, inputs, grad_outputs=out.data.new(out.shape).fill_(1),create_graph=True)[0]
      secondderiv=torch.autograd.grad(dH[:,0], inputs, grad_outputs=dH[:,0].data.new(dH[:,0].shape).fill_(1),create_graph=True)[0]
      d2H1 = torch.mean(torch.abs(secondderiv[:,-1]))
      return [torch.nan_to_num(d2H1,nan=np.nan).detach().cpu().numpy(), time.time()-start]


def Method3(model, inputs,device): # discretised derivative, wrt to median
  start = time.time()
  if inputs.shape[1] == 3: 
    inputs=inputs.clone().detach().to(device).float()
    z_diag = torch.tile(torch.tensor([[torch.median(inputs[:,0]),torch.median(inputs[:,1]),torch.median(inputs[:,2])]]), (inputs.shape[0],1) ).to(device).float()
    z_diag[:,1] = inputs[:,1]
    z_a, z_b, diff = torch.vstack((inputs[:,0],inputs[:,1],z_diag[:,2])).T, torch.vstack((z_diag[:,0],inputs[:,1],inputs[:,2])).T, inputs- z_diag
  if inputs.shape[1] == 2: 
    inputs=inputs.clone().detach().to(device).float()
    z_diag = torch.tile(torch.tensor([[torch.median(inputs[:,0]),torch.median(inputs[:,1])]]), (inputs.shape[0],1) ).to(device).float()
    z_a, z_b, diff = torch.vstack((inputs[:,0],z_diag[:,1])).T, torch.vstack((z_diag[:,0],inputs[:,1])).T, inputs- z_diag
  out = torch.abs(model(z_diag) +model(inputs) - model(z_a) - model(z_b)).detach().cpu().numpy()
  return [np.mean(np.abs(out)), time.time()-start]


def Method4(model, inputs,device): # discretised derivative, wrt to median, normalized
  start = time.time()
  if inputs.shape[1] == 3: 
    inputs=inputs.clone().detach().to(device).float()
    z_diag = torch.tile(torch.tensor([[torch.median(inputs[:,0]),torch.median(inputs[:,1]),torch.median(inputs[:,2])]]), (inputs.shape[0],1) ).to(device).float()
    z_diag[:,1] = inputs[:,1]
    z_a, z_b, diff = torch.vstack((inputs[:,0],inputs[:,1],z_diag[:,2])).T, torch.vstack((z_diag[:,0],inputs[:,1],inputs[:,2])).T, inputs- z_diag
    dist = torch.unsqueeze(diff[:,0] * diff[:,2], 1).detach().cpu().numpy() #torch.unsqueeze(torch.sqrt(torch.sum(diff[:,0:2]**2, 1))*diff[:,2],1).detach().cpu().numpy()
  if inputs.shape[1] == 2: 
    inputs=inputs.clone().detach().to(device).float()
    z_diag = torch.tile(torch.tensor([[torch.median(inputs[:,0]),torch.median(inputs[:,1])]]), (inputs.shape[0],1) ).to(device).float()
    z_a, z_b, diff = torch.vstack((inputs[:,0],z_diag[:,1])).T, torch.vstack((z_diag[:,0],inputs[:,1])).T, inputs- z_diag
    dist = torch.unsqueeze(diff[:,0] * diff[:,1], 1).detach().cpu().numpy() #torch.unsqueeze(diff[:,0]*diff[:,1],1).detach().cpu().numpy()
  out = torch.abs(model(z_diag) +model(inputs) - model(z_a) - model(z_b)).detach().cpu().numpy()
  return [np.mean(np.abs(out[dist!= 0]/dist[dist!= 0])), time.time()-start]


def Method2(model, inputs,device): # discretised derivative, wrt to all squares, normalized
  start = time.time()
  inputs=inputs.clone().detach().to(device).float()
  err = 0
  if inputs.shape[1] == 3: 
    for i in range(len(inputs)-1):
      z_diag = inputs[(np.linspace(i+1,i+len(inputs),num = len(inputs)))%len(inputs)]
      z_diag[:,1] = inputs[:,1]
      z_a, z_b, diff = torch.vstack((inputs[:,0],inputs[:,1],z_diag[:,2])).T, torch.vstack((z_diag[:,0],inputs[:,1],inputs[:,2])).T, inputs- z_diag
      dist = torch.unsqueeze(diff[:,0] * diff[:,2], 1).detach().cpu().numpy() #torch.unsqueeze(torch.sqrt(torch.sum(diff[:,0:2]**2, 1))*diff[:,2],1).detach().cpu().numpy()
      out = torch.abs(model(z_diag) +model(inputs) - model(z_a) - model(z_b)).detach().cpu().numpy()
      err += np.mean(np.abs(out[dist != 0]/dist[dist != 0]))
    return [err/27000, time.time()-start]
  if inputs.shape[1] == 2: 
    for i in range(len(inputs)-1):
      z_diag = inputs[(np.linspace(i+1,i+len(inputs),num = len(inputs)))%len(inputs)]
      z_a, z_b, diff = torch.vstack((inputs[:,0],z_diag[:,1])).T, torch.vstack((z_diag[:,0],inputs[:,1])).T, inputs- z_diag
      dist = torch.unsqueeze(diff[:,0] * diff[:,1], 1).detach().cpu().numpy() #torch.unsqueeze(torch.sqrt(torch.sum((diff)**2, 1)),1).detach().cpu().numpy()
      out = torch.abs(model(z_diag) +model(inputs) - model(z_a) - model(z_b)).detach().cpu().numpy()
      err += np.mean(np.abs(out[dist != 0]/dist[dist != 0]))
    return [err/900, time.time()-start]

def Method1(model, inputs,device): # discretised derivative, wrt to all squares
  start = time.time()
  inputs=inputs.clone().detach().to(device).float()
  err = 0
  if inputs.shape[1] == 3: 
    for i in range(len(inputs)-1):
      z_diag = inputs[(np.linspace(i+1,i+len(inputs),num = len(inputs)))%len(inputs)]
      z_diag[:,1] = inputs[:,1]
      z_a, z_b, diff = torch.vstack((inputs[:,0],inputs[:,1],z_diag[:,2])).T, torch.vstack((z_diag[:,0],inputs[:,1],inputs[:,2])).T, inputs- z_diag
      out = torch.abs(model(z_diag) +model(inputs) - model(z_a) - model(z_b)).detach().cpu().numpy()
      err += np.mean(out)
    return [err/27000, time.time()-start]
  if inputs.shape[1] == 2: 
    for i in range(len(inputs)-1):
      z_diag = inputs[(np.linspace(i+1,i+len(inputs),num = len(inputs)))%len(inputs)]
      z_a, z_b, diff = torch.vstack((inputs[:,0],z_diag[:,1])).T, torch.vstack((z_diag[:,0],inputs[:,1])).T, inputs- z_diag
      out = torch.abs(model(z_diag) +model(inputs) - model(z_a) - model(z_b)).detach().cpu().numpy()
      err += np.mean(out)
    return [err/900, time.time()-start]


ev = 12

df = pd.DataFrame()

for f in tqdm(glob.glob(loc+"/Results-UniformRandomData/*.pt")):
  details = f.split("_")
  if int(details[1])==ev:


    if (int(details[1]) == 0):
        net, ddnet = Net(2,26,1), ddNet(2,26,1)
        net.load_state_dict(torch.load(f))
        ddnet.load_state_dict(torch.load(f))
        net = net.to(device)
        ddnet = ddnet.to(device)
        M1 = Method1(net, z2, device) 
        M2 = Method2(net, z2, device) 
        M3 = Method3(net, z2, device) 
        M4 = Method4(net, z2, device) 
        M5 = Method5(net, z2, device) 
        M6 = Method6(net, z2, device) 
        M7 = Method7(net, z2, device)
        M8 = Method8(ddnet, z2, device)
        data = pd.Series({'seed':int(details[5]), "i":details[1], "j":details[2], "k":details[3],'epochs':int(details[6]),'type':details[4],
				"Method1error":M1[0], "Method1time":M1[1], "Method2error":M2[0], "Method2time":M2[1], "Method3error":M3[0], "Method3time":M3[1],
                                "Method4error":M4[0], "Method4time":M4[1], "Method5error":M5[0], "Method5time":M5[1], "Method6error":M6[0], "Method6time":M6[1],
                                "Method7error":M7[0], "Method7time":M7[1], "Method8error":M8[0], "Method8time":M8[1],
				 }) 
        df = pd.concat([df,data.to_frame().T], ignore_index = True)
    elif int(details[1]) > 0:
        net, ddnet = Net(3,26,1), ddNet(3,26,1)
        net.load_state_dict(torch.load(f))
        ddnet.load_state_dict(torch.load(f))
        net = net.to(device)
        ddnet = ddnet.to(device)
        M1 = Method1(net, z3, device) 
        M2 = Method2(net, z3, device) 
        M3 = Method3(net, z3, device) 
        M4 = Method4(net, z3, device) 
        M5 = Method5(net, z3, device) 
        M6 = Method6(net, z3, device) 
        M7 = Method7(net, z3, device)
        M8 = Method8(ddnet, z3, device)
        data = pd.Series({'seed':int(details[5]), "i":details[1], "j":details[2], "k":details[3],'epochs':int(details[6]),'type':details[4],
				"Method1error":M1[0], "Method1time":M1[1], "Method2error":M2[0], "Method2time":M2[1], "Method3error":M3[0], "Method3time":M3[1],
                                "Method4error":M4[0], "Method4time":M4[1], "Method5error":M5[0], "Method5time":M5[1], "Method6error":M6[0], "Method6time":M6[1],
                                "Method7error":M7[0], "Method7time":M7[1], "Method8error":M8[0], "Method8time":M8[1],
				 }) 
        df = pd.concat([df,data.to_frame().T], ignore_index = True)

print(df)
df.to_csv(loc+"/Results-Evaluated/%s.csv" %ev)
