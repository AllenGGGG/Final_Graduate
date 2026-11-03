"""Causal one-second queue-arrival load prediction, 100ms sampling, 3s history."""
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

class GRU(nn.Module):
 def __init__(self):
  super().__init__();self.gru=nn.GRU(20,32,batch_first=True);self.head=nn.Linear(32,20)
 def forward(self,x):
  y,_=self.gru(torch.log1p(torch.clamp(x,min=0,max=100)))
  raw=self.head(y[:,-1]).reshape(-1,10,2)
  median=torch.nn.functional.softplus(raw[:,:,0])
  upper=median+torch.nn.functional.softplus(raw[:,:,1])
  return torch.stack((median,upper),-1)

def history_array(history):
 rows=np.zeros((30,20),np.float32)
 for i,row in enumerate(history[-30:]):rows[30-min(30,len(history))+i]=row['load']+row['queue']
 return rows

def windows(path):
 with gzip.open(path,'rt') as stream:rows=[json.loads(line) for line in stream]
 features=np.array([r['load']+r['queue'] for r in rows],np.float32)
 x=[];y=[];times=[]
 for t in range(len(rows)-10):
  x.append(history_array(rows[max(0,t-29):t+1]));y.append(features[t+1:t+11,:10].mean(0));times.append(rows[t]['time'])
 return np.asarray(x,np.float32),np.asarray(y,np.float32),times

def baseline_predictions(x):
 load=x[:,:,:10]
 previous=load[:,-1]
 average=load.mean(1)
 ewma=load[:,0].copy()
 for t in range(1,30):ewma=.2*load[:,t]+.8*ewma
 return {'previous':previous,'moving_average':average,'ewma':ewma}

def loss(pred,target):
 error=target[:,:,None]-pred
 q=torch.tensor([.5,.9],device=pred.device)
 return torch.maximum(q*error,(q-1)*error).mean()

class Predictor:
 def __init__(self,path):
  self.model=GRU();data=torch.load(path,map_location='cpu',weights_only=True);self.model.load_state_dict(data['state_dict']);self.model.eval()
 def __call__(self,history):
  with torch.inference_mode():return self.model(torch.from_numpy(history_array(history)[None]))[0].numpy()

def metrics(pred,y):
 e=y[:,:,None]-pred;q=np.array([.5,.9])
 return {'mae':float(np.abs(y-pred[:,:,0]).mean()),'pinball':float(np.maximum(q*e,(q-1)*e).mean()),
  'p50_coverage':float((y<=pred[:,:,0]).mean()),'p90_coverage':float((y<=pred[:,:,1]).mean()),
  'p50_p90_interval_coverage':float(((y>=pred[:,:,0])&(y<=pred[:,:,1])).mean())}
