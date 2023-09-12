import numpy as np 

x = np.linspace(-3.,3.,100)
y = np.linspace(-3.,3.,100)
z = np.linspace(-3.,3.,100)

functions = [lambda x : x, lambda x: x**2, lambda x: (x/3)**3, lambda x: 1/(x+4), lambda x: np.sin(x), lambda x: np.cos(x), lambda x: np.sin(x)**2, lambda x: np.cos(x)**2,
				lambda x: np.exp(x), lambda x: np.log(x+4), lambda x: np.sqrt(np.abs(x)), lambda x: np.cbrt(x)]



import torch
import os
import time
from tqdm import tqdm
import math

loc = os.getcwd()

# define model
def softplus(x):
    return torch.log(torch.exp(x)+1)

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

def lossfuc(model,mat,x,y,device,verbose=False):
    loss=torch.mean((torch.squeeze(y,dim=1)-mat[:,-1])**2)
    return loss


# evaluate loss of dataset 
def get_loss(model,device,bs,mat,verbose=False):
    # this function is used to calculate average loss of a whole dataset
    # rootpath: path of set to be calculated loss
    # model: model
    # trainset: is training set or not
    avg_loss=0
    for count in range(0,len(mat),bs):
      curmat=mat[count:count+bs]
      x=torch.autograd.Variable((curmat[:,:-1]).float(),requires_grad=True)
      y=model(x)
      loss=torch.mean((torch.squeeze(y,dim=1)-curmat[:,-1])**2)
      avg_loss+=loss.detach().cpu().item()
    num_batches=len(mat)//bs
    avg_loss/=num_batches
    if verbose:
        print(' loss=',avg_loss)
    return avg_loss


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            ä¸Šæ¬¡éªŒè¯é›†æŸå¤±å€¼æ”¹å–„åŽç­‰å¾…å‡ ä¸ªepoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            å¦‚æžœæ˜¯Trueï¼Œä¸ºæ¯ä¸ªéªŒè¯é›†æŸå¤±å€¼æ”¹å–„æ‰“å°ä¸€æ¡ä¿¡æ¯
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            ç›‘æµ‹æ•°é‡çš„æœ€å°å˜åŒ–ï¼Œä»¥ç¬¦åˆæ”¹è¿›çš„è¦æ±‚
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if abs(self.counter-self.patience)<5:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        éªŒè¯æŸå¤±å‡å°‘æ—¶ä¿å­˜æ¨¡åž‹ã€‚
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint2.pt')     # è¿™é‡Œä¼šå­˜å‚¨è¿„ä»Šæœ€ä¼˜æ¨¡åž‹çš„å‚æ•°
        # torch.save(model, 'checkpoint2.pt')                 # è¿™é‡Œä¼šå­˜å‚¨è¿„ä»Šæœ€ä¼˜çš„æ¨¡åž‹
        self.val_loss_min = val_loss

def train(net,name,bs,num_epoch,device,wholemat,evalmat,LR,patience):
    starttime = time.time() 
    # function of training process
    # net: the model
    # bs: batch size 
    # num_epoch: max of epoch to run
    # initial_conditions: number of trajectory in train set
    # patience: EarlyStopping parameter
    # c1~c4: hyperparameter for loss function

    avg_lossli=[]
    avg_vallosses=[]
    
    start = time.time()
    lr = LR # initial learning rate
    net=net.to(device)

    early_stopping = EarlyStopping(patience=patience, verbose=False,delta=0.001) # delta
    optimizer=torch.optim.Adam(net.parameters() , lr=lr )
    for epoch in range(num_epoch):

        running_loss=0
        num_batches=0
        
        # train
        shuffled_indices=torch.randperm(len(wholemat))
        net.train()
        for count in range(0,len(wholemat),bs):
            optimizer.zero_grad()

            indices=shuffled_indices[count:count+bs]
            mat=wholemat[indices]

            x=torch.autograd.Variable(torch.tensor(mat[:,:-1]).float(),requires_grad=True)
            y=net(x)

            loss=lossfuc(net,mat,x,y,device)
            loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), 1)

            optimizer.step()

            # compute some stats
            running_loss += loss.detach().item()

            num_batches+=1
            torch.cuda.empty_cache()



        avg_loss = running_loss/num_batches
        elapsed_time = time.time() - start
        
        avg_lossli.append(avg_loss)
        
        
        # evaluate
        net.eval()
        avg_val_loss=get_loss(net,device,len(evalmat),evalmat)
        avg_vallosses.append(avg_val_loss)
        
        if epoch % 100 == 0 : 
            # print(' ')
            print('epoch=',epoch, ' time=', elapsed_time,
                  ' loss=', avg_loss ,' val_loss=',avg_val_loss,)

        # if time.time() - starttime > smarker:
        #     torch.save(net.state_dict(), "%s_%s_%s.pt" %(name,epoch,time.time()-starttime))
        #     smarker += 20
        
        if epoch%100 == 0:
            torch.save(net.state_dict(), 'checkpoint2.pt')
        
        if math.isnan(running_loss):
            text_file = open("nan_report.txt", "w")
            text_file.write('name=%s at epoch %s' %(name, epoch))
            text_file.close()
            print("saving this file and ending the training")
            net.load_state_dict(torch.load('checkpoint2.pt')) 
            return net,epoch,avg_vallosses,avg_lossli

        
        
        early_stopping(avg_val_loss,net)
        if early_stopping.early_stop:
            print('Early Stopping')
            break
            
    net.load_state_dict(torch.load('checkpoint2.pt')) #net=torch.load('checkpointpt')
    return net,epoch,avg_vallosses,avg_lossli


for seed in range(1):
	if torch.cuda.is_available():
	  device=torch.device('cuda:0')
	else:
	  device=torch.device('cpu')
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	for i in range(len(functions)):
		for j in range(len(functions)):


		   	 	LR = 0.01 

		   	 	xv, yv = np.meshgrid(x, y)
		   	 	xv, yv = (np.random.rand(100,100)-0.5)*6, (np.random.rand(100,100)-0.5)*6

		   	 	f = functions[i](xv) * functions[j](yv) #+0.01*(np.random.random(xv.shape)-0.5)*2
		   	 	data = torch.tensor(np.vstack((np.ravel(xv), np.ravel(yv), np.ravel(f)))).T.to(device)
		   	 	wholemat,evalmat=train_test_split(data, train_size=0.8, random_state=seed)

		   	 	net = Net(2,26,1)
		   	 	starttime = time.time() 	    
		   	 	results = train(net,name="",bs=128,num_epoch=150001,device=device, wholemat=wholemat,evalmat=evalmat,LR=LR,patience=500)
		   	 	net, epochs = results[0], results[1]
		   	 	NNtraintime = time.time()-starttime
		   	 	torch.save(net.state_dict(), '%s/Results-UniformRandomData/%s_%s_%s_%s_%s_%s_%s_%s.pt' 
							%(loc,"NN",0,int(i+1),int(j+1),"x",seed,epochs,NNtraintime))

