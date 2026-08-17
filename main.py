import warnings
warnings.filterwarnings("ignore")
from utils.MvDataloaders import Get_dataloaders
from utils.MvLoad_models import load
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import torch
import scipy.io as scio
import os
import random
import torch.nn as nn
import Nmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

min_max_scaler = preprocessing.MinMaxScaler()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Adaptive learning rate scheduler
class AdaptiveLRScheduler:
    def __init__(self, optimizer, min_lr=1e-6, max_lr=1e-3, patience=50, factor=0.5):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        self.factor = factor
        self.best_loss = float('inf')
        self.counter = 0
        self.lr_history = []
        
    def step(self, current_loss):
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        new_lr = current_lr
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            new_lr = min(current_lr * 1.05, self.max_lr)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                new_lr = max(current_lr * self.factor, self.min_lr)
                self.counter = 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

# Adaptive early stopping mechanism
class AdaptiveEarlyStopping:
    def __init__(self, patience=50, min_delta=0.001, improvement_threshold=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.improvement_threshold = improvement_threshold
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.loss_history = []
        
    def __call__(self, val_loss):
        self.loss_history.append(val_loss)
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                recent_improvement = self._check_recent_improvement()
                if not recent_improvement:
                    self.early_stop = True
        return self.early_stop
    
    def _check_recent_improvement(self):
        if len(self.loss_history) < 5:
            return True
        recent_losses = self.loss_history[-5:]
        improvements = []
        for i in range(1, len(recent_losses)):
            improvement = (recent_losses[i-1] - recent_losses[i]) / recent_losses[i-1]
            improvements.append(improvement)
        return any(imp > self.improvement_threshold for imp in improvements)

setup_seed(5)

NMI_c = []
NMI_cz = []
ACC_c = []
ACC_cz = []

datasets = [
    'Multi-COIL-10'       
    
]
settings = [[1, 32], [1, 32], [0, 32], [1, 64], [1, 64], [0, 64], [1, 32],[1, 64], [1, 32], [0, 64], [1, 96],[1, 32]]
iters_to_add_capacity = [25000] * len(datasets)

for d in [0]:
    DATA = datasets[d]
    share = settings[d][0]
    Batch_size = settings[d][1]
    iters_add_capacity = iters_to_add_capacity[d]
    Epochs = 500
    Net = 'C'
    hidden_dim = 256
    num_heads = 4
    
    
    beta = 40
    capacity = 8
    lr = 1e-5
    weight_decay = 1e-5
    max_grad_norm = 5.0
    z_variables = 12
    dropout_rate = 0.2
    
    print(f"Dataset {DATA} parameters:")
    print(f"  beta: {beta}, capacity: {capacity}")
    print(f"  lr: {lr}, weight_decay: {weight_decay}, max_grad_norm: {max_grad_norm}")
    print(f"  z_variables: {z_variables}, dropout: {dropout_rate}")
    
    runs = 1
    TEST = True
    for i in range(runs):
        model_name = DATA + '.pt'
        print('Run:' + str(i))
        if Net == 'C':
            train_loader, val_loader, view_num, n_clusters, size, norm_methods = Get_dataloaders(
                batch_size=Batch_size,
                DATANAME=DATA + '.mat',
                val_ratio=0.1
            )
            print('Iters:' + str(size / Batch_size * Epochs))
            cont_capacity = [capacity, beta, iters_add_capacity]
            disc_capacity = [np.log(n_clusters), beta, iters_add_capacity]

        latent_spec = {'cont': z_variables, 'disc': [n_clusters]}
        use_cuda = torch.cuda.is_available()
        print("cuda is available?")
        print(use_cuda)
        
       
        img_size = (1, 32, 32)
        if TEST == False:
            from multi_vae.MvModels import VAE

            model = VAE(latent_spec=latent_spec, img_size=img_size,
                        view_num=view_num, use_cuda=use_cuda,
                        Network=Net, hidden_dim=hidden_dim, 
                        shareAE=share, num_heads=num_heads,
                        dropout=dropout_rate)
            if use_cuda:
                model.cuda()

            print(model)

            from torch import optim
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            from multi_vae.MvTraining import Trainer

           
            trainer = Trainer(model, optimizer,
                              cont_capacity=cont_capacity,
                              disc_capacity=disc_capacity, 
                              view_num=view_num, 
                              use_cuda=use_cuda, 
                              DATA=DATA,
                              weight_decay=weight_decay,
                              max_grad_norm=max_grad_norm)

            lr_scheduler = AdaptiveLRScheduler(optimizer)
            early_stopping = AdaptiveEarlyStopping(patience=50)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(Epochs):
                train_loss = trainer._train_epoch(train_loader)
                
                if val_loader is not None:
                    val_loss = trainer.validate(val_loader)
                    print(f'Epoch {epoch+1}: Train Loss = {train_loss[0]:.4f}, Val Loss = {val_loss:.4f}')
                    current_lr = lr_scheduler.step(val_loss)
                    if early_stopping(val_loss):
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), f'best_model_{DATA}.pt')
                    else:
                        patience_counter += 1
                else:
                    print(f'Epoch {epoch+1}: Train Loss = {train_loss[0]:.4f}')
            
            if val_loader is not None and os.path.exists(f'best_model_{DATA}.pt'):
                model.load_state_dict(torch.load(f'best_model_{DATA}.pt'))
                print("Loaded best model based on validation loss")
            
            torch.save(model.state_dict(), './models/' + model_name)
            TEST = True

        if TEST == True:
            path_to_model_folder = './models/' + model_name
            batch_size_test = 140000
            if Net == 'C':
                train_loader, _, view_num, n_clusters, _, _ = Get_dataloaders(
                    batch_size=batch_size_test,
                    DATANAME=DATA + '.mat',
                    val_ratio=0.0
                )
            model = load(latent_spec=latent_spec,
                         path=path_to_model_folder,
                         view_num=view_num,
                         img_size=img_size,
                         Network=Net,
                         hid=hidden_dim, 
                         shareAE=share,
                         use_cuda=use_cuda,
                         num_heads=num_heads)

            device = torch.device("cuda" if use_cuda else "cpu")
            model.to(device)
            model_device = next(model.parameters()).device
            print(f"Model device: {model_device}")

            print(model.MvLatent_spec)
            print(model)

            for batch_idx, Data in enumerate(train_loader):
                break
            
            data_list = list(Data)
            data_views = [d.to(device) for d in data_list[:view_num]]
            labels = data_list[view_num].to(device)
            masks = [m.to(device) for m in data_list[view_num+1:view_num+1+view_num]]
            
            inputs = []
            for i in range(view_num):
                inputs.append(data_views[i])            
           
            encodings, _ = model.encode(inputs, masks)
            
            kmeans = KMeans(n_clusters=n_clusters, n_init=100)
            x = encodings['disc'][0].cpu().detach().data.numpy()
            multiview_z = []
            multiview_cz = []
            for i in range(view_num):
                name = 'cont' + str(i + 1)
                x_c = encodings[name][0].cpu().detach().data.numpy()
                xi = min_max_scaler.fit_transform(x_c)
                multiview_z.append(np.concatenate([xi, x], axis=1))
                multiview_cz.append(xi)
                print(multiview_z[-1].shape)
                print(multiview_z[-1][0])
            y = labels.cpu().detach().data.numpy()

            p = kmeans.fit_predict(x)            
            print(x.shape)
            Nmetrics.test(y, p)
            p = x.argmax(1)           
            Nmetrics.test(y, p)
            
            ACCc = Nmetrics.acc(y, p)
            NMIc = Nmetrics.nmi(y, p)
            ARIc = Nmetrics.ari(y, p)
            PURc = Nmetrics.purity(y, p)

            X_all = np.concatenate(multiview_cz, axis=1)
            p = kmeans.fit_predict(X_all)            
            print(X_all.shape)
            Nmetrics.test(y, p)
            print('k-means on [zv]\nk-means on [C, zv]')
            print(multiview_cz[0].shape, multiview_z[0].shape)
            for i in range(view_num):
                name = 'cont' + str(i + 1)
                x_cz = encodings[name][0].cpu().detach().data.numpy()
                x_Conz = multiview_z[i]
                p = kmeans.fit_predict(x_cz)
                Nmetrics.test(y, p)
                p = kmeans.fit_predict(x_Conz)
                Nmetrics.test(y, p)
                print('\n')

            multiview_cz.append(x)
            X_all = np.concatenate(multiview_cz, axis=1)
            p = kmeans.fit_predict(X_all)
            
            Nmetrics.test(y, p)
            ACCcz = Nmetrics.acc(y, p)
            NMIcz = Nmetrics.nmi(y, p)
            ARIcz = Nmetrics.ari(y, p)
            PURcz = Nmetrics.purity(y, p)

            
            print(f"Multi-VAE-final:\n"
            f"ACC:{ACCcz:.4f}, NMI:{NMIcz:.4f}, VME:{NMIcz:.4f}, ARI:{ARIcz:.4f}, PUR:{PURcz:.4f}")