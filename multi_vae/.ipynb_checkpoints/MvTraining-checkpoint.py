import os
import imageio
import numpy as np
import torch
from torch.nn import functional as F

EPS = 1e-12

class Trainer():
    def __init__(self, model, optimizer, cont_capacity=None,
                 disc_capacity=None, print_loss_every=50, record_loss_every=100,
                 use_cuda=False, view_num=2, DATA='DATA',
                 weight_decay=1e-5, max_grad_norm=1.0,
                 mi_weight=0.001):  
        self.model = model
        self.optimizer = optimizer 
        self.cont_capacity = cont_capacity
        self.disc_capacity = disc_capacity
        self.print_loss_every = print_loss_every
        self.record_loss_every = record_loss_every
        self.use_cuda = use_cuda
        self.view_num = view_num
        self.DATA = DATA
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.mi_weight = mi_weight

        if self.model.is_continuous and self.cont_capacity is None:
            raise RuntimeError("Model is continuous but cont_capacity not provided.")
        if self.model.is_discrete and self.disc_capacity is None:
            raise RuntimeError("Model is discrete but disc_capacity not provided.")

        if self.use_cuda:
            self.model.cuda()

        # VAE optimizer
        vae_params = [p for n, p in model.named_parameters() if 'mi_estimators' not in n]
        self.vae_optimizer = torch.optim.Adam(
            vae_params,
            lr=optimizer.param_groups[0]['lr'],
            weight_decay=weight_decay
        )

        # Statistics network optimizer
        stat_params = [p for n, p in model.named_parameters() if 'mi_estimators' in n]
        if len(stat_params) > 0:
            self.stat_optimizer = torch.optim.Adam(
                stat_params,
                lr=optimizer.param_groups[0]['lr'], 
                weight_decay=weight_decay
            )
        else:
            self.stat_optimizer = None
           

        self.num_steps = 0
        self.beta = [1.0] * self.view_num
        self.batch_size = None
        self.losses = {'loss': [],
                       'recon_loss': [],
                       'kl_loss': []}
        self.loss_r = [[] for _ in range(self.view_num)]
        self.loss_z = [[] for _ in range(self.view_num)]
        self.loss_c = [[] for _ in range(self.view_num)]
        self.mean_loss = [[] for _ in range(self.view_num)]

        if self.model.is_continuous:
            self.losses['kl_loss_cont'] = []
            for i in range(self.model.latent_spec['cont']):
                self.losses['kl_loss_cont_' + str(i)] = []

        if self.model.is_discrete:
            self.losses['kl_loss_disc'] = []
            for i in range(len(self.model.latent_spec['disc'])):
                self.losses['kl_loss_disc_' + str(i)] = []

  
    def _detach_latent_dist(self, latent_dist):
        """Detach all tensors in latent_dist to freeze VAE when updating statistics network."""
        detached = {}
        for k, v in latent_dist.items():
            if k == 'disc':
                detached[k] = [alpha.detach() for alpha in v]
            else:  
                detached[k] = [v[0].detach(), v[1].detach()]
        return detached

   
    def validate(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, Data in enumerate(data_loader):
                loss = self._compute_loss(Data)
                total_loss += loss
        return total_loss / len(data_loader.dataset)

    def _compute_loss(self, Data):
        all_data = list(Data)
        labels = all_data[self.view_num]
        masks = all_data[self.view_num+1:self.view_num+1+self.view_num]
        X = all_data[:self.view_num]
        device = next(self.model.parameters()).device
        for i in range(self.view_num):
            X[i] = X[i].to(device)
            masks[i] = masks[i].to(device)

        listout, latent_dist, hiddens = self.model(X, masks)
        recon_batchs = []
        for i in range(self.view_num):
            recon_batchs.append(listout[i] if listout[i] is not None else None)

        Loss = []
        max_recon_loss = 0
        for i in range(self.view_num):
            if masks[i] is not None and masks[i].sum() > 0:
                if recon_batchs[i] is not None:
                    target = X[i].view(-1, self.model.num_pixels[i])
                    recon = recon_batchs[i].view(-1, self.model.num_pixels[i])
                    view_loss = F.binary_cross_entropy(recon, target, reduction='none')
                    view_loss = view_loss.mean(dim=1)
                    valid_samples = masks[i]
                    view_loss = (view_loss * valid_samples).sum() / valid_samples.sum().clamp(min=1)
                    view_loss *= self.model.num_pixels[i]
                    if view_loss > max_recon_loss:
                        max_recon_loss = view_loss
                else:
                    view_loss = torch.tensor(0.0, device=device)
            else:
                view_loss = torch.tensor(0.0, device=device)

            total_view_loss = self._add_kl_loss(view_loss, i, latent_dist, max_recon_loss)
            Loss.append(total_view_loss)

        mi_loss = self.model.compute_mutual_info_loss(latent_dist)
        mi_loss = self.mi_weight * mi_loss

        valid_losses = [l for l in Loss if l is not None and not torch.isclose(l, torch.tensor(0.0, dtype=l.dtype, device=l.device))]
        total_loss = torch.stack(valid_losses).sum() if valid_losses else torch.tensor(0.0, device=device)
        total_loss = total_loss + mi_loss
        return total_loss.item()

   
    def train(self, data_loader, epochs=10, save_training_gif=None, val_loader=None):
        if save_training_gif is not None:
            training_progress_images = []

        self.batch_size = data_loader.batch_size
        self.model.train()

        patience = 50
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            mean_epoch_loss = self._train_epoch(data_loader)
            for i in range(self.view_num):
                print('Epoch: {}'.format(epoch + 1) + '. Average loss view-' + str(i+1) + ': {:.2f}'.format(
                    self.batch_size * self.model.num_pixels[i] * mean_epoch_loss[i]))
                self.mean_loss[i].append(self.batch_size * self.model.num_pixels[i] * mean_epoch_loss[i])

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f'Epoch {epoch+1}: Val Loss: {val_loss:.4f}')
                if val_loss < best_val_loss - 0.001:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), f'best_model_{self.DATA}.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            if save_training_gif is not None:
                viz = save_training_gif[1]
                viz.save_images = False
                img_grid = viz.all_latent_traversals(size=10)
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                training_progress_images.append(img_grid)

        if val_loader is not None and os.path.exists(f'best_model_{self.DATA}.pt'):
            self.model.load_state_dict(torch.load(f'best_model_{self.DATA}.pt'))
            print("Loaded best model based on validation loss")

        np.save('./mean_loss.npy', self.mean_loss)

        if save_training_gif is not None:
            imageio.mimsave(save_training_gif[0], training_progress_images, fps=24)

    def _train_epoch(self, data_loader):
        self.model.train()
        epoch_loss = [0.] * self.view_num

        for batch_idx, Data in enumerate(data_loader):
            iter_loss = self._train_iteration(Data)
            for i in range(self.view_num):
                epoch_loss[i] += iter_loss[i]

        for i in range(self.view_num):
            epoch_loss[i] /= len(data_loader.dataset)
        return epoch_loss

    def _add_kl_loss(self, recon_loss, view_idx, latent_dist, max_recon_loss):
        """Compute ELBO loss for a single view (reconstruction + KL + capacity)."""
        if recon_loss == 0:
            return torch.tensor(0.0, device=recon_loss.device, dtype=recon_loss.dtype)

        kl_cont_loss = 0
        kl_disc_loss = 0
        cont_capacity_loss = 0
        disc_capacity_loss = 0

        cont_max, cont_gamma, iters_add_cont_max = self.cont_capacity
        disc_max, disc_gamma, iters_add_disc_max = self.disc_capacity

        step = cont_max / iters_add_cont_max

        if self.model.is_continuous:
            countinus_name = 'cont' + str(view_idx+1)
            if countinus_name in latent_dist:
                mean, logvar = latent_dist[countinus_name]
                kl_cont_loss = self._kl_normal_loss(mean, logvar)
                cont_cap_current = step * self.num_steps
                cont_cap_current = min(cont_cap_current, cont_max)
                if self.DATA in ['Object-Digit-Product']:
                    cont_gamma = cont_gamma * self.beta[view_idx]
                cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_cont_loss)

        if self.model.is_discrete and view_idx == 0 and 'disc' in latent_dist:
            kl_disc_loss = self._kl_multiple_discrete_loss(latent_dist['disc'])
            disc_cap_current = step * self.num_steps
            disc_cap_current = min(disc_cap_current, disc_max)
            if self.DATA in ['Object-Digit-Product']:
                disc_gamma = disc_gamma * self.beta[view_idx]
            disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_disc_loss)

        kl_loss = kl_cont_loss + kl_disc_loss
        capacity_loss = cont_capacity_loss + disc_capacity_loss
        total_loss = recon_loss + capacity_loss + kl_loss

        if self.model.training and self.num_steps % self.record_loss_every == 0:
            self.losses['recon_loss'].append(recon_loss.item())
            self.losses['kl_loss'].append(kl_loss.item())
            self.losses['loss'].append(total_loss.item())

        self.loss_r[view_idx].append(recon_loss.item())
        self.loss_z[view_idx].append(kl_cont_loss.item() if self.model.is_continuous else 0)
        self.loss_c[view_idx].append(kl_disc_loss.item() if self.model.is_discrete and view_idx == 0 else 0)

        return total_loss / self.model.num_pixels[view_idx]

   
    def _train_iteration(self, data):
        self.num_steps += 1
        all_data = list(data)
        labels = all_data[self.view_num]
        masks = all_data[self.view_num+1:self.view_num+1+self.view_num]
        X = all_data[:self.view_num]

        device = next(self.model.parameters()).device
        for i in range(self.view_num):
            X[i] = X[i].to(device)
            masks[i] = masks[i].to(device)

        # Forward pass
        listout, latent_dist, hiddens = self.model(X, masks)
        recon_batchs = [out if out is not None else None for out in listout]

        # Compute ELBO loss (reconstruction + KL + capacity)
        ELBO_per_view = []
        max_recon_loss = 0
        for i in range(self.view_num):
            if masks[i] is not None and masks[i].sum() > 0 and recon_batchs[i] is not None:
                target = X[i].view(-1, self.model.num_pixels[i])
                recon = recon_batchs[i].view(-1, self.model.num_pixels[i])
                view_loss = F.binary_cross_entropy(recon, target, reduction='none')
                view_loss = view_loss.mean(dim=1)
                view_loss = (view_loss * masks[i]).sum() / masks[i].sum().clamp(min=1)
                view_loss *= self.model.num_pixels[i]
                if view_loss > max_recon_loss:
                    max_recon_loss = view_loss
            else:
                view_loss = torch.tensor(0.0, device=device)

            elbo_view = self._add_kl_loss(view_loss, i, latent_dist, max_recon_loss)
            ELBO_per_view.append(elbo_view)

        ELBO_loss = torch.stack(ELBO_per_view).sum()

        # Update statistics network
        if self.stat_optimizer is not None:
            self.stat_optimizer.zero_grad()
            detached_latent = self._detach_latent_dist(latent_dist)
            mi_loss_stat = self.model.compute_mutual_info_loss(detached_latent)  # returns -MI
            mi_loss_stat.backward()
            torch.nn.utils.clip_grad_norm_(self.model.mi_estimators.parameters(), self.max_grad_norm)
            self.stat_optimizer.step()   

            # Freeze statistics network parameters for VAE update step
            for p in self.model.mi_estimators.parameters():
                p.requires_grad_(False)

        # Update VAE (minimize ELBO + λ * (-MI))
        mi_loss_vae = self.model.compute_mutual_info_loss(latent_dist)  

        self.vae_optimizer.zero_grad()
        total_vae_loss = ELBO_loss + self.mi_weight * mi_loss_vae
        total_vae_loss.backward()  

        # Clip gradients only for VAE parameters (excluding statistics network)
        vae_params = [p for n, p in self.model.named_parameters() if 'mi_estimators' not in n]
        torch.nn.utils.clip_grad_norm_(vae_params, self.max_grad_norm)
        self.vae_optimizer.step()

        # Re-enable gradients for statistics network for the next iteration
        if self.stat_optimizer is not None:
            for p in self.model.mi_estimators.parameters():
                p.requires_grad_(True)

        # Record loss
        train_loss = [l.item() if isinstance(l, torch.Tensor) else 0.0 for l in ELBO_per_view]
        return train_loss
     
    def _kl_normal_loss(self, mean, logvar):
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        kl_means = torch.mean(kl_values, dim=0)
        kl_loss = torch.sum(kl_means)

        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['kl_loss_cont'].append(kl_loss.item())
            for i in range(self.model.latent_spec['cont']):
                self.losses['kl_loss_cont_' + str(i)].append(kl_means[i].item())

        return kl_loss

    def _kl_multiple_discrete_loss(self, alphas):
        kl_losses = [self._kl_discrete_loss(alpha) for alpha in alphas]
        kl_loss = torch.sum(torch.cat(kl_losses))

        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['kl_loss_disc'].append(kl_loss.item())
            for i in range(len(alphas)):
                self.losses['kl_loss_disc_' + str(i)].append(kl_losses[i].item())

        return kl_loss

    def _kl_discrete_loss(self, alpha):
        disc_dim = int(alpha.size()[-1])
        log_dim = torch.tensor([np.log(disc_dim)], device=alpha.device, dtype=alpha.dtype)
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)
        kl_loss = log_dim + mean_neg_entropy
        return kl_loss