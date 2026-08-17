import torch
from torch import nn
from torch.nn import functional as F
EPS = 1e-12

# Enhanced multi-head attention mechanism
class EnhancedMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, use_residual=True):
        super(EnhancedMultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_residual = use_residual
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        batch_size, seq_len, _ = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(x.device)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout_layer(attention)
        out = torch.matmul(attention, V)
        
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, self.hidden_dim)
        
        if self.use_residual:
            out = out + residual
        out = self.norm2(out)
        
        ff_out = self.feed_forward(out)
        if self.use_residual:
            ff_out = ff_out + out
            
        out = self.out(ff_out)
        return out, attention

# Mutual information estimator
class MIEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, y):
        joint_input = torch.cat([x, y], dim=1)
        joint_output = self.net(joint_input)
        shuffled_y = y[torch.randperm(y.size(0))]
        marginal_input = torch.cat([x, shuffled_y], dim=1)
        marginal_output = self.net(marginal_input)
        mi_est = joint_output.mean() - torch.log(marginal_output.exp().mean() + 1e-8)
        return mi_est

class VAE(nn.Module):
    def __init__(self, img_size, latent_spec, temperature=0.67, 
                 use_cuda=False, view_num=2, Network='C',
                 hidden_dim=256, shareAE=1, num_heads=4, dropout=0.2):
        super(VAE, self).__init__()
        self.use_cuda = use_cuda
        self.img_size = img_size
        self.is_continuous = 'cont' in latent_spec
        self.is_discrete = 'disc' in latent_spec
        self.latent_spec = latent_spec
        self.view_num = view_num
        self.MvLatent_spec = {'disc': latent_spec['disc']}
        for i in range(self.view_num):
            continue_name = 'cont' + str(i + 1)
            self.MvLatent_spec[continue_name] = latent_spec['cont']
        
        self.Net = Network
        self.share = shareAE
        if self.Net == 'C':
            self.num_pixels = []
            for i in range(self.view_num):
                self.num_pixels.append(img_size[1] * img_size[2])
            
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.reshape = (64, 4, 4)
        self.latent_cont_dim = 0
        self.latent_disc_dim = 0
        self.num_disc_latents = 0
        if self.is_continuous:
            self.latent_cont_dim = self.latent_spec['cont']
        if self.is_discrete:
            self.latent_disc_dim += sum([dim for dim in self.latent_spec['disc']])
            self.num_disc_latents = len(self.latent_spec['disc'])
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        
        if self.Net == 'C':
            Mv_img_to_features = []
            Mv_features_to_hidden = []
            Mv_latent_to_features = []
            Mv_features_to_img = []
            for i in range(self.view_num):
                encoder_layers = [
                    nn.Conv2d(self.img_size[0], 32, (4, 4), stride=2, padding=1),
                    nn.ReLU(),
                    self.dropout
                ]
                if self.img_size[1:] == (64, 64):
                    encoder_layers += [
                        nn.Conv2d(32, 32, (4, 4), stride=2, padding=1),
                        nn.ReLU(),
                        self.dropout
                    ]
                elif self.img_size[1:] == (32, 32):
                    pass
                else:
                    raise RuntimeError(f"{img_size} sized images not supported")
                
                encoder_layers += [
                    nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
                    nn.ReLU(),
                    self.dropout,
                    nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
                    nn.ReLU(),
                    self.dropout
                ]
                Mv_img_to_features.append(nn.Sequential(*encoder_layers))
                
                features_to_hidden = nn.Sequential(
                    nn.Linear(64 * 4 * 4, self.hidden_dim),
                    nn.ReLU(),
                    self.dropout
                )
                Mv_features_to_hidden.append(features_to_hidden)

            if self.share:
                self.Mv_img_to_features = nn.ModuleList([Mv_img_to_features[0]])
                self.Mv_features_to_hidden = nn.ModuleList([Mv_features_to_hidden[0]])
            else:
                self.Mv_img_to_features = nn.ModuleList(Mv_img_to_features)
                self.Mv_features_to_hidden = nn.ModuleList(Mv_features_to_hidden)
                
            means = []
            log_vars = []
            if self.is_continuous:
                for i in range(self.view_num):
                    means.append(nn.Linear(self.hidden_dim, self.latent_cont_dim))
                    log_vars.append(nn.Linear(self.hidden_dim, self.latent_cont_dim))
            self.means = nn.ModuleList(means)
            self.log_vars = nn.ModuleList(log_vars)
            
            if self.is_discrete:
                fc_alphas = []
                for disc_dim in self.latent_spec['disc']:
                    fc_alphas.append(nn.Linear(self.hidden_dim, disc_dim))
                self.fc_alphas = nn.ModuleList(fc_alphas)
            
            # Multi-head attention
            self.enhanced_multihead_attention = EnhancedMultiHeadAttention(
                hidden_dim=self.hidden_dim, 
                num_heads=num_heads,
                dropout=dropout,
                use_residual=True
            )

            # Gated fusion
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

            for i in range(self.view_num):
                latent_to_features = nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden_dim),
                    nn.ReLU(),
                    self.dropout,
                    nn.Linear(self.hidden_dim, 64 * 4 * 4),
                    nn.ReLU(),
                    self.dropout
                )
                Mv_latent_to_features.append(latent_to_features)
                
                decoder_layers = []
                if self.img_size[1:] == (64, 64):
                    decoder_layers += [
                        nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=1),
                        nn.ReLU(),
                        self.dropout
                    ]

                decoder_layers += [
                    nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
                    nn.ReLU(),
                    self.dropout,
                    nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1),
                    nn.ReLU(),
                    self.dropout,
                    nn.ConvTranspose2d(32, self.img_size[0], (4, 4), stride=2, padding=1),
                    nn.Sigmoid()
                ]
                Mv_features_to_img.append(nn.Sequential(*decoder_layers))

            if self.share:
                self.Mv_latent_to_features = nn.ModuleList([Mv_latent_to_features[0]])
                self.Mv_features_to_img = nn.ModuleList([Mv_features_to_img[0]])
            else:
                self.Mv_latent_to_features = nn.ModuleList(Mv_latent_to_features)
                self.Mv_features_to_img = nn.ModuleList(Mv_features_to_img)
        
        # Mutual information estimators
        self.mi_estimators = nn.ModuleList()
        for i in range(view_num):
            mi_input_dim = latent_spec['disc'][0] + latent_spec['cont']
            self.mi_estimators.append(MIEstimator(mi_input_dim, hidden_dim//2))

    def encode(self, X, masks):
        batch_size = X[0].size()[0]
        features = []
        hiddens = []
        valid_hiddens = []
        device = next(self.parameters()).device
        
        for i in range(self.view_num):
            # For complete views, masks are all 1, so this branch is always active
            if masks[i] is not None and masks[i].sum() > 0:
                if self.share:
                    net_num = 0
                else:
                    net_num = i
                    
                X[i] = X[i].to(device)
                    
                features.append(self.Mv_img_to_features[net_num](X[i]))
                hidden = self.Mv_features_to_hidden[net_num](features[i].view(batch_size, -1))
                hiddens.append(hidden)
                valid_hiddens.append(hidden)
            else:
                # This branch should not be reached for complete views (kept as safety net)
                hiddens.append(None)
                features.append(None)
        
        # Fuse all valid views
        fusion = self.enhanced_fusion(hiddens, masks)
        
        latent_dist = {}
        if self.is_continuous:
            for i in range(self.view_num):
                continue_name = 'cont' + str(i+1)
                if hiddens[i] is not None:
                    latent_dist[continue_name] = [self.means[i](hiddens[i]), self.log_vars[i](hiddens[i])]
                else:
                    # Safety net: use zero mean and unit variance if view is missing (should not happen)
                    zero_mean = torch.zeros(batch_size, self.latent_cont_dim, device=device)
                    unit_logvar = torch.zeros(batch_size, self.latent_cont_dim, device=device)
                    latent_dist[continue_name] = [zero_mean, unit_logvar]
        
        if self.is_discrete:
            latent_dist['disc'] = []
            for fc_alpha in self.fc_alphas:
                latent_dist['disc'].append(F.softmax(fc_alpha(fusion), dim=1))
        
        return latent_dist, hiddens

    def enhanced_fusion(self, hiddens, masks):
        device = next(self.parameters()).device
        batch_size = hiddens[0].size(0) if hiddens[0] is not None else 0
        
        valid_hiddens = []
        for h in hiddens:
            if h is not None:
                valid_hiddens.append(h)
        
        if len(valid_hiddens) == 0:
            fusion = torch.zeros(batch_size, self.hidden_dim, device=device)
            noise = torch.randn_like(fusion) * 0.1
            return fusion + noise
        
        attn_input = torch.stack(valid_hiddens, dim=1)  # [B, V, H]
        attn_mask = torch.ones(batch_size, self.view_num, device=device)
        for i, h in enumerate(hiddens):
            if h is None:
                attn_mask[:, i] = 0
        
        attn_output, _ = self.enhanced_multihead_attention(attn_input, attn_mask.unsqueeze(1).unsqueeze(2))
        
        if len(valid_hiddens) > 1:
            weights = []
            for h in valid_hiddens:
                weight = torch.sigmoid(self.fusion_gate(h)).squeeze()
                weights.append(weight)
            weights = torch.stack(weights, dim=1)
            weights = F.softmax(weights, dim=1)
            fusion = torch.sum(attn_output * weights.unsqueeze(2), dim=1)
        else:
            fusion = attn_output.squeeze(1)
        
        return fusion

    def reparameterize(self, latent_dist):
        latent_sample = []
        device = next(self.parameters()).device
        
        if self.is_continuous:
            for i in range(self.view_num):
                countinus_name = 'cont' + str(i+1)
                if countinus_name in latent_dist:
                    mean, logvar = latent_dist[countinus_name]
                    mean = mean.to(device)
                    logvar = logvar.to(device)
                    cont_sample = self.sample_normal(mean, logvar)
                    latent_sample.append(cont_sample)

        if self.is_discrete and 'disc' in latent_dist:
            for alpha in latent_dist['disc']:
                alpha = alpha.to(device)
                disc_sample = self.sample_gumbel_softmax(alpha)
                latent_sample.append(disc_sample)

        latent_sample = [sample.to(device) for sample in latent_sample if sample is not None]
        
        if latent_sample:
            return torch.cat(latent_sample, dim=1)
        else:
            return torch.empty(0, device=device)

    def sample_normal(self, mean, logvar):
        device = mean.device
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros_like(std).normal_()
            return mean + std * eps
        else:
            return mean

    def sample_gumbel_softmax(self, alpha):
        device = alpha.device
        if self.training:
            unif = torch.rand_like(alpha)
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros_like(alpha)
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1), 1)
            return one_hot_samples

    def decode(self, latent_samples, masks):
        features_to_img = []
        device = next(self.parameters()).device
        
        for i in range(self.view_num):
            if self.share:
                net_num = 0
            else:
                net_num = i
                
            if i < len(latent_samples) and latent_samples[i].numel() > 0:
                latent_input = latent_samples[i]
                feature = self.Mv_latent_to_features[net_num](latent_input)
            else:
                feature = torch.zeros(0, device=device)
                
            if self.Net == 'C' and feature.numel() > 0:
                recon = self.Mv_features_to_img[net_num](feature.view(-1, *self.reshape))
            else:
                recon = feature
            features_to_img.append(recon)
                
        return features_to_img

    def forward(self, X, masks):
        device = next(self.parameters()).device
        X = [x.to(device) for x in X]
        masks = [mask.to(device) for mask in masks]
        
        latent_dist, hiddens = self.encode(X, masks)
        latent_sample = self.reparameterize(latent_dist)
        
        if latent_sample.numel() == 0:
            return [None] * self.view_num, latent_dist, hiddens
        
        split_list = []
        for i in range(self.view_num):
            split_list.append(self.latent_spec['cont'])
        split_list.append(self.latent_spec['disc'][0])
        
        latent_parts = latent_sample.split(split_list, dim=1)
        decode_list = []
        shared_c = latent_parts[-1]
        
        for i in range(self.view_num):
            view_specific = latent_parts[i]
            view_latent = torch.cat([view_specific, shared_c], dim=1)
            decode_list.append(view_latent)
        
        out_list = self.decode(decode_list, masks)
        return out_list, latent_dist, hiddens
    
    def compute_mutual_info_loss(self, latent_dist):
        """
        Compute the mutual information loss between c and each z_i.
        """
        mi_loss = 0
        if 'disc' not in latent_dist or len(latent_dist['disc']) == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        c = latent_dist['disc'][0]
        
        for i in range(self.view_num):
            cont_name = 'cont' + str(i+1)
            if cont_name in latent_dist:
                z_i_mean, z_i_logvar = latent_dist[cont_name]
                z_i = self.sample_normal(z_i_mean, z_i_logvar)
                mi_est = self.mi_estimators[i](c, z_i)
                mi_loss = mi_loss - mi_est
        
        return mi_loss