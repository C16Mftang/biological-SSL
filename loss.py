import numpy as np
import torch
from torch import nn

class HingeLoss(nn.Module):
    def __init__(self, device='cpu'):
        super(HingeLoss, self).__init__()
        self.device = device

    def forward(self, output, target):
        one_hot_mask = torch.zeros(len(target), output.shape[1]).to(self.device)
        one_hot_mask.scatter_(1, target.unsqueeze(1), 1.)
        loss = torch.sum((1-torch.diag(output[:,target])).clamp(min=0))+torch.sum(((1+output)*(1-one_hot_mask)).clamp(min=0))/10
        return loss/target.shape[0]

# # Take mean among nagative cases
# class ContrastiveHinge(torch.nn.Module):
#     def __init__(self, batch_size, threshold = -2., device='cpu'):
#         super(ContrastiveHinge, self).__init__()
#         self.device = device
#         self.batch_size = batch_size
#         self.threshold = threshold
#         self.mask = self.create_mask(batch_size)
#         self.bmask = self.create_bmask(batch_size)

#     # create a mask that enables us to sum over positive pairs only
#     def create_mask(self, batch_size):
#         mask = 2.*torch.eye(batch_size).to(self.device)-1.
#         return mask

#     def create_bmask(self, batch_size):
#         bmask = torch.eye(batch_size, dtype=torch.bool).to(self.device)
#         return bmask

#     def forward(self, output):
#         # norm = torch.nn.functional.normalize(output, dim=1)
#         #print(norm.shape)
#         norm = output
#         h1,h2 = torch.split(norm, self.batch_size)

#         h1b = h1.repeat([self.batch_size, 1])
#         h2b = h2.repeat_interleave(self.batch_size, dim=0)
#         outd = h1b - h2b
#         # 1-norm
#         outd = torch.sum(torch.abs(outd), dim=1)
#         # distance matrix, flipped and negated
#         OUT = -outd.reshape(self.batch_size, self.batch_size).transpose(0, 1)
#         # Multiply by y=-1/1; 2*eye-1 is a matrix with 1 in the diagonal, -1 off diagonal
#         OUT = (OUT-self.threshold)*self.mask
#         OUT = torch.relu(1-OUT)
#         loss = torch.mean(torch.diag(OUT))+torch.mean(OUT[~self.bmask])
#         return loss

class GazeHingeNN(torch.nn.Module):
    def __init__(self, pars):
        super(GazeHingeNN, self).__init__()
        self.device = pars.device
        self.batch_size = pars.batch_size
        self.thr1 = pars.thr1
        self.thr2 = pars.thr2
        self.n_patches = pars.n_patches
        self.n_negs = pars.n_negs

    def forward(self, output):
        all_embedd = output.reshape(-1, self.n_patches, self.n_patches, output.shape[-1]) # b, 7, 7, 64
        loss = 0
        for k in range(self.n_patches-2):
            anchor = all_embedd[:,k:k+1,:,:] # bsz, 1, 7, 64
            positives = all_embedd[:,k+2:self.n_patches,:,:] # bsz, 5, 7, 64
            pos_dists = torch.sum(torch.relu(anchor-positives)+torch.relu(positives-anchor), dim=-1) # bsz, 5, 7
            # sample with replacement
            rand_idx = torch.randint(output.size(0), (self.n_negs * anchor.shape[0] * self.n_patches * (self.n_patches-k-2),))
            negatives = output[rand_idx].reshape(-1, (self.n_patches-k-2)*self.n_negs, self.n_patches, output.shape[-1]) # bsz, 5*n_neg, 7, 64
            neg_dists = torch.sum(torch.relu(anchor-negatives)+torch.relu(negatives-anchor), dim=-1) # bsz, 5*n_neg, 7
            # neg_dists = torch.mean(neg_dists.reshape(-1, (self.n_patches-k-2), self.n_negs, self.n_patches), dim=2) # average over n negatives
            loss += torch.sum(torch.relu(pos_dists - self.thr1)) + torch.sum(torch.relu(self.thr2 - neg_dists))
        return loss

class CLAPPHinge(torch.nn.Module):
    def __init__(self, pars, n_features):
        super(CLAPPHinge, self).__init__()
        self.device = pars.device
        self.batch_size = pars.batch_size
        self.n_patches = pars.n_patches
        self.n_negs = pars.n_negs
        self.pars = pars
        self.W_pred = nn.ModuleList(
            nn.Conv2d(n_features, n_features, 1, bias=False) # in_channels: z, out_channels: c
            for _ in range(self.n_patches-2)
        ).to(self.pars.device)

    def forward(self, output):
        n_features, hw = output.shape[-3], output.shape[-2]
        all_embedd = output.reshape(-1, self.n_patches, self.n_patches, n_features, hw, hw) # b, 7, 7, n_features, hw, hw
        loss = 0
        bsz = all_embedd.shape[0]
        for k in range(self.n_patches-2):
            # Positives
            context = all_embedd[:,k:k+1,:,:,:,:].reshape(-1, n_features, hw, hw) # b*7, n_features, hw, hw
            Wc = self.W_pred[k](context) # b*7, n_features, hw, hw
            Wc = Wc.flatten(start_dim=1).unsqueeze(-2) # b*7, 1, n_features*hw*hw
            positives = all_embedd[:,k+2:self.n_patches,:,:,:,:].flatten(start_dim=-3) # b, 5, 7, n_features*hw*hw
            positives = positives.permute(0,2,3,1).reshape(-1, n_features*hw*hw, self.n_patches-k-2) # b*7, n_features*hw*hw, 5
            # compute z_pWc
            score_pos = torch.matmul(Wc, positives).squeeze() # b*7, 5
            score_pos = torch.relu(1 - 1 * score_pos)

            # Negative
            rand_idx = torch.randint(output.shape[0], (bsz * self.n_negs * self.n_patches * (self.n_patches-k-2),))
            negatives = output[rand_idx].squeeze().reshape(-1, self.n_negs*(self.n_patches-k-2), self.n_patches, n_features, hw, hw) # b, 5*n_neg, 7, n_features, hw, hw
            negatives = negatives.flatten(start_dim=-3) # b, 5*n_neg, 7, n_features*hw*hw
            negatives = negatives.permute(0,2,3,1).reshape(-1, n_features*hw*hw, self.n_negs*(self.n_patches-k-2)) # b*7, n_features, 5*n_neg
            # compute z_nWc
            score_neg = torch.matmul(Wc, negatives).squeeze() # b*7, 5*n_neg
            score_neg = torch.relu(1 - (-1) * score_neg)

            loss += torch.mean(score_pos) + torch.mean(score_neg)
        return loss

class ContrastiveHinge(torch.nn.Module):
    def __init__(self, batch_size, thr=2., delta=1., device='cpu'):
        super(ContrastiveHinge, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.thr = thr
        self.delta = delta

    def forward(self, output):
        n = output.shape[0] # this is 4*batch_size if using gaze shift
        output = torch.nn.functional.normalize(output, dim=1)
        out0, out1 = torch.split(output, n//2)
        # out0 = out0.clone().detach() # grad block
        out0b = out0.repeat([n//2, 1])
        out1b = out1.repeat_interleave(n//2, dim=0)
        outd = out0b - out1b
        outd = torch.sum(torch.relu(outd) + torch.relu(-outd), dim=1)
        OUT = -outd.reshape(n//2, n//2).transpose(0, 1)
     
        # Multiply by y=-1/1
        OUT = (OUT + self.thr) * (2. * torch.eye(n//2).to(self.device) - 1.)
        #print('mid',time.time()-t1)
        loss = torch.sum(torch.relu(self.delta - OUT))
        return loss

class ContrastiveHingeNN(torch.nn.Module):
    def __init__(self, batch_size, thr=2., delta=1., device='cpu'):
        super(ContrastiveHingeNN, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.thr = thr
        self.delta = delta

    def forward(self, output):
        n = output.shape[0]
        out0, out1 = torch.split(output, n//2)
        out0 = out0.clone().detach() # grad block
        out0b = out0.repeat([n//2, 1])
        out1b = out1.repeat_interleave(n//2, dim=0)
        outd = out0b - out1b
        outd = torch.sum(torch.relu(outd) + torch.relu(-outd), dim=1)
        OUT = -outd.reshape(n//2, n//2).transpose(0, 1)
     
        # Multiply by y=-1/1
        OUT = (OUT + self.thr) * (2. * torch.eye(n//2).to(self.device) - 1.)
        #print('mid',time.time()-t1)
        loss = torch.sum(torch.relu(self.delta - OUT))
        return loss
        
class ContrastiveHingeNN2(torch.nn.Module):
    def __init__(self, batch_size, thr1=1., thr2=3., device='cpu'):
        super(ContrastiveHingeNN2, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.thr1 = thr1
        self.thr2 = thr2
        self.bmask = self.create_bmask(batch_size)

    def create_bmask(self, batch_size):
        bmask = torch.eye(batch_size, dtype=torch.bool).to(self.device)
        return bmask

    def forward(self, output):
        #norm = torch.nn.functional.normalize(output, dim=1)
        #print(norm.shape)
        # norm = torch.nn.functional.tanh(output)
        norm = output
        h1,h2 = torch.split(norm, self.batch_size)
        h1b = h1.repeat([self.batch_size, 1]) # bsz*bsz, 64
        h2b = h2.repeat_interleave(self.batch_size, dim=0) # bsz*bsz, 64
        outd = h1b - h2b # bsz*bsz, 64
        # 1-norm
        outd = torch.sum(torch.abs(outd), dim=1) # bsz*bsz, 1
        # distance matrix, flipped 
        OUT = outd.reshape(self.batch_size, self.batch_size).transpose(0, 1) # bsz, bsz
        # Multiply by y=-1/1; 2*eye-1 is a matrix with 1 in the diagonal, -1 off diagonal
        pos = torch.diag(OUT) # bsz,
        neg = OUT[~self.bmask] # bsz*(bsz-1),
        loss = torch.sum(torch.relu(pos-self.thr1))+torch.sum(torch.relu(self.thr2-neg))
        return loss

class HingeNNFewerNegs(torch.nn.Module):
    def __init__(self, batch_size, thr=2., delta=1., future=5, device='cpu'):
        super(HingeNNFewerNegs, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.thr = thr
        self.delta = delta
        self.future = future
        self.mask = self.create_bmask(batch_size)

    def create_bmask(self, batch_size):
        mask = torch.eye(batch_size).to(self.device)
        mask = 2. * mask - 1.
        return mask

    def forward(self, output):
        n = output.shape[0]
        out0, out1 = torch.split(output, n//2)
        out0 = out0.clone().detach() # grad block
        out0b = out0.repeat([n//2, 1])
        out1b = out1.repeat_interleave(n//2, dim=0)
        outd = out0b - out1b
        outd = torch.sum(torch.relu(outd) + torch.relu(-outd), dim=1)
        OUT = -outd.reshape(n//2, n//2).transpose(0, 1)
        # Multiply by y=-1/1
        OUT = (OUT + self.thr) * self.mask

        if self.future != 0:
            loss = 0
            for i in range(self.future):
                fac = 1. if i==0 else 1./self.future
                loss += fac*(torch.sum(torch.relu(self.delta - torch.diagonal(OUT,i))))
        else:
            loss = torch.sum(torch.relu(self.delta - OUT))

        return loss

class SimCLRLoss(torch.nn.Module):
    def __init__(self, batch_size, device='cpu'):
        super(SimCLRLoss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.mask = self.create_mask(batch_size)
        self.criterion = torch.nn.CrossEntropyLoss()

    # create a mask that enables us to sum over positive pairs only
    def create_mask(self, batch_size):
        mask = torch.eye(batch_size, dtype=torch.bool).to(self.device)
        return mask

    def forward(self, output, tau=0.1):
        norm = torch.nn.functional.normalize(output, dim=1)
        h1,h2 = torch.split(norm, self.batch_size)

        aa = torch.mm(h1,h1.transpose(0,1))/tau
        aa_s = aa[~self.mask].view(aa.shape[0],-1)
        bb = torch.mm(h2,h2.transpose(0,1))/tau
        bb_s = bb[~self.mask].view(bb.shape[0],-1)
        ab = torch.mm(h1,h2.transpose(0,1))/tau
        ba = torch.mm(h2,h1.transpose(0,1))/tau

        labels = torch.arange(self.batch_size).to(output.device)
        loss_a = self.criterion(torch.cat([ab,aa_s],dim=1),labels)
        loss_b = self.criterion(torch.cat([ba,bb_s],dim=1),labels)

        loss = loss_a+loss_b
        return loss