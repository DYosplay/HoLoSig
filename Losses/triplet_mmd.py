import torch
import torch.nn.functional as F
import torch.nn as nn
from Losses import mmd_loss
import DTW.soft_dtw_cuda as soft_dtw
import numpy as np

class Triplet_MMD(nn.Module):
    def __init__(self, ng : int, nf : int, nr : int, nw : int, margin : float, random_margin : float, alpha : float, beta : float, p : float, r : float, mmd_kernel_num : float, mmd_kernel_mul : float, tau: float, s : float):
        """_summary_

        Args:
            ng (torch.nn.Parameter): number of genuine signatures inside the mini-batch
            nf (torch.nn.Parameter): number of forgeries signatures inside the mini-batch
            nw (torch.nn.Parameter): number of mini-batches (writers) inside a batch
            margin (torch.nn.Parameter): triplet loss margin
            model_lambda (torch.nn.Parameter): control intraclass variation
            alpha (torch.nn.Parameter): weighting factor for MMD
            p (torch.nn.Parameter): weighting factor for skilled (p) and random (1-p) forgeries
            q (torch.nn.Parameter): weighting factor for variance of genuines signatures inside the batch
            mmd_kernel_num (torch.nn.Parameter): number of kernels for MMD
            mmd_kernel_mul (torch.nn.Parameter): multipler for MMD
        """
        super(Triplet_MMD, self).__init__()
        # Hyperparameters
        self.ng = ng
        self.nf = nf
        self.nw = nw
        self.nr = nr
        self.margin = margin
        self.random_margin = random_margin
        self.sdtw = soft_dtw.SoftDTW(True, gamma=5, normalize=False, bandwidth=0.1)
        self.mmd_loss = mmd_loss.MMDLoss(kernel_num=mmd_kernel_num, kernel_mul=mmd_kernel_mul)
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.r = r
        self.s = s
        self.tau = tau


        self.siz = np.sum(np.array(list(range(1,self.nw+1))))

    def forward(self, data, lens):
        non_zero_random = 0

        step = (self.ng + self.nf + self.nr+ 1)
        total_loss = 0
        mmds = 0
        dists = torch.tensor([],device=data.device)

        for i in range(0, self.nw):
            anchor    = data[i * step]
            positives = data[i * step + 1 : i * step + 1 + self.ng]
            negatives = data[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf + self.nr]

            len_a = lens[i * step]
            len_p = lens[i * step + 1 : i * step + 1 + self.ng]
            len_n = lens[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf + self.nr]

            dist_g = torch.zeros((len(positives)), dtype=data.dtype, device=data.device)
            dist_n = torch.zeros((len(negatives)), dtype=data.dtype, device=data.device)
            
            for j in range(len(positives)):
                dist_g[j] = self.sdtw(anchor[None, :int(len_a)], positives[j:j+1, :int(len_p[j])])[0] / ((len_a**2 + len_p[j]**2)**0.5)
              
            for j in range(len(negatives)):
                dist_n[j] = self.sdtw(anchor[None, :int(len_a)], negatives[j:j+1, :int(len_n[j])])[0] / ((len_a**2 + len_n[j]**2)**0.5)
               
            dists = torch.cat((dists, dist_g, dist_n), dim=0)
            lk_skilled = F.relu(dist_g.unsqueeze(1) + self.margin - dist_n[:self.nf].unsqueeze(0))
            lk_random = F.relu(dist_g.unsqueeze(1) + self.random_margin - dist_n[self.nf:].unsqueeze(0))

            ca = torch.mean(dist_g)
            cb = torch.mean(dist_n[:self.nf])
            inter_loss = torch.sum(F.relu(self.beta - ((ca - cb).norm(dim=0, p=2))))

            lv = (torch.sum(lk_skilled) + torch.sum(lk_random)) / (lk_skilled.data.nonzero(as_tuple=False).size(0) + lk_random.data.nonzero(as_tuple=False).size(0) + 1)

            non_zero_random += lk_random.data.nonzero(as_tuple=False).size(0)

            user_loss = lv + inter_loss * self.r

            total_loss += user_loss

        total_loss /= self.nw

        ctr = 0
        mmds = torch.zeros((self.nw*(self.nw-1))//2, dtype=dists.dtype, device=dists.device)
        step -= 1 # desconsidera a âncora
        
        for i in range(0, self.nw):
            for j in range(i+1, self.nw):
                mmds[ctr] = self.mmd_loss(dists[step*i:step*(i+1)], dists[step*j: step*(j+1)]) #* self.alpha
                ctr+=1
        
        mmd1 = torch.max(mmds) * self.alpha

        return total_loss + mmd1, total_loss, mmd1, non_zero_random
