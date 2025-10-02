# HoLoSig: Holistic and Local Representation Learning for Online Signature Verification

HoLoSig uses offline rotation augmentation.  
The script for augmenting DeepSignDB signatures is located in `utils/rotate.py`.  
All reported results follow the official DeepSignDB experimental protocols.

The following directory structure is required:
- Data
    - DeepSignDB
- HoLoSig

## Training:

    $ python run_holosig.py -seed=555 -dropc=0.1 -dropr=0.1 -c -ce -a 0.3 -b 1.8 -p 0.809 -r 0.297 -lr 0.01 -dc 0.9 -ep=35 -stop=36 -es=3 -bs=64 -pf=<parent_folder> -t=<test_name> -nr=5 -nf=5 -ng=5 -nw=4 --cew=0.171 --epsilon=0.0918 -nhead=16 -hdim=32 -rot

## Evaluation on DeepSignDB:

    $ python run_holosig.py -seed=555 -dropc=0.1 -dropr=0.1 -c -ce -a 0.3 -b 1.8 -p 0.809 -r 0.297 -lr 0.01 -dc 0.9 -ep=35 -stop=36 -es=3 -bs=64 -pf=<parent_folder> -t=<test_name> -nr=5 -nf=5 -ng=5 -nw=4 --cew=0.171 --epsilon=0.0918 -nhead=16 -hdim=32 -rot -ev -w=best.pt

You can use any weight file `-w`, which can be found in the results folder at `<parent_folder>/<test_name>`.

## Results
### Separation and alignment errors (in %) for each DeepSignDB stylus scenario
<img width="1254" height="540" alt="image" src="https://github.com/user-attachments/assets/98eb96f7-cb0a-4a9a-b13a-1b4bf7b0cee4" />

### EER (in %) for each subdataset in DeepSignDB 
<img width="939" height="630" alt="image" src="https://github.com/user-attachments/assets/fe153a57-0b99-4eb5-906f-31bbd552c09c" />

### Comparison with the state of the art
<img width="939" height="317" alt="image" src="https://github.com/user-attachments/assets/be951780-9274-4485-be6a-304efc611ce5" />

## Citation
If you use this work, please cite:

```bibtex
@InProceedings{10.1007/978-3-032-04630-7_25,
author="Felix de Almeida, Jo{\~a}o Pedro
and De Almeida Bandeira Macedo, Lucas
and Garcia Freitas, Pedro",
editor="Yin, Xu-Cheng
and Karatzas, Dimosthenis
and Lopresti, Daniel",
title="HoLoSig: Holistic and Local Representation Learning for Online Signature Verification",
booktitle="Document Analysis and Recognition --  ICDAR 2025",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="433--450",
abstract="In this paper, we approach the problem of minimizing the global Equal Error Rate (EER) in a Writer-Independent (WI) Online Signature Verification (OSV) system by decomposing it into two components: the separation error and the alignment error. The separation error arises from the lack of a correct separation between genuine and forged signatures, while the alignment error arises from the inability of a single global threshold to effectively capture the writer-specific separations. To this, we propose HoLoSig, a novel framework that integrates two popular deep signature representations via a shared 1D convolutional backbone that bifurcates into two specialized branches. On one branch, we employ Triplet Loss with Soft-DTW to learn variable-length local representations whose dissimilarity scores are shifted to a common region with the help of the Maximum Mean Discrepancy (MMD) to improve the system's performance when using a global threshold. On the other branch, we employ Poly-1 Cross Entropy Loss to learn fixed-length holistic representations that are used to further boost the separation created by the local representation branch. HoLoSig achieves state-of-the-art results on DeepSignDB, the largest OSV dataset to date, against skilled and random forgeries in the stylus scenario, with EERs of 1.73{\%} (4vs1 skilled), 3.29{\%} (1vs1 skilled), 0.43{\%} (4vs1 random) and 0.89{\%} (1vs1 random). The source code is available at https://github.com/DYosplay/HoLoSig. ",
isbn="978-3-032-04630-7"
}
```
## References

[1] J. Jiang, S. Lai, L. Jin and Y. Zhu, "DsDTW: Local Representation Learning With Deep soft-DTW for Dynamic Signature Verification," in IEEE Transactions on Information Forensics and Security, vol. 17, pp. 2198-2212, 2022, doi: 10.1109/TIFS.2022.3180219

[2] Lai S, Jin L. Recurrent adaptation networks for online signature verification[J]. IEEE Transactions on information forensics and security, 2018, 14(6): 1624-1637.

[3] Cuturi M, Blondel M. Soft-dtw: a differentiable loss function for time-series[C]//International conference on machine learning. PMLR, 2017: 894-903.

[4] Tolosana R, Vera-Rodriguez R, Fierrez J, et al. DeepSign: Deep on-line signature verification[J]. IEEE Transactions on Biometrics, Behavior, and Identity Science, 2021.

[5] S. Lai, L. Jin, Y. Zhu, Z. Li and L. Lin, "SynSig2Vec: Forgery-Free Learning of Dynamic Signature Representations by Sigma Lognormal-Based Synthesis and 1D CNN," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 6472-6485, 1 Oct. 2022, doi: 10.1109/TPAMI.2021.3087619.

[6] https://github.com/LaiSongxuan/SynSig2Vec

[7] https://github.com/Maghoumi/pytorch-softdtw-cuda.git.

[8] https://github.com/KAKAFEI123/DsDTW/tree/main

[9] https://github.com/yiyixuxu/polyloss-pytorch/blob/master/PolyLoss.py

[10] https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py


