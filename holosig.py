from typing import List, Tuple, Dict, Any
from utils.utilities import dump_hyperparameters
import torch.nn.utils as nutils
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.optim as optim
from matplotlib import pyplot as plt
import os
import sys
import math
from tqdm import tqdm
import numpy as np
from Losses import triplet_mmd
import DataLoader.batches_gen as batches_gen
import DataLoader.loader as loader
import DTW.dtw_cuda as dtw
import utils.metrics as metrics
from Losses import polyloss
import time

import pickle

import wandb

import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


BUFFER = ''

ROC_FAR = 0
ROC_FRR = 0
TOTAL_P = 0
TOTAL_N = 0


class SelectivePooling(nn.Module):
    # from https://github.com/LaiSongxuan/SynSig2Vec
    def __init__(self, in_dim, head_dim, num_heads, tau=1.0):
        super(SelectivePooling, self).__init__()
        self.keys = nn.Parameter(torch.Tensor(num_heads, head_dim), requires_grad=True)
        self.w_q = nn.Conv1d(in_dim, head_dim * num_heads, kernel_size=1)
        self.norm = 1 / head_dim**0.5
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.count = 0
        nn.init.orthogonal_(self.keys, gain=1)
        # nn.init.kaiming_normal_(self.keys, a=1)
        nn.init.kaiming_normal_(self.w_q.weight, a=1)
        nn.init.zeros_(self.w_q.bias)

    def forward(self, x, mask, save=False):
        N = x.shape[0]; T = x.shape[2]
        queries = values = self.w_q(x).transpose(1, 2).view(N, T, self.num_heads, self.head_dim) #(N, T, num_heads, head_dim)
        atten = F.softmax(torch.sum(queries * self.keys, dim=-1) * self.norm - (1.-mask).unsqueeze(2)*1000, dim=1) #(N, T, num_heads)  
        head = torch.sum(values * atten.unsqueeze(3), dim=1).view(N, -1) #(N, num_heads * head_dim)
        # if save: numpy.save("./expScripts/attenWeight_bsid/atten%d.npy"%self.count, atten.detach().cpu().numpy()); self.count += 1
        return head

    def orthoNorm(self):
        keys = F.normalize(self.keys, dim=1)
        corr = torch.mm(keys, keys.transpose(0, 1))
        return torch.sum(torch.triu(corr, 1).abs_())
    
class HoLoSig(nn.Module):
    def __init__(self, hyperparameters : Dict[str, Any]):    
        super(HoLoSig, self).__init__()

        # Tuneable Hyperparameters
        self.hyperparameters = hyperparameters
        self.margin = torch.nn.Parameter(torch.tensor(hyperparameters['margin']), requires_grad=False)
        self.lr = torch.nn.Parameter(torch.tensor(hyperparameters['learning_rate']), requires_grad=False)
        self.alpha = torch.nn.Parameter(torch.tensor(hyperparameters['alpha']), requires_grad=False)
        self.beta = torch.nn.Parameter(torch.tensor(hyperparameters['beta']), requires_grad=False)
        self.p = torch.nn.Parameter(torch.tensor(hyperparameters['p']), requires_grad=False)
        self.r = torch.nn.Parameter(torch.tensor(hyperparameters['r']), requires_grad=False)
        self.q =hyperparameters['q']
        self.epsilon = hyperparameters['epsilon']
        self.cew = hyperparameters['cew']

        # variáveis de controle
        self.z = hyperparameters['zscore']
        self.batch_size = hyperparameters['batch_size']
        self.dataset_folder = hyperparameters['dataset_folder']
        
        # parametros de definicao da rede
        self.n_layers = hyperparameters["ngru"]
        self.n_in = hyperparameters["ninput"]
        self.n_out = hyperparameters["nout"]
        self.n_hidden = hyperparameters["nhidden"]
        self.n_head = hyperparameters["nhead"]
        self.n_hdim = hyperparameters["hdim"]
        
        self.n_classes = 574 * 2

        if hyperparameters['dataset_scenario'] == "mix":
            self.n_classes += 76 * 2
        if hyperparameters['dataset_scenario'] == "finger":
            self.n_classes = 76 * 2

        print("Number of classes: " + str(self.n_classes))

        # Variáveis que lidam com as métricas/resultados
        self.buffer = "File, epoch, mean_local_eer, global_eer, th_global, separation_eer, alignment_eer, path100mean, path100var\n"
        self.eer = []
        self.best_eer_skilled = math.inf
        self.best_eer = math.inf
        self.best_eer_random = math.inf
        self.last_eer = math.inf
        self.loss_variation = []

        # Definição da rede
        self.cran  = nn.Sequential(
        nn.Conv1d(in_channels=self.n_in, out_channels=self.n_out, kernel_size=7, stride=1, padding=3, bias=True),
        nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True), 
        nn.BatchNorm1d(self.n_out),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels=self.n_out, out_channels=self.n_hidden, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm1d(self.n_hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=self.hyperparameters['dropout_cnn'])
        )

        self.rnn = nn.GRU(self.n_hidden, self.n_hidden, self.n_layers, dropout=self.hyperparameters['dropout_rnn'], batch_first=True, bidirectional=False)
        self.sp = SelectivePooling(in_dim=self.n_hidden, head_dim=self.n_hdim, num_heads=self.n_head)
        self.bn = nn.BatchNorm1d(self.n_hdim*self.n_head)
        self.linear2 = nn.Linear(self.n_hdim*self.n_head, self.n_classes, bias=False)


        self.h0 = Variable(torch.zeros(self.n_layers, self.batch_size, self.n_hidden).cuda(), requires_grad=False)
        self.h1 = Variable(torch.zeros(self.n_layers, 5, self.n_hidden).cuda(), requires_grad=False)
        self.h2 = Variable(torch.zeros(self.n_layers, 2, self.n_hidden).cuda(), requires_grad=False)

        for i in range(self.n_layers):
            eval("self.rnn.bias_hh_l%d"%i)[self.n_hidden:2*self.n_hidden].data.fill_(-1e10) #Initial update gate bias
            eval("self.rnn.bias_ih_l%d"%i)[self.n_hidden:2*self.n_hidden].data.fill_(-1e10) #Initial update gate bias
    
        self.linear = nn.Linear(self.n_hidden, 64, bias=False)

        nn.init.kaiming_normal_(self.linear.weight, a=1)
        nn.init.kaiming_normal_(self.linear2.weight, a=1) 
        nn.init.kaiming_normal_(self.cran[0].weight, a=0)
        nn.init.kaiming_normal_(self.cran[4].weight, a=0)
        nn.init.zeros_(self.cran[0].bias)
        nn.init.zeros_(self.cran[4].bias)


        self.triplet_mmd = triplet_mmd.Triplet_MMD(ng=hyperparameters['ng'],nf=self.hyperparameters['nf'], nr=self.hyperparameters['nr'],nw=self.hyperparameters['nw'],margin=self.hyperparameters['margin'], random_margin=self.hyperparameters['random_margin'], alpha=self.hyperparameters['alpha'], beta=self.hyperparameters['beta'], p=self.hyperparameters['p'], r=self.hyperparameters['r'], mmd_kernel_num=self.hyperparameters['mmd_kernel_num'], mmd_kernel_mul=self.hyperparameters['mmd_kernel_mul'], tau=self.hyperparameters['tau'], s=self.hyperparameters['s'])
        
        self.dtw = dtw.DTW(True, normalize=False, bandwidth=1)
        self.polyloss = polyloss.PolyLoss(epsilon=self.epsilon)

        self.hyperparameters['signs_dev'] = None
        self.hyperparameters['signs_eva'] = None

        # Ativa cache, se necessário
        if self.hyperparameters['cache']:
            self.hyperparameters['signs_dev'], self.hyperparameters['signs_eva'] = self.generate_signatures()

    def getOutputMask(self, lens):    
        lens = np.array(lens, dtype=np.int32)
        lens = (lens + 1) // 2
        N = len(lens); D = np.max(lens)
        mask = np.zeros((N, D), dtype=np.float32)
        for i in range(N):
            mask[i, 0:lens[i]] = 1.0
        return mask
    
    def forward(self, x, mask, n_epoch):
        length = torch.sum(mask, dim=1)

        '''Sorting according length'''
        length, indices = torch.sort(length, descending=True)
        x = torch.index_select(x, 0, indices)
        mask = torch.index_select(mask, 0, indices)

        h = self.cran(x)
        h = h.transpose(1,2)
        h = h * mask.unsqueeze(2)

        h4, h3, h2 = None, None, None

        self.h0 = Variable(torch.zeros(self.n_layers, h.shape[0], self.n_hidden).cuda(), requires_grad=False)
        h = nutils.rnn.pack_padded_sequence(h, list(length.cpu().numpy().astype(int)), batch_first=True)
        h, hidden = self.rnn(h, self.h0)
        
        h, length = nutils.rnn.pad_packed_sequence(h, batch_first=True) 
        length = Variable(length).cuda()

        '''Recover the original order'''
        _, indices = torch.sort(indices, descending=False)
        h = torch.index_select(h, 0, indices)
        h4 = h
        hidden = torch.index_select(hidden.transpose(0,1), 0, indices)
        length = torch.index_select(length, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        
        h1 = self.linear(h)

        
        if (not self.training) or (self.training and self.hyperparameters['cross_entropy']):
            h2 = self.sp(h.transpose(1,2), mask)



        if self.training:
            if self.hyperparameters['cross_entropy']:
                h3 = self.bn(h2)
                h3 = self.linear2(h3)
  
            return F.avg_pool1d(h1.permute(0,2,1),2,2,ceil_mode=False).permute(0,2,1), (length//2).float(), h2, h3, h4

        return h1 * mask.unsqueeze(2), length.float(), h2
    
    def _dte(self, x, y, len_x, len_y, optimal_choice = True):   
        v, matrix = self.dtw(x[None, :int(len_x)], y[None, :int(len_y)], optimal_choice) 
        return v / (64*((len_x**2 + len_y**2)**0.5)), matrix
    
    def getEER(self, FAR, FRR):
        a = FRR <= FAR
        s = np.sum(a)
        a[-s-1] = 1
        a[-s+1:] = 0
        FRR = FRR[a]
        FAR = FAR[a] 
        a = [[FRR[1]-FRR[0], FAR[0]-FAR[1]], [-1, 1]]
        b = [(FRR[1]-FRR[0])*FAR[0]-(FAR[1]-FAR[0])*FRR[0], 0]
        return np.linalg.solve(a, b)

    
    def _inference(self, files : str, n_epoch : int, result_folder : str = None, cache : dict = None, cache_l2 : dict = None) -> Tuple[float, str, int, Dict, Dict, List]:
        """
        Args:
            files (str): string no formato: ref1 [,ref2, ref3, ref4], sign, label 
            result_folder (str): pasta onde os resultados serao salvos (nao utilizada)
            cache (Dict[Dict[str,float]]): distancias ja computadas para acelerar inferencia. 
        Raises:
            ValueError: "Arquivos de comparação com formato desconhecido"
        Returns:
            float, str, int: distância da assinatura, usuário, label, cache dtw, cache l2, tamanho do template
        """
        with torch.no_grad():
            tokens = files.split(" ")
            user_key = tokens[0].split("_")[0]
            
            result = math.nan
            refs = []
            sign = ""
            s_avg = 0
            s_min = 0

            if len(tokens) == 2:
                a = tokens[0].split('_')[0]
                b = tokens[1].split('_')[0]
                result = 0 if a == b and '_g_' in tokens[1] else 1; refs.append(tokens[0]); sign = tokens[1]
            elif len(tokens) == 3: result = int(tokens[2]); refs.append(tokens[0]); sign = tokens[1]
            elif len(tokens) == 6: result = int(tokens[5]); refs = tokens[0:4]; sign = tokens[4]
            else: raise ValueError("Arquivos de comparação com formato desconhecido")

            refs_names = refs
            sign_name = sign
            dk_names = tokens[:-1]
            synthetic_signs = []

            test_batch, lens = batches_gen.files2array(synthetic_signs + refs + [sign], hyperparameters=self.hyperparameters, z=self.z, development=self.hyperparameters["development"])

            mask = self.getOutputMask(lens)
            
            mask = Variable(torch.from_numpy(mask)).cuda()
            inputs = Variable(torch.from_numpy(test_batch)).cuda()

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            embeddings, lengths, h2 = self(inputs.float(), mask, n_epoch)   

            dk_dists1 = []
            dk_dists2 = []

            """Referencias para dk"""
            for i in range(0, len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    dka = math.nan; dka2 = math.nan
                    if(tokens[i] in cache and tokens[j] in cache[tokens[i]]):
                        dka = cache[tokens[i]][tokens[j]].cuda()
                        dka2 = cache_l2[tokens[i]][tokens[j]].cuda()
                    else:
                        dka = self._dte(embeddings[i], embeddings[j], lengths[i], lengths[j])[0]
                        dka2 = torch.norm(h2[i]/(self.n_head*self.n_hdim)-h2[j]/(self.n_head*self.n_hdim),p=2)
                        
                        if tokens[i] not in cache:
                            cache[tokens[i]] = {tokens[j] : dka.cpu()}
                            cache_l2[tokens[i]] = {tokens[j] : dka2.cpu()}
                        else:
                            cache[tokens[i]][tokens[j]] = dka.cpu()
                            cache_l2[tokens[i]][tokens[j]] = dka2.cpu()

                    dk_dists1.append(dka.item())
                    dk_dists2.append(dka2.item())

            dk_dists1 = np.array(dk_dists1)
            dk_dists2 = np.array(dk_dists2)

            """Referencias para query"""

            dists_query1 = []
            dists_query2 = []
            for i in range(0, len(refs_names)):
                v1 = cache[refs_names[i]][sign_name].item()
                v2 = cache_l2[refs_names[i]][sign_name].item()
                dists_query1.append(v1)
                dists_query2.append(v2)

            dists_query1 = np.array(dists_query1)
            dists_query2 = np.array(dists_query2)
    
            """Calculo dk"""
            dk1_sqrt = (np.mean(dk_dists1)) ** 0.5 + 0.0005
            dk2_sqrt = (np.mean(dk_dists2)) ** 0.5 + 0.0005

            """Calculo de pontuacao"""
            dists_query1 /= dk1_sqrt
            s_avg1 = np.mean(dists_query1)
            s_min1 = min(dists_query1)
            score1 = (s_avg1 + s_min1)

            dists_query2 /= dk2_sqrt
            s_avg2 = np.mean(dists_query2)
            s_min2 = min(dists_query2)
            score2 = (s_avg2 + s_min2)

            score = (score1 + score2) / 2

            torch.cuda.synchronize()
            end_time = time.perf_counter()

            return score, user_key, result, cache, cache_l2, lens[:-1], end_time - start_time

    def new_evaluate(self, comparison_file : str, n_epoch : int, result_folder : str):
        """ Avaliação da rede conforme o arquivo de comparação

        Args:
            comparison_file (str): path do arquivo que contém as assinaturas a serem comparadas entre si, bem como a resposta da comparação. 0 é positivo (original), 1 é negativo (falsificação).
            n_epoch (int): número que indica após qual época de treinamento a avaliação está sendo realizada.
            result_folder (str): path onde salvar os resultados.
        """

        self.train(mode=False)
        lines = []
        with open(comparison_file, "r") as fr:
            lines = fr.readlines()

        if not os.path.exists(result_folder): os.mkdir(result_folder)

        file_name = (comparison_file.split(os.sep)[-1]).split('.')[0]
        print("\n\tAvaliando " + file_name)
        comparison_folder = result_folder + os.sep + file_name
        if not os.path.exists(comparison_folder): os.mkdir(comparison_folder)

        users = {}
        dists_dict = {}
        cache = {}
        cache_l2 = {}

        for line in tqdm(lines, "Calculando distâncias..."):
            distance, user_id, true_label, cache, cache_l2, template_len, time_taken = self._inference(line, n_epoch=n_epoch, result_folder=result_folder, cache=cache, cache_l2=cache_l2)
            dists_dict[line] = distance
            inconstant_frequency = ("_w4_" in line or "_w5_" in line)
            
            if user_id not in users: 
                users[user_id] = {"distances": [distance], "true_label": [true_label], "predicted_label": [], "template_len": template_len, "inconstant_frequency":inconstant_frequency, "time_taken": []}
            else:
                users[user_id]["distances"].append(distance)
                users[user_id]["time_taken"].append(time_taken)
                users[user_id]["true_label"].append(true_label)

        # Nesse ponto, todos as comparações foram feitas
        buffer = "user, eer_local, threshold, mean_eer, var_th, amp_th, th_range, path100mean, path100var\n"
        local_buffer = ""
        global_true_label = []
        global_distances = []
        global_separation = []

        eers = []

        local_ths = []

        global_time_taken = []

        # Calculo do EER local por usuário:
        for user in tqdm(users, desc="Obtendo EER local..."):
            global_true_label += users[user]["true_label"]
            global_distances  += users[user]["distances"]
            global_time_taken += users[user]["time_taken"]
            
            # if "Task" not in comparison_file:
            if 0 in users[user]["true_label"] and 1 in users[user]["true_label"]:
                    
                eer, eer_threshold = metrics.get_eer(y_true=users[user]["true_label"], y_scores=users[user]["distances"])
                users[user]["th"] = eer_threshold
                global_separation += (np.array(users[user]["distances"]) - eer_threshold).tolist()
                th_range_local = np.max(np.array(users[user]["distances"])[np.array(users[user]["distances"]) < eer_threshold])

                local_ths.append(eer_threshold)
                eers.append(eer)
                local_buffer += user + ", " + "{:.5f}".format(eer) + ", " + "{:.5f}".format(eer_threshold) + ", 0, 0, 0, " + "{:.5f}".format(eer_threshold -th_range_local) + " (" + "{:.5f}".format(th_range_local) + "~" + "{:.5f}".format(eer_threshold) + ")\n"

        print("Obtendo EER global...")
        
        # Calculo do EER global
        scores = np.array(global_distances)
        labels = np.array(global_true_label)

        user_p = scores[(np.where(labels == 0))]
        user_n = scores[(np.where(labels == 1))]

        th = np.arange(0, 5, 0.001)[None,:]
        FRR = 1. - np.sum(user_p[:,None] - th <= 0, axis=0) / float(user_p.shape[0])
        FAR = 1. - np.sum(user_n[:,None] - th >= 0, axis=0) / float(user_n.shape[0])
        print(self.getEER(FAR, FRR)[0] * 100)


        eer_global, eer_threshold_global = metrics.get_eer(global_true_label, global_distances, result_folder=comparison_folder, generate_graph=True, n_epoch=n_epoch)
        eer_separation, separation_th = metrics.get_eer(global_true_label, global_separation, result_folder=comparison_folder, generate_graph=False, n_epoch=n_epoch)
        alignment_error = eer_global - eer_separation


        """Plot histograma de erro de acordo com duracao da assinatura"""

        correct = []
        incorrect = []

        correct_separation = []
        incorrect_separation = []

        for line in tqdm(lines, "Calculando distâncias..."):
            if("_w4_" in line or "_w5_" in line): continue  # ignora w4 e w5

            user_id = line.split(" ")[0].split("_")[0]
            avg_template_size = np.mean(np.array(users[user_id]["template_len"]))
            id = int(user_id.split("u")[-1])
            database = loader.get_database(id, development=False, hyperparameters=self.hyperparameters)
            if database == loader.EBIOSIGN1_DS1 or database == loader.EBIOSIGN1_DS2:
                avg_template_size /= 200
            else:
                avg_template_size /= 100
            
            true_label = bool(int((line.split(" ")[-1]).strip())) # true se eh falsificacao
            predicted = dists_dict[line] >= eer_threshold_global
            predicted_separation = dists_dict[line] - users[user]["th"]
            predicted_separation = predicted_separation >= 0
            
            if true_label == predicted:
                correct.append(avg_template_size)
            else:
                incorrect.append(avg_template_size)

            if true_label == predicted_separation:
                correct_separation.append(avg_template_size)
            else:
                incorrect_separation.append(avg_template_size)

        histogram_name = os.path.join(comparison_folder, f"{n_epoch:03d}" + "_Length_" + file_name)
        time_plot_name = os.path.join(comparison_folder, f"{n_epoch:03d}" + "_TimeTaken_" + file_name)
        histogram_name_sep = os.path.join(comparison_folder, f"{n_epoch:03d}" + "_Length_Sep_" + file_name)

        metrics.plot_histogram_error_by_length(correct,incorrect,histogram_name)
        metrics.plot_histogram_error_by_length(correct_separation,incorrect_separation,histogram_name_sep)
        metrics.plot_inference_time_kde(global_time_taken, time_plot_name)




        """Gera relatorios"""
        local_eer_mean = np.mean(np.array(eers))
        local_ths = np.array(local_ths)
        local_ths_var  = np.var(local_ths)
        local_ths_amp  = np.max(local_ths) - np.min(local_ths)
        
        th_range_global = np.max(np.array(global_distances)[np.array(global_distances) < eer_threshold_global])

        buffer += "Global, " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + ", " + "{:.5f}".format(local_eer_mean) + ", " + "{:.5f}".format(eer_separation) + ", " + "{:.5f}".format(alignment_error) + ", " + "{:.5f}".format(eer_threshold_global -th_range_global) + " (" + "{:.5f}".format(th_range_global) + "~" + "{:.5f}".format(eer_threshold_global) + "), 0, 0" + "\n" + local_buffer

        self.buffer += str(n_epoch) + ", " + "{:.5f}".format(local_eer_mean) + ", " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + ", " + "{:.5f}".format(eer_separation) + ", " + "{:.5f}".format(alignment_error) + ", " + "{:.5f}".format(eer_threshold_global -th_range_global) + " (" + "{:.5f}".format(th_range_global) + "~" + "{:.5f}".format(eer_threshold_global) + "), 0, 0" + "\n"

        with open(comparison_folder + os.sep + file_name + " epoch=" + str(n_epoch) + ".csv", "w") as fw:
            fw.write(buffer)

        self.last_eer = eer_global

        if "random" in comparison_file:
            self.best_eer_random = min(self.best_eer_random, eer_global)
        elif "skilled" in comparison_file.lower() and "4" in comparison_file:
            self.best_eer_skilled = min(self.best_eer_skilled, eer_global)
            self.eer.append(eer_global)

        key = comparison_file.split(os.sep)[-1]
        ret_metrics = {"Global EER - " + key: eer_global, "Mean Local EER - " + key: local_eer_mean, "Global Threshold - " + key: eer_threshold_global, "Separation EER - " + key: eer_separation, "Alignment EER - " + key: alignment_error}
        if n_epoch != 0 and n_epoch < 100:
            if eer_global < self.best_eer:
                torch.save(self.state_dict(), result_folder + os.sep + "Backup" + os.sep + "best.pt")
                self.best_eer = eer_global

        with open(os.path.join(comparison_folder,'acc_distance_dict_' + str(n_epoch) + '.pickle'), 'wb') as fw:
            pickle.dump(dists_dict, fw)

        with open(os.path.join(comparison_folder,'cache_dtw_' + str(n_epoch) + '.pickle'), 'wb') as fw:
            pickle.dump(cache, fw)
        
        with open(os.path.join(comparison_folder,'cache_l2_' + str(n_epoch) + '.pickle'), 'wb') as fw:
            pickle.dump(cache_l2, fw)

        print("\n\t Resultados:")
        print(ret_metrics)

        self.train(mode=True)

        del cache
        del cache_l2

        return ret_metrics
    
    def generate_signatures(self): 
        signs_dev = None
        signs_eva = None

        if self.hyperparameters['dataset_scenario'] == 'mix':
            os.makedirs(os.path.join(self.hyperparameters['dataset_folder'], "mix"), exist_ok=True)
            data_path = os.path.join(self.hyperparameters['dataset_folder'], "mix")
            signs_dev = {}
            if os.path.exists(data_path + os.sep + "signs_dev.pickle"):
                with open(data_path + os.sep + "signs_dev.pickle", 'rb') as file:
                    signs_dev = pickle.load(file)

            else:     
                data_paths = os.path.join(self.hyperparameters['dataset_folder'], "Development", 'stylus'),os.path.join(self.hyperparameters['dataset_folder'], "Development", 'finger')  
                for data_path in data_paths:    
                    files = sorted(os.listdir(data_path))

                    for f in tqdm(files, desc="Generating Training Signatures"):
                        if "_syn_" in f or ".pickle" in f: continue

                        path = os.path.join(data_path, f)
                        feat = loader.get_features(path,self.hyperparameters,self.hyperparameters['zscore'],development=True)
                        signs_dev[path] = feat

                with open(os.path.join(self.hyperparameters['dataset_folder'], "mix","signs_dev.pickle"), 'wb') as file:
                    pickle.dump(signs_dev, file)

                print("signs_dev.pickle generated. Size: " + str(sys.getsizeof(signs_dev)))

        ##
        data_path = os.path.join(self.hyperparameters['dataset_folder'], "Development", self.hyperparameters['dataset_scenario'])       
        if os.path.exists(data_path + os.sep + "signs_dev.pickle"):
            with open(data_path + os.sep + "signs_dev.pickle", 'rb') as file:
                signs_dev = pickle.load(file)
        
        if signs_dev is None:
            files = sorted(os.listdir(data_path))
            signs_dev = {}
            for f in tqdm(files, desc="Generating Training Signatures"):
                if "_syn_" in f: continue

                path = os.path.join(data_path, f)
                feat = loader.get_features(path,self.hyperparameters,self.hyperparameters['zscore'],development=True)
                signs_dev[f] = feat

            with open(data_path + os.sep + "signs_dev.pickle", 'wb') as file:
                pickle.dump(signs_dev, file)

            print("signs_dev.pickle generated. Size: " + str(sys.getsizeof(signs_dev)))


        if self.hyperparameters['dataset_scenario'] == 'mix' or self.hyperparameters['dataset_scenario'] == 'finger':
            data_path = os.path.join(self.hyperparameters['dataset_folder'], "Evaluation", 'finger')
        else:       
            data_path = os.path.join(self.hyperparameters['dataset_folder'], "Evaluation", self.hyperparameters['dataset_scenario'])
        if os.path.exists(data_path + os.sep + "signs_eva.pickle"):
            with open(data_path + os.sep + "signs_eva.pickle", 'rb') as file:
                signs_eva = pickle.load(file)


        if signs_eva is None:
            files = sorted(os.listdir(data_path))
            signs_eva = {}
            for f in tqdm(files, "Generating Testing Signatures"):
                if "_syn_" in f: continue
                path = os.path.join(data_path, f)
                feat = loader.get_features(path,self.hyperparameters,self.hyperparameters['zscore'],development=False)
                signs_eva[f] = feat

            with open(data_path + os.sep + "signs_eva.pickle", 'wb') as file:
                pickle.dump(signs_eva, file)

            print("signs_eva.pickle generated. Size: " + str(sys.getsizeof(signs_eva)))
    
        return signs_dev, signs_eva
    
    def _get_epoch(self,data_path : str) -> List[str]:
        """ Gera epoca com as assinaturas presentes em data_path seguindo os criterios definidos pelos hiperparametros.
        Args:
            data_path (str): pasta com as assinaturas
        Returns:
            List[str]: nome das assinaturas que formam a epoca
        """
        train_offset = [(1, 498), (1009, 1084)]

        epoch = batches_gen.generate_epoch(dataset_folder=data_path, hyperparameters=self.hyperparameters, train_offset=train_offset, development=True)
        
        return epoch
            
    def start_train(self, comparison_files : List[str], result_folder : str):
        """ Loop de treinamento

        Args:
            comparison_files (List[str]): Lista com as paths dos arquivos de comparação a serem avaliados durante o treinamento.
            result_folder (str): Path de onde os resultados de avaliação e o backup dos pesos devem ser armazenados.
        """

        dump_hyperparameters(hyperparameters=self.hyperparameters, res_folder=result_folder)

        optimizer = optim.SGD(self.parameters(), lr=self.hyperparameters['learning_rate'], momentum=self.hyperparameters['momentum'])
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hyperparameters['decay']) 

        bckp_path = os.path.join(result_folder, "Backup")
        data_path = os.path.join(self.hyperparameters['dataset_folder'], "Development", self.hyperparameters['dataset_scenario'])
    
        os.makedirs(bckp_path, exist_ok=True)

        mini_batch_size = self.hyperparameters['ng'] + self.hyperparameters['nf'] + self.hyperparameters['nr']
        
        lce_acc, tm_acc, non_zero_random, nonzero, running_loss, l2_acc, mmd_acc,domain_acc = 0, 0, 0, 0, 0, 0,0,0
        for i in range(1, self.hyperparameters['epochs']+1):
            epoch = None

            print("Learning Rate: " + str(lr_scheduler.get_last_lr()))

            epoch = self._get_epoch(data_path)
            epoch_size = len(epoch)

            self.loss_value = running_loss/epoch_size
            mini_batches = self.batch_size//mini_batch_size

            total = (epoch_size//mini_batches)
            pbar = tqdm(total=total, position=0, leave=True, desc="Epoch " + str(i) +" PAL: " + "{:.4f}".format(self.loss_value))

            lce_acc, tm_acc, non_zero_random, nonzero, running_loss, l2_acc, mmd_acc, domain_acc = 0, 0, 0, 0, 0, 0,0,0

            """% de triplas na epoca com falsificacoes aleatorias cuja loss foi maior que 0"""
            triples_per_mini_batch = self.hyperparameters['ng'] * self.hyperparameters['nr']
            triples_in_batch = triples_per_mini_batch * self.hyperparameters['nw']
            triples_in_epoch = triples_in_batch * epoch_size
    
            iterations = 0
            while epoch != []:
                iterations += 1 
                optimizer.zero_grad()
                batch, lens, epoch, targets, _ = batches_gen.get_batch_from_epoch(epoch, self.batch_size, z=self.z, hyperparameters=self.hyperparameters)

                """Prepara entrada da rede"""
                mask = self.getOutputMask(lens)
                mask = Variable(torch.from_numpy(mask)).cuda()
                inputs = Variable(torch.from_numpy(batch)).cuda()
                
                """Forward"""
                outputs, length, output2, output3, output4 = self(inputs.float(), mask, i)

                nw = self.batch_size // mini_batch_size
                """Calcula cross entropy, se necessário"""
                if self.hyperparameters["cross_entropy"]:
                    targets = torch.tensor(targets).cuda()
                    
                    loss2 = 0
                    for j in range(0, nw):
                        loss2 = loss2 + self.polyloss(output3[j*mini_batch_size:(j+1)*mini_batch_size], targets[j*mini_batch_size:(j+1)*mini_batch_size])
                    loss2 = (loss2 / nw) * self.cew

                    loss2.backward(retain_graph=True)
                    lce_acc += loss2.item()  
                    
                loss, tloss, mmdl, nonzero = self.triplet_mmd(outputs, length)
                loss = loss  
                loss.backward()
              
                total_loss = tloss.item() + mmdl.item() + loss2.item()

                tm_acc += tloss.item()
                mmd_acc += mmdl.item()
                non_zero_random += nonzero
                
                optimizer.step()

                running_loss += total_loss
            
                pbar.update(1)

            pbar.close()

            
            nonzero_random = non_zero_random / triples_in_epoch
            print("Non zero random triplets %:\t\t" + str(nonzero_random * 100))

            print(f"Cross Entropy: {lce_acc:.6f}\tTriplet Loss: {tm_acc:.6f}")

            """Inferencia no conjunto de testes"""
            if (i > 1 and ((i-1) % self.hyperparameters['eval_step'] == 0 or i > (self.hyperparameters['epochs'] - 1) )):
                for idx, cf in enumerate(comparison_files):
                    key = cf.split(os.sep)[-1]
                    ret_metrics = self.new_evaluate(comparison_file=cf, n_epoch=i, result_folder=result_folder)
            
            """Passo no SGD"""
            if self.hyperparameters['optimizer'] == 'sgd':
                lr_scheduler.step()
            

            """Backup dos pesos"""
            torch.save(self.state_dict(), bckp_path + os.sep + "epoch" + str(i) + ".pt")

        # Loss graph
        plt.xlabel("#Epoch")
        plt.ylabel("Loss")
        plt.plot(list(range(0,len(self.loss_variation))), self.loss_variation)
        plt.savefig(result_folder + os.sep + "loss.png")
        plt.cla()