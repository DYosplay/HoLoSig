import random
import numpy as np
import os
import DataLoader.loader as loader
from tqdm import tqdm
from typing import Dict, Any
import DTW.dtw_cuda as dtw


def get_files(dataset_folder : str = "../Data/DeepSignDB/Development/stylus", hyperparameters = None):
    files = os.listdir(dataset_folder)
    users = {}
    for file in files:
        if "signs" in file: continue
        if (not hyperparameters["rotation"]) and "rot" in file: continue
        tokens = file.split('_')
        id = int(tokens[0].split('u')[-1])
        database = loader.get_database(id,development=True,hyperparameters=hyperparameters)
       
        key = tokens[0] + tokens[1]
        if key in users:
            users[key].append(dataset_folder + os.sep + file)
        else:
            users[key] = [dataset_folder + os.sep + file]

    return users

def files2array(batch, hyperparameters : Dict[str,Any], z : bool, development : bool):
    data = []; lens = []

    for file in batch:
        feat = None
        
        file = file.replace('\\', os.sep)
        file = file.replace('/', os.sep)
        file = file.strip()
        
        if hyperparameters['signature_path'] is not None:
            file = os.path.join(hyperparameters['signature_path'], file)
            development = 'Development' in file
        elif hyperparameters['dataset_scenario'] != "mix":
            subset_folder = 'Development' if development else "Evaluation"
            file_path = os.path.join(hyperparameters['dataset_folder'], subset_folder, hyperparameters['dataset_scenario'])
            if file_path not in file: file = os.path.join(file_path, file)
        
        signs = None
        if hyperparameters['cache']:
            signs = hyperparameters['signs_dev'] if development else hyperparameters['signs_eva']
        
        if signs is None:
            feat = loader.get_features(file, hyperparameters=hyperparameters, z=z, development=development)
        else:
            key = file.split(os.sep)[-1]
            if hyperparameters['dataset_scenario'] == "mix": key = file
            feat = signs[key]

        data.append(feat)
        lens.append(len(feat[0]))

    max_size = max(lens)

    generated_batch = []
    for i in range(0, len(data)):
        #resized = resize(data[i], max_size)
        resized = np.pad(data[i], [(0,0),(0,max_size-len(data[i][0]))]) 
        generated_batch.append(resized)

    return np.array(generated_batch), lens


def get_batch_from_epoch(epoch, batch_size : int, z : bool, hyperparameters : Dict[str,Any]):
    mini_batch_size = 0
    mini_batch_size = hyperparameters['nf'] + hyperparameters['nr'] + hyperparameters['ng'] + 1
    
    assert batch_size % mini_batch_size == 0
    
    step = batch_size // mini_batch_size

    users = []
    batch = []

    labels = []

    for i in range(0, step):
        b = epoch.pop()
        batch += b
        
        
        for j in range(0, len(b)):
            removed = 0
                
            id = int(b[j].split(os.sep)[-1].split('_')[0].split('u')[1]) #- 1

            if id >= 1008: id -= 511
            if id >= removed: id -= removed
            if '_s_' in b[j]: id += (574-removed) # falsificacao profissional
            users.append(id)

    data, lens = files2array(batch, hyperparameters=hyperparameters, z=z, development=True)
    
    return data, lens, epoch, users, labels

def get_random_ids(user_id, database, hyperparameters, samples = 5):
    if database == loader.EBIOSIGN1_DS1:
        return list(set(random.sample(list(range(1009,1039)), samples+1)) - set([user_id]))[:hyperparameters['nr']]
    elif database == loader.EBIOSIGN1_DS2:
        return list(set(random.sample(list(range(1039,1085)), samples+1)) - set([user_id]))[:hyperparameters['nr']]
    elif database == loader.MCYT:
        return list(set(random.sample(list(range(1,231)), samples+1)) - set([user_id]))[:hyperparameters['nr']]
    elif database == loader.BIOSECUR_ID:
        return list(set(random.sample(list(range(231,499)), samples+1)) - set([user_id]))[:hyperparameters['nr']]
    
    raise ValueError("Dataset desconhecido")


def generate_epoch(dataset_folder : str, hyperparameters : Dict[str, Any], train_offset = [(1, 498), (1009, 1084)], users=None, development = True, model = None, stylus = True):
    files = get_files(dataset_folder=dataset_folder, hyperparameters=hyperparameters)
    files_backup = files.copy()


    train_users = []
    if users is None:
        for t in train_offset:
            train_users += list(range(t[0], t[1]+1))
    else:
        train_users = users

    epoch = []
    number_of_mini_baches = 0

    multiplier = 1
    if hyperparameters['rotation']:
        multiplier = 2

    batch_dtw = dtw.DTW(True, normalize=False, bandwidth=0.07)

    database = None

    print("Gererating new epoch")

    for user_id in tqdm(train_users):
        
        database = loader.get_database(user_id=user_id, development=development, hyperparameters=hyperparameters)

        if stylus:
            # durante alguns experimentos usei apenas um mini batch no ebiosign
            if database == loader.EBIOSIGN1_DS1:
                # continue
                # number_of_mini_baches = 6 * multiplier
                number_of_mini_baches = 1 * multiplier
            elif database == loader.EBIOSIGN1_DS2:
                number_of_mini_baches = 1 * multiplier
            elif database == loader.MCYT or database == loader.BIOSECURE_DS2:
                number_of_mini_baches = 4 * multiplier
                # continue
            elif database == loader.BIOSECUR_ID:
                number_of_mini_baches = 2 * multiplier
                # continue
            else:
                raise ValueError("Dataset desconhecido!")
        else:
            if database == loader.EBIOSIGN1_DS1:
                # continue
                # number_of_mini_baches = 2 * multiplier
                number_of_mini_baches = 1 * multiplier
            elif database == loader.EBIOSIGN1_DS2:
                # number_of_mini_baches = 2 * multiplier
                number_of_mini_baches = 1 * multiplier
            elif database == loader.MCYT or database == loader.BIOSECURE_DS2:
                number_of_mini_baches = 4 * multiplier
                # continue
            elif database == loader.BIOSECUR_ID:
                number_of_mini_baches = 2 * multiplier
                # continue
            else:
                raise ValueError("Dataset desconhecido!")

        for i in range(0, number_of_mini_baches):
            genuines = []
            syn_genuines = []
            s_forgeries = []
            syn_s_forgeries = []
            
           
            genuines = random.sample(files['u' + f"{user_id:04}" + 'g'], hyperparameters['ng'] + 1)
            files['u' + f"{user_id:04}" + 'g'] = list(set(files['u' + f"{user_id:04}" + 'g']) - set(genuines))

            s_forgeries = random.sample(files['u' + f"{user_id:04}" + 's'], hyperparameters['nf'])
            files['u' + f"{user_id:04}" + 's'] = list(set(files['u' + f"{user_id:04}" + 's']) - set(s_forgeries))


            

            # ids aleatórios podem ser de qualquer mini dataset
            random_forgeries_ids = list(set(random.sample(train_users, hyperparameters['nr']+1)) - set([user_id]))[:hyperparameters['nr']]
            # ids aleatórios apenas do mesmo dataset
            # random_forgeries_ids = get_random_ids(user_id=user_id, database=database, hyperparameters=hyperparameters, samples=hyperparameters['nr'])

            # random_dict = {}
            # random_forgeries = list(dict(sorted(random_dict.items(), key=lambda item: item[1])).keys())[:hyperparameters['nr']]
            
            random_forgeries = []

            for id in random_forgeries_ids[:5]:
                random_forgeries.append(random.sample(files_backup['u' + f"{id:04}" + 'g'], 1)[0])
            for id in random_forgeries_ids[5:]:
                random_forgeries.append(random.sample(files_backup['u' + f"{id:04}" + 's'], 1)[0])

            a = [genuines[0]]
            p = genuines[1:] + syn_genuines
            n = s_forgeries + syn_s_forgeries + random_forgeries

            mini_batch = a + p + n

            epoch.append(mini_batch)
    
    del batch_dtw
    

    random.shuffle(epoch)

    bs = len(epoch) // (len(epoch) // hyperparameters['nw'])
    bs *= (len(epoch) // hyperparameters['nw'])
    bs = len(epoch) - bs
    for i in range(0,bs):
        epoch.pop()

    
    return epoch