import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../Transformer'))

import pandas as pd
import numpy as np
import torch
import math
from sklearn.preprocessing import MinMaxScaler
from Transformer import TransformerEncoder

def true_model(df, order_num):
    print("Preprocessing data for Transformer...")
    # Get group lengths and group items
    groups = df.groupby('orderid')
    
    # The Transformer was trained with max_len = 320 (norm_shape=[320, 88])
    max_len = 320
    
    cat_indices = [0, 1, 2, 3, 4, 11, 12]
    conti_indices = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]
    
    data_cat = np.zeros((order_num, max_len, 7), dtype=np.int32)
    data_conti = np.zeros((order_num, max_len, 10), dtype=np.float32)
    valid_len = np.zeros(order_num, dtype=np.int32)
    
    feature_cols = [c for c in df.columns if c not in ['orderid', 'label']]
    
    df_conti = df[feature_cols].iloc[:, conti_indices]
    scaler = MinMaxScaler()
    scaler.fit(df_conti)
    
    for i, (orderid, group) in enumerate(groups):
        v_len = len(group)
        if v_len > max_len:
            v_len = max_len
            group = group.iloc[:max_len]
            
        valid_len[i] = v_len
        
        cat_vals = group[feature_cols].iloc[:, cat_indices].values
        conti_vals = scaler.transform(group[feature_cols].iloc[:, conti_indices].values)
        
        data_cat[i, :v_len, :] = cat_vals
        data_conti[i, :v_len, :] = conti_vals
        
    print("Initializing Transformer...")
    # Transformer parameters
    feature_sizes = [11, 7, 97, 63, 2, 2, 2]
    k = 5
    key_size = query_size = value_size = 88
    num_hiddens, num_layers, dropout = 88, 1, 0.5
    norm_shape = [320, 88]
    norm_shape_init = 10
    ffn_num_input, ffn_num_hiddens, num_heads = 88, 176, 4
    
    embedding_sizes = []
    for i in range(len(feature_sizes)):
        if feature_sizes[i] != 1:
            embedding_sizes.append(math.ceil(k * math.log(feature_sizes[i])))
        else:
            embedding_sizes.append(1)
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = TransformerEncoder(
        key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout, feature_sizes, k, embedding_sizes, norm_shape_init)
        
    model_path = os.path.join(os.path.dirname(__file__), '../Transformer/Simulator/simulator.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), '../Transformer/checkpoints/best_network.pth')
        
    # Load the weights (weights_only=False since old pytorch version might be used to save it, but we can try True first, actually let's just use torch.load)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("Running inference...")
    batch_size = 128
    pro = []
    with torch.no_grad():
        i = 0
        while i < order_num:
            j = min(i + batch_size, order_num)
            
            cat_batch = torch.from_numpy(data_cat[i:j]).to(torch.long).to(device)
            conti_batch = torch.from_numpy(data_conti[i:j]).to(torch.float).to(device)
            valid_lens_batch = torch.from_numpy(valid_len[i:j]).to(torch.long).to(device)
            
            outputs = model([cat_batch, conti_batch, valid_lens_batch])
            
            # outputs shape: (batch_size, sequence_length, 1)
            for jj in range(outputs.shape[0]):
                valid_l = valid_lens_batch[jj].item()
                # apply softmax over the valid choices
                logits = outputs[jj, 0:valid_l, 0]
                probs = torch.softmax(logits, dim=0).cpu().numpy()
                
                # The notebook expects the output to have size (len(df) / order_num)
                out_len = int(len(df) / order_num)
                padded_probs = np.zeros(out_len)
                padded_probs[:valid_l] = probs
                pro.append(padded_probs)
                
            i = j
            
    return np.array(pro)
