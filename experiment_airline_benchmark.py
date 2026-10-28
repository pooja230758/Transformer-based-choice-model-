import os
import sys
import time
import math
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('Transformer'))
sys.path.append(os.path.abspath('Choice_Models_Training_and_Comparison'))

import choicemodels as cm
from Transformer import TransformerEncoder, MaskedSoftmaxCELoss

# Set deterministic seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"=== Choice Models Reproducible Benchmark ===")
print(f"Device: {device}, Random Seed: {SEED}")

# ==========================================
# 1. Load and Preprocess Airline Dataset
# ==========================================
df = pd.read_csv('data_github/transaction_data.csv')
print(f"Loaded transaction_data.csv: {len(df)} rows across {df['orderid'].nunique()} assortments.")

conti_cols = [
    'stayDurationMinutes', 'totalPrice', 'totalTripDurationMinutes', 'dtd',
    'nAirlines', 'nFlights', 'outDepTime_sin', 'outDepTime_cos', 'outArrTime_sin', 'outArrTime_cos'
]
cat_cols = ['officeID', 'depWeekDay', 'OD', 'fAirline', 'staySaturday', 'isContinental', 'isDomestic']

# Min-Max normalization for continuous features
for col in conti_cols:
    c_min = df[col].min()
    c_max = df[col].max()
    if c_max > c_min:
        df[col] = (df[col] - c_min) / (c_max - c_min)
    else:
        df[col] = 0.0

for col in cat_cols:
    df[col] = df[col].astype(int)

# Group by orderid
unique_orders = df['orderid'].unique()
np.random.shuffle(unique_orders)

# Stratified/Consistent train/val/test split by orders
n_train = 4000
n_val = 1000
n_test = 1000

train_oids = set(unique_orders[:n_train])
val_oids = set(unique_orders[n_train:n_train + n_val])
test_oids = list(unique_orders[n_train + n_val:n_train + n_val + n_test])

df_train = df[df['orderid'].isin(train_oids)].copy().reset_index(drop=True)
df_val = df[df['orderid'].isin(val_oids)].copy().reset_index(drop=True)
df_test = df[df['orderid'].isin(set(test_oids))].copy().reset_index(drop=True)

# Build 3D tensors: shape (num_orders, choice_size=30, feature_size=17)
MAX_CHOICE_SIZE = 30

def create_assortment_tensors(data_df, order_list, choice_size=MAX_CHOICE_SIZE):
    num_orders = len(order_list)
    cat_tensor = np.zeros((num_orders, choice_size, 17), dtype=np.int64)
    weight_tensor = np.zeros((num_orders, choice_size, 17), dtype=np.float32)
    label_tensor = np.zeros((num_orders, choice_size), dtype=np.float32)
    valid_lens = np.zeros(num_orders, dtype=np.int64)
    
    order_grouped = data_df.groupby('orderid')
    
    cat_slots = [0, 1, 2, 3, 4, 11, 12]
    conti_slots = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]
    
    for i, oid in enumerate(order_list):
        group = order_grouped.get_group(oid)
        alen = min(len(group), choice_size)
        valid_lens[i] = alen
        
        # Categorical slots
        for idx, col in enumerate(cat_cols):
            cat_tensor[i, :alen, cat_slots[idx]] = group[col].values[:alen]
            weight_tensor[i, :alen, cat_slots[idx]] = 1.0
            
        # Continuous slots
        for idx, col in enumerate(conti_cols):
            cat_tensor[i, :alen, conti_slots[idx]] = 0
            weight_tensor[i, :alen, conti_slots[idx]] = group[col].values[:alen]
            
        label_tensor[i, :alen] = group['label'].values[:alen]
        
    return cat_tensor, weight_tensor, label_tensor, valid_lens

train_cat, train_wt, train_lbl, train_vlen = create_assortment_tensors(df_train, list(train_oids))
val_cat, val_wt, val_lbl, val_vlen = create_assortment_tensors(df_val, list(val_oids))
test_cat, test_wt, test_lbl, test_vlen = create_assortment_tensors(df_test, test_oids)

# Build 28-feature Assortment Tensors for DeepFM-a
def create_assortment_features_28(cat_17, wt_17, valid_lens):
    num_orders, choice_size, _ = cat_17.shape
    cat_28 = np.concatenate([cat_17, np.zeros((num_orders, choice_size, 9), dtype=np.int64), np.zeros((num_orders, choice_size, 2), dtype=np.int64)], axis=2)
    wt_28 = np.concatenate([wt_17, np.ones((num_orders, choice_size, 9), dtype=np.float32), np.ones((num_orders, choice_size, 2), dtype=np.float32)], axis=2)
    
    for i in range(num_orders):
        alen = valid_lens[i]
        if alen == 0:
            continue
        durations = wt_17[i, :alen, 5]
        prices = wt_17[i, :alen, 6]
        trips = wt_17[i, :alen, 7]
        
        wt_28[i, :alen, 17] = np.min(durations)
        wt_28[i, :alen, 18] = np.max(durations)
        wt_28[i, :alen, 19] = np.mean(durations)
        
        wt_28[i, :alen, 20] = np.min(prices)
        wt_28[i, :alen, 21] = np.max(prices)
        wt_28[i, :alen, 22] = np.mean(prices)
        
        wt_28[i, :alen, 23] = np.min(trips)
        wt_28[i, :alen, 24] = np.max(trips)
        wt_28[i, :alen, 25] = np.mean(trips)
        
        min_p = np.min(prices)
        max_p = np.max(prices)
        for j in range(alen):
            if wt_17[i, j, 6] == min_p:
                cat_28[i, j, 26] = 1
            if wt_17[i, j, 6] == max_p:
                cat_28[i, j, 27] = 1
    return cat_28, wt_28

train_cat_28, train_wt_28 = create_assortment_features_28(train_cat, train_wt, train_vlen)
val_cat_28, val_wt_28 = create_assortment_features_28(val_cat, val_wt, val_vlen)
test_cat_28, test_wt_28 = create_assortment_features_28(test_cat, test_wt, test_vlen)

# Build Transformer 320-padded inputs
TRANS_CHOICE_SIZE = 320
def create_transformer_inputs(data_df, order_list):
    num_orders = len(order_list)
    cat_trans = np.zeros((num_orders, TRANS_CHOICE_SIZE, 7), dtype=np.int64)
    conti_trans = np.zeros((num_orders, TRANS_CHOICE_SIZE, 10), dtype=np.float32)
    lbl_trans = np.zeros((num_orders, TRANS_CHOICE_SIZE), dtype=np.float32)
    vlen_trans = np.zeros(num_orders, dtype=np.int64)
    
    order_grouped = data_df.groupby('orderid')
    for i, oid in enumerate(order_list):
        group = order_grouped.get_group(oid)
        alen = min(len(group), TRANS_CHOICE_SIZE)
        vlen_trans[i] = alen
        cat_trans[i, :alen] = group[cat_cols].values[:alen]
        conti_trans[i, :alen] = group[conti_cols].values[:alen]
        lbl_trans[i, :alen] = group['label'].values[:alen]
    return cat_trans, conti_trans, lbl_trans, vlen_trans

test_trans_cat, test_trans_conti, test_trans_lbl, test_trans_vlen = create_transformer_inputs(df_test, test_oids)
train_trans_cat, train_trans_conti, train_trans_lbl, train_trans_vlen = create_transformer_inputs(df_train, list(train_oids))

print("Data preparation complete!")

# ==========================================
# Evaluation Helper Function
# ==========================================
def evaluate_choice_predictions(prob_matrix, label_matrix, valid_lens):
    """
    prob_matrix: shape (N, choice_size) choice probabilities
    label_matrix: shape (N, choice_size) binary ground truth
    valid_lens: shape (N) number of products in assortment
    """
    top1_correct = 0
    top3_correct = 0
    total_purchase_assortments = 0
    
    total_log_loss = 0.0
    rmse_sum = 0.0
    total_elements = 0
    
    # Assortment size breakdown
    small_top1 = 0
    small_total = 0
    large_top1 = 0
    large_total = 0
    
    N = len(valid_lens)
    for i in range(N):
        alen = valid_lens[i]
        if alen == 0:
            continue
        p = prob_matrix[i, :alen]
        y = label_matrix[i, :alen]
        
        # Normalize probabilities over the active assortment
        s = np.sum(p)
        if s > 0:
            p = p / s
        else:
            p = np.ones(alen) / alen
            
        rmse_sum += np.sum((p - y) ** 2)
        total_elements += alen
        
        if np.sum(y) > 0:
            total_purchase_assortments += 1
            true_idx = np.argmax(y)
            pred_sorted = np.argsort(-p)
            
            # Top-1
            is_top1 = (pred_sorted[0] == true_idx)
            if is_top1:
                top1_correct += 1
            # Top-3
            if true_idx in pred_sorted[:min(3, alen)]:
                top3_correct += 1
                
            # Assortment size breakdown: Small <= 15, Large > 15
            if alen <= 15:
                small_total += 1
                if is_top1:
                    small_top1 += 1
            else:
                large_total += 1
                if is_top1:
                    large_top1 += 1
                    
            # Cross-entropy / Log Loss on chosen item
            chosen_prob = max(p[true_idx], 1e-12)
            total_log_loss += -np.log(chosen_prob)
            
    top1_acc = (top1_correct / total_purchase_assortments) * 100 if total_purchase_assortments > 0 else 0
    top3_acc = (top3_correct / total_purchase_assortments) * 100 if total_purchase_assortments > 0 else 0
    avg_log_loss = total_log_loss / total_purchase_assortments if total_purchase_assortments > 0 else 0
    rmse = np.sqrt(rmse_sum / total_elements) if total_elements > 0 else 0
    
    small_acc = (small_top1 / small_total) * 100 if small_total > 0 else 0
    large_acc = (large_top1 / large_total) * 100 if large_total > 0 else 0
    
    return {
        'top1_acc': top1_acc,
        'top3_acc': top3_acc,
        'log_loss': avg_log_loss,
        'rmse': rmse,
        'small_acc': small_acc,
        'large_acc': large_acc,
        'purchase_count': total_purchase_assortments
    }

results = {}

# ==========================================
# Model 1: MNL (Multinomial Logit)
# ==========================================
print("\n--- Training Model 1: MNL ---")
model_mnl = cm.MLE().to(device)
opt_mnl = optim.Adam(model_mnl.parameters(), lr=0.01)
criterion_mnl = cm.MNL_loss().to(device)

t0_mnl = time.time()
batch_size = 64
n_batches = int(np.ceil(train_cat.shape[0] / batch_size))

for epoch in range(15):
    model_mnl.train()
    perm = np.random.permutation(train_cat.shape[0])
    for b in range(n_batches):
        idxs = perm[b*batch_size : min((b+1)*batch_size, train_cat.shape[0])]
        b_cat = torch.from_numpy(train_cat[idxs]).to(device)
        b_wt = torch.from_numpy(train_wt[idxs]).to(device)
        b_lbl = torch.from_numpy(train_lbl[idxs]).to(device)
        
        opt_mnl.zero_grad()
        out, u = model_mnl([b_cat, b_wt])
        loss = criterion_mnl(out, b_lbl, u)
        loss.backward()
        opt_mnl.step()
t_train_mnl = time.time() - t0_mnl
print(f"MNL trained in {t_train_mnl:.2f}s")

# Inference MNL
t0_inf = time.time()
model_mnl.eval()
with torch.no_grad():
    test_out, _ = model_mnl([torch.from_numpy(test_cat).to(device), torch.from_numpy(test_wt).to(device)])
    pred_mnl = torch.exp(test_out).cpu().numpy()
t_inf_mnl = time.time() - t0_inf

mnl_metrics = evaluate_choice_predictions(pred_mnl, test_lbl, test_vlen)
mnl_metrics['train_time'] = t_train_mnl
mnl_metrics['inf_time'] = t_inf_mnl
mnl_metrics['inf_latency_ms'] = (t_inf_mnl / len(test_vlen)) * 1000
results['MNL'] = mnl_metrics
print(f"MNL Results: Top-1={mnl_metrics['top1_acc']:.2f}%, Top-3={mnl_metrics['top3_acc']:.2f}%, LogLoss={mnl_metrics['log_loss']:.3f}, Latency={mnl_metrics['inf_latency_ms']:.3f}ms")

# ==========================================
# Model 2: MMNL (Mixed Multinomial Logit)
# ==========================================
print("\n--- Training Model 2: MMNL (3 Latent Classes) ---")
K_classes = 3
mmnl_models = [cm.MLE().to(device) for _ in range(K_classes)]
t0_mmnl = time.time()
for k, m in enumerate(mmnl_models):
    torch.manual_seed(SEED + k * 10)
    opt_k = optim.Adam(m.parameters(), lr=0.015)
    for epoch in range(12):
        m.train()
        perm = np.random.permutation(train_cat.shape[0])
        for b in range(n_batches):
            idxs = perm[b*batch_size : min((b+1)*batch_size, train_cat.shape[0])]
            b_cat = torch.from_numpy(train_cat[idxs]).to(device)
            b_wt = torch.from_numpy(train_wt[idxs]).to(device)
            b_lbl = torch.from_numpy(train_lbl[idxs]).to(device)
            opt_k.zero_grad()
            out, u = m([b_cat, b_wt])
            loss = criterion_mnl(out, b_lbl, u)
            loss.backward()
            opt_k.step()
alphas = np.array([0.4, 0.35, 0.25])
t_train_mmnl = time.time() - t0_mmnl
print(f"MMNL trained in {t_train_mmnl:.2f}s")

# Inference MMNL
t0_inf = time.time()
with torch.no_grad():
    mmnl_probs = np.zeros_like(test_lbl)
    for k, m in enumerate(mmnl_models):
        m.eval()
        k_out, _ = m([torch.from_numpy(test_cat).to(device), torch.from_numpy(test_wt).to(device)])
        k_prob = torch.exp(k_out).cpu().numpy()
        for i in range(len(test_vlen)):
            alen = test_vlen[i]
            if alen > 0:
                s = np.sum(k_prob[i, :alen])
                if s > 0:
                    k_prob[i, :alen] /= s
        mmnl_probs += alphas[k] * k_prob
t_inf_mmnl = time.time() - t0_inf

mmnl_metrics = evaluate_choice_predictions(mmnl_probs, test_lbl, test_vlen)
mmnl_metrics['train_time'] = t_train_mmnl
mmnl_metrics['inf_time'] = t_inf_mmnl
mmnl_metrics['inf_latency_ms'] = (t_inf_mmnl / len(test_vlen)) * 1000
results['MMNL'] = mmnl_metrics
print(f"MMNL Results: Top-1={mmnl_metrics['top1_acc']:.2f}%, Top-3={mmnl_metrics['top3_acc']:.2f}%, LogLoss={mmnl_metrics['log_loss']:.3f}, Latency={mmnl_metrics['inf_latency_ms']:.3f}ms")

# ==========================================
# Model 3: DeepFM
# ==========================================
print("\n--- Training Model 3: DeepFM ---")
model_dfm = cm.DeepCell().to(device)
opt_dfm = optim.Adam(model_dfm.parameters(), lr=0.005)
weight1 = 25.0
class_weight = torch.tensor([1.0, weight1]).to(device)

t0_dfm = time.time()
for epoch in range(15):
    model_dfm.train()
    perm = np.random.permutation(train_cat.shape[0])
    for b in range(n_batches):
        idxs = perm[b*batch_size : min((b+1)*batch_size, train_cat.shape[0])]
        b_cat = torch.from_numpy(train_cat[idxs]).to(device)
        b_wt = torch.from_numpy(train_wt[idxs]).to(device)
        b_lbl = torch.from_numpy(train_lbl[idxs]).to(device)
        
        opt_dfm.zero_grad()
        out = model_dfm([b_cat, b_wt])
        w_var = class_weight[b_lbl.long()]
        crit = nn.BCEWithLogitsLoss(weight=w_var)
        loss = crit(out, b_lbl)
        loss.backward()
        opt_dfm.step()
t_train_dfm = time.time() - t0_dfm
print(f"DeepFM trained in {t_train_dfm:.2f}s")

# Inference DeepFM
t0_inf = time.time()
model_dfm.eval()
with torch.no_grad():
    dfm_out = model_dfm([torch.from_numpy(test_cat).to(device), torch.from_numpy(test_wt).to(device)])
    pred_dfm = torch.sigmoid(dfm_out).cpu().numpy()
t_inf_dfm = time.time() - t0_inf

dfm_metrics = evaluate_choice_predictions(pred_dfm, test_lbl, test_vlen)
dfm_metrics['train_time'] = t_train_dfm
dfm_metrics['inf_time'] = t_inf_dfm
dfm_metrics['inf_latency_ms'] = (t_inf_dfm / len(test_vlen)) * 1000
results['DeepFM'] = dfm_metrics
print(f"DeepFM Results: Top-1={dfm_metrics['top1_acc']:.2f}%, Top-3={dfm_metrics['top3_acc']:.2f}%, LogLoss={dfm_metrics['log_loss']:.3f}, Latency={dfm_metrics['inf_latency_ms']:.3f}ms")

# ==========================================
# Model 4: DeepFM-a (Assortment Features)
# ==========================================
print("\n--- Training Model 4: DeepFM-a ---")
model_dfma = cm.DeepCell_With_Assortment().to(device)
opt_dfma = optim.Adam(model_dfma.parameters(), lr=0.005)

t0_dfma = time.time()
for epoch in range(15):
    model_dfma.train()
    perm = np.random.permutation(train_cat_28.shape[0])
    for b in range(n_batches):
        idxs = perm[b*batch_size : min((b+1)*batch_size, train_cat_28.shape[0])]
        b_cat = torch.from_numpy(train_cat_28[idxs]).to(device)
        b_wt = torch.from_numpy(train_wt_28[idxs]).to(device)
        b_lbl = torch.from_numpy(train_lbl[idxs]).to(device)
        
        opt_dfma.zero_grad()
        out = model_dfma([b_cat, b_wt])
        w_var = class_weight[b_lbl.long()]
        crit = nn.BCEWithLogitsLoss(weight=w_var)
        loss = crit(out, b_lbl)
        loss.backward()
        opt_dfma.step()
t_train_dfma = time.time() - t0_dfma
print(f"DeepFM-a trained in {t_train_dfma:.2f}s")

# Inference DeepFM-a
t0_inf = time.time()
model_dfma.eval()
with torch.no_grad():
    dfma_out = model_dfma([torch.from_numpy(test_cat_28).to(device), torch.from_numpy(test_wt_28).to(device)])
    pred_dfma = torch.sigmoid(dfma_out).cpu().numpy()
t_inf_dfma = time.time() - t0_inf

dfma_metrics = evaluate_choice_predictions(pred_dfma, test_lbl, test_vlen)
dfma_metrics['train_time'] = t_train_dfma
dfma_metrics['inf_time'] = t_inf_dfma
dfma_metrics['inf_latency_ms'] = (t_inf_dfma / len(test_vlen)) * 1000
results['DeepFM-a'] = dfma_metrics
print(f"DeepFM-a Results: Top-1={dfma_metrics['top1_acc']:.2f}%, Top-3={dfma_metrics['top3_acc']:.2f}%, LogLoss={dfma_metrics['log_loss']:.3f}, Latency={dfma_metrics['inf_latency_ms']:.3f}ms")

# ==========================================
# Model 5: Transformer-Based Choice Model
# ==========================================
print("\n--- Evaluating Model 5: Transformer Choice Model ---")
feature_sizes = [11, 7, 97, 63, 2, 2, 2]
k = 5
norm_shape = [320, 88]
norm_shape_init = 10
key_size = query_size = value_size = 88
num_hiddens, num_layers, dropout = 88, 1, 0.5
ffn_num_input, ffn_num_hiddens, num_heads = 88, 176, 4
embedding_sizes = [math.ceil(k * math.log(fs)) if fs != 1 else 1 for fs in feature_sizes]

model_trans = TransformerEncoder(
    key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout, feature_sizes, k, embedding_sizes, norm_shape_init).to(device)

# Load pre-trained simulator checkpoint
ckpt_path = 'Transformer/Simulator/simulator.pth'
if os.path.exists(ckpt_path):
    model_trans.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("Loaded pretrained simulator.pth weights successfully!")

# Measure training / fine-tuning time for 5 epochs
t0_trans = time.time()
opt_trans = optim.Adadelta(model_trans.parameters(), lr=0.1)
crit_trans = MaskedSoftmaxCELoss(torch.tensor([30.0])).to(device)
model_trans.train()
t_batch_size = 32
t_nbatches = int(np.ceil(train_trans_cat.shape[0] / t_batch_size))
for epoch in range(5):
    perm = np.random.permutation(train_trans_cat.shape[0])
    for b in range(t_nbatches):
        idxs = perm[b*t_batch_size : min((b+1)*t_batch_size, train_trans_cat.shape[0])]
        b_cat = torch.from_numpy(train_trans_cat[idxs]).to(device)
        b_conti = torch.from_numpy(train_trans_conti[idxs]).to(device)
        b_lbl = torch.from_numpy(train_trans_lbl[idxs]).to(device)
        b_lens = torch.from_numpy(train_trans_vlen[idxs]).to(device)
        
        opt_trans.zero_grad()
        out = model_trans([b_cat, b_conti, b_lens])
        loss = crit_trans(out, b_lbl, b_lens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_trans.parameters(), 8.0)
        opt_trans.step()
t_train_trans = time.time() - t0_trans
print(f"Transformer fine-tuned 5 epochs in {t_train_trans:.2f}s")

# Inference Transformer
t0_inf = time.time()
model_trans.eval()
trans_probs = np.zeros_like(test_lbl)
with torch.no_grad():
    for b in range(int(np.ceil(test_trans_cat.shape[0] / t_batch_size))):
        start = b * t_batch_size
        end = min((b + 1) * t_batch_size, test_trans_cat.shape[0])
        b_cat = torch.from_numpy(test_trans_cat[start:end]).to(device)
        b_conti = torch.from_numpy(test_trans_conti[start:end]).to(device)
        b_lens = torch.from_numpy(test_trans_vlen[start:end]).to(device)
        
        out = model_trans([b_cat, b_conti, b_lens]) # (batch, 320, 1)
        for j in range(end - start):
            alen = min(test_trans_vlen[start + j], MAX_CHOICE_SIZE)
            trans_probs[start + j, :alen] = out[j, :alen, 0].cpu().numpy()
t_inf_trans = time.time() - t0_inf

trans_metrics = evaluate_choice_predictions(trans_probs, test_lbl, test_vlen)
trans_metrics['train_time'] = t_train_trans
trans_metrics['inf_time'] = t_inf_trans
trans_metrics['inf_latency_ms'] = (t_inf_trans / len(test_vlen)) * 1000
results['Transformer'] = trans_metrics
print(f"Transformer Results: Top-1={trans_metrics['top1_acc']:.2f}%, Top-3={trans_metrics['top3_acc']:.2f}%, LogLoss={trans_metrics['log_loss']:.3f}, Latency={trans_metrics['inf_latency_ms']:.3f}ms")

# ==========================================
# Print Results Summary Table
# ==========================================
print("\n" + "="*85)
print(f"{'Model':<15} | {'Top-1 (%)':<10} | {'Top-3 (%)':<10} | {'Log Loss':<10} | {'RMSE':<10} | {'Train(s)':<9} | {'Inference(ms)':<13}")
print("="*85)
for m_name in ['MNL', 'MMNL', 'DeepFM', 'DeepFM-a', 'Transformer']:
    m = results[m_name]
    print(f"{m_name:<15} | {m['top1_acc']:<10.2f} | {m['top3_acc']:<10.2f} | {m['log_loss']:<10.3f} | {m['rmse']:<10.4f} | {m['train_time']:<9.2f} | {m['inf_latency_ms']:<13.3f}")
print("="*85)

print("\n--- Assortment Size Breakdown: Small (<=15) vs Large (>15) ---")
print(f"{'Model':<15} | {'Small Assortment Top-1 (%)':<26} | {'Large Assortment Top-1 (%)':<26}")
print("-" * 75)
for m_name in ['MNL', 'MMNL', 'DeepFM', 'DeepFM-a', 'Transformer']:
    m = results[m_name]
    print(f"{m_name:<15} | {m['small_acc']:<26.2f} | {m['large_acc']:<26.2f}")
print("-" * 75)

# Save results to CSV
res_rows = []
for m_name, m in results.items():
    row = {'Model': m_name}
    row.update(m)
    res_rows.append(row)
res_df = pd.DataFrame(res_rows)
res_df.to_csv('experiment_results_summary.csv', index=False)
print("Saved experiment_results_summary.csv")

# ==========================================
# Generate Graphs
# ==========================================
os.makedirs('figures', exist_ok=True)
models = ['MNL', 'MMNL', 'DeepFM', 'DeepFM-a', 'Transformer']
colors = ['#4A90E2', '#50E3C2', '#F5A623', '#E94E77', '#9013FE']

# Graph 1: Top-1 and Top-3 Accuracy
plt.figure(figsize=(9, 5))
x = np.arange(len(models))
width = 0.35
top1_vals = [results[m]['top1_acc'] for m in models]
top3_vals = [results[m]['top3_acc'] for m in models]

plt.bar(x - width/2, top1_vals, width, label='Top-1 Accuracy (%)', color='#2E86AB', edgecolor='black')
plt.bar(x + width/2, top3_vals, width, label='Top-3 Accuracy (%)', color='#A23B72', edgecolor='black')
plt.xlabel('Choice Model Architecture', fontsize=12, fontweight='bold')
plt.ylabel('Prediction Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Choice Model Accuracy Comparison on Airline Booking Data', fontsize=14, fontweight='bold')
plt.xticks(x, models, fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
for i in range(len(models)):
    plt.text(x[i] - width/2, top1_vals[i] + 0.6, f"{top1_vals[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
    plt.text(x[i] + width/2, top3_vals[i] + 0.6, f"{top3_vals[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/model_accuracy_comparison.png', dpi=300)
plt.close()

# Graph 2: Accuracy vs Inference Latency Pareto Tradeoff
plt.figure(figsize=(9, 5))
latencies = [results[m]['inf_latency_ms'] for m in models]
log_losses = [results[m]['log_loss'] for m in models]

for i, m in enumerate(models):
    plt.scatter(latencies[i], top1_vals[i], color=colors[i], s=180, edgecolors='black', zorder=5)
    offset_y = 0.5 if i % 2 == 0 else -0.8
    plt.annotate(f"{m}\n({latencies[i]:.2f}ms, {top1_vals[i]:.1f}%)", (latencies[i], top1_vals[i] + offset_y), fontsize=10, fontweight='bold', ha='center')

plt.xlabel('Inference Latency (ms per assortment)', fontsize=12, fontweight='bold')
plt.ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Accuracy vs. Computational Latency Trade-Off', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/model_latency_tradeoff.png', dpi=300)
plt.close()

# Graph 3: Assortment Size Breakdown (Small vs Large)
plt.figure(figsize=(9, 5))
small_vals = [results[m]['small_acc'] for m in models]
large_vals = [results[m]['large_acc'] for m in models]

plt.bar(x - width/2, small_vals, width, label='Small Assortments (<= 15 products)', color='#2B5B84', edgecolor='black')
plt.bar(x + width/2, large_vals, width, label='Large Assortments (> 15 products)', color='#E07A5F', edgecolor='black')
plt.xlabel('Choice Model Architecture', fontsize=12, fontweight='bold')
plt.ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Model Performance on Small vs. Large Airline Assortments', fontsize=14, fontweight='bold')
plt.xticks(x, models, fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
for i in range(len(models)):
    plt.text(x[i] - width/2, small_vals[i] + 0.6, f"{small_vals[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
    plt.text(x[i] + width/2, large_vals[i] + 0.6, f"{large_vals[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/assortment_size_breakdown.png', dpi=300)
plt.close()

print("Figures successfully generated and saved to figures/ directory!")
