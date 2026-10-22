# -*- coding: utf-8 -*-
from __future__ import annotations
import math, random, os, gc
from pathlib import Path
import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
import logging
import json


LOG_DIR = r"D:\Config\Code\MicroSpy\log_tf"
TRAIN_H5 = r"D:\Config\Code\MicroSpy\datasets\trainset.h5"
TEST_H5      = r"D:\Config\Code\MicroSpy\datasets\testset.h5"

HEADS = [
    "layer_type","filters","units","kernel_size","strides",
    "pooling_type","pool_size","activation","padding"
]
HEAD_TYPES = {
    "layer_type":"cls",
    "filters":"cls",
    "units":"cls",
    "kernel_size":"cls",
    "strides":"cls",
    "pooling_type":"cls",
    "pool_size":"cls",
    "activation":"cls",
    "padding":"cls",
}


MAX_LAYERS = 16
FEATURE_DIM= 8 
D_MODEL   = 96
N_HEAD    = 8
ENC_LAY   = 2
DEC_LAY   = 4
DROPOUT   = 0.1
ILLEGAL = -1

EPOCHS  = 20
BATCH   = 64
LR      = 1e-4
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
WORKERS = 0 if os.name=="nt" else 4
PATIENCE= 5
SEED    = 42

LABEL_SMOOTH = 0.005
PAD_WEIGHT   = 1.0
WEIGHT_DECAY = 1e-4
PER_SAMPLE_NORM = True

LT_CONV = 0
LT_POOL = 1
LT_DENSE = 2

HEAD_APPLICABLE_LTS = {
    "filters": {LT_CONV},
    "units": {LT_DENSE},
    "kernel_size": {LT_CONV},
    "strides": {LT_CONV, LT_POOL},
    "pooling_type": {LT_POOL},
    "pool_size": {LT_POOL},
    "activation": {LT_CONV, LT_DENSE},
    "padding": {LT_CONV, LT_POOL},
}

TASK_WEIGHTS = {
    "layer_type": 2,
    "filters": 1.0,
    "units": 1.0,
    "kernel_size": 1.0,
    "strides": 1.0,
    "pooling_type": 1.0,
    "pool_size": 1.0,
    "activation": 1.0,
    "padding": 1.0,
}


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)


def collate(batch):
    em  = torch.stack([b[0] for b in batch])                   # (B, L, C)
    lbl = {h: torch.stack([b[1][h] for b in batch]) for h in HEADS}
    return em, lbl



def make_loaders(bs: int, workers: int = 0, seed: int = 42):
    rng = np.random.default_rng(seed)

    with h5py.File(TRAIN_H5, "r") as f_pool:
        n_pool = len(f_pool["em"])
    if n_pool < 2:
        raise ValueError("The training pool must contain at least 2 samples.")

    indices = rng.permutation(n_pool)
    n_train = int(n_pool * 0.8)
    n_train = min(max(n_train, 1), n_pool - 1)

    idx_train_all = indices[:n_train].tolist()
    idx_val_all = indices[n_train:].tolist()

    with h5py.File(TEST_H5, "r") as ftest:
        n_test = len(ftest["em"])
    idx_test_all = list(range(n_test))

    ds_tr = MultiH5Subset(
        sources=[(TRAIN_H5, idx_train_all)],
        heads=HEADS,
        external_stats=None,
        stats_cache_tag="feat_stats_train_split.npz",
        chunk=512,
    )

    external_stats = (ds_tr._mu, ds_tr._std)

    ds_vl = MultiH5Subset(
        sources=[(TRAIN_H5, idx_val_all)],
        heads=HEADS,
        external_stats=external_stats,
        stats_cache_tag=None,
        chunk=512,
    )

    ds_te = MultiH5Subset(
        sources=[(TEST_H5, idx_test_all)],
        heads=HEADS,
        external_stats=external_stats,
        stats_cache_tag=None,
        chunk=512,
    )

    mk = lambda d, sh: DataLoader(
        d,
        bs,
        sh,
        collate_fn=collate,
        num_workers=workers,
        pin_memory=True,
    )

    tr_loader = mk(ds_tr, True)
    vl_loader = mk(ds_vl, False)
    te_loader = mk(ds_te, False)

    return tr_loader, vl_loader, te_loader, ds_tr.num_cls, ds_tr.freq



class RegHead(nn.Module):
    def __init__(self, d, lo: float, hi: float):
        super().__init__()
        self.fc = nn.Linear(d, 1)
        self.lo = lo
        self.hi = hi
    def forward(self, z):  # z: (B, T, d)
        y = self.fc(z)
        y = torch.sigmoid(y) * (self.hi - self.lo) + self.lo
        return y  # (B, T, 1)

class PosEnc(nn.Module):
    def __init__(self, d:int, max_len: int = 4096):
        super().__init__()
        self.d = d
        pe = self._build_table(max_len)
        self.register_buffer("pe", pe)
    def _build_table(self, L):
        pe = torch.zeros(L, self.d)  
        pos = torch.arange(L, dtype=torch.float32).unsqueeze(1)  
        div = torch.exp(torch.arange(0, self.d, 2, dtype=torch.float32)  
                        * (-math.log(10000.0) / self.d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, L, d)
    def forward(self, x):
        L = x.size(1) 
        if L > self.pe.size(1):
            new = self._build_table(L*2)   
            self.pe = new.to(self.pe.device)
        return x + self.pe[:, :L]  

class LayerQueryDecoder(nn.Module):
    def __init__(self, d=128, nhead=8, n=2, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(MAX_LAYERS, d))  
        layer = nn.TransformerDecoderLayer(
            d, nhead, 4*d, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, n)
    def forward(self, mem):
        q = self.queries.unsqueeze(0).expand(mem.size(0), -1, -1)  
        return self.decoder(q, mem)

class MultiTaskTF(nn.Module):
    def __init__(self, num_cls:dict[str,int], d=128, nhead=8, e=4, de=2, dropout=0.1): 
        super().__init__()
        self.proj=nn.Linear(FEATURE_DIM,d)  
        self.pos =PosEnc(d)

        enc=nn.TransformerEncoderLayer(d, nhead, 4*d, dropout=dropout, batch_first=True) 
        self.encoder=nn.TransformerEncoder(enc, e)  
        self.decoder=LayerQueryDecoder(d, nhead, de, dropout)
        self.heads=nn.ModuleDict()  
        for h in HEADS:
            if HEAD_TYPES[h]=="cls":
                self.heads[h]=nn.Linear(d, num_cls[h]) 
            else:
                lo, hi = RANGE[h]
                self.heads[h]=RegHead(d, lo, hi)
    def forward(self,x):
        mem=self.encoder(self.pos(self.proj(x)))
        z  =self.decoder(mem)                   # (B, MAX_LAYERS, d)
        out={h:head(z) for h,head in self.heads.items()}
        return out

class MultiTaskCNN(nn.Module):
    def __init__(self, num_cls: dict[str, int], d: int = D_MODEL, dropout: float = DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv1d(FEATURE_DIM, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool  = nn.MaxPool1d(kernel_size=2, stride=2)
        self.act   = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.flat_dim = 128 * (800 // 2 // 2 // 2)

        self.fc1 = nn.Linear(self.flat_dim, 128)
        self.fc2 = nn.Linear(128, d)

        self.layer_embed = nn.Embedding(MAX_LAYERS, d)

        self.heads = nn.ModuleDict()
        for h in HEADS:
            self.heads[h] = nn.Linear(d, num_cls[h])

    def forward(self, x):
        # x: (B, L, C=FEATURE_DIM) -> (B, C, L)
        x = x.permute(0, 2, 1)

        # 3 conv + pool blocks -> (B, 128, 100)
        x = self.pool(self.act(self.conv1(x)))  # (B, 32, 400)
        x = self.pool(self.act(self.conv2(x)))  # (B, 64, 200)
        x = self.pool(self.act(self.conv3(x)))  # (B, 128, 100)

        # flatten + 2 FCs -> global feature (B, d)
        x = x.reshape(x.size(0), -1)            # (B, 12800)
        x = self.dropout(F.relu(self.fc1(x)))   # (B, 128)
        g = self.dropout(self.fc2(x))          # (B, d)

        # add layer embeddings -> (B, MAX_LAYERS, d)
        B = g.size(0)
        layer_idx = torch.arange(MAX_LAYERS, device=g.device)
        E = self.layer_embed(layer_idx)         # (MAX_LAYERS, d)
        z = g.unsqueeze(1) + E.unsqueeze(0)     # (B, MAX_LAYERS, d)

        # heads: each returns (B, MAX_LAYERS, num_cls[h])
        out = {h: self.heads[h](z) for h in HEADS}
        return out


def build_weight(freq, power=0.5, max_w=5., w_pad=PAD_WEIGHT): 
    f=np.asarray(freq, np.float32)                         
    w=(f.max()/(f+1e-3))**power   
    w[0]=w_pad 
    return torch.tensor(np.clip(w,1,max_w),dtype=torch.float32,device=DEVICE)


def illegal_map(lt):
    return {
        "kernel_size":   (lt != 0),
        "strides":       (lt == 2),
        "pooling_type":  (lt != 1),
        "pool_size":     (lt != 1),
    }


def criterion(out, lbl, cls_w, task_weights=None):
    if task_weights is None:
        task_weights = {h: 1.0 for h in HEADS}

    loss = 0.0
    correct = {}
    total = {}

    layer_type_raw = lbl["layer_type"]          # (B, T), raw label: conv=0,pool=1,dense=2, invalid=-1
    layer_exist = (layer_type_raw != -1) 

    for h, logits in out.items():
        tgt = (lbl[h] + 1).clamp(min=0)         # -1 -> 0, valid classes shift by +1

        logits2d = logits.reshape(-1, logits.size(-1))
        tgt1d = tgt.reshape(-1)


        if h == "layer_type":
            loss_mask = torch.ones_like(tgt, dtype=torch.bool)
        else:
            allowed_lts = HEAD_APPLICABLE_LTS[h]
            applicable = torch.zeros_like(layer_type_raw, dtype=torch.bool) 

            for lt_id in allowed_lts:
                applicable |= (layer_type_raw == lt_id)

            loss_mask = layer_exist & applicable  

        loss_mask1d = loss_mask.reshape(-1)

        # ---------- cross entropy ----------
        if loss_mask1d.any():
            ce = F.cross_entropy(
                logits2d[loss_mask1d],
                tgt1d[loss_mask1d],
                weight=cls_w[h],
                label_smoothing=LABEL_SMOOTH if LABEL_SMOOTH > 0 else 0.0
            )
        else:
            ce = logits2d.sum() * 0.0

        lambda_t = task_weights.get(h, 1.0)
        loss += lambda_t * ce   
        # loss += ce

        with torch.no_grad():
            pred = logits.argmax(-1)
            correct[h] = (pred[loss_mask] == tgt[loss_mask]).sum().item()
            total[h] = loss_mask.sum().item()

    return loss, correct, total

def mean_acc(metrics:dict) -> float:
    if not metrics:
        return 0.0
    return float(np.mean(list(metrics.values())))

def run_epoch(model, loader, cls_w, train, optim=None, sched=None, warm=0, global_step=0, epoch=0):

    model.train() if train else model.eval()
    sum_correct = {h: 0 for h in HEADS}  
    sum_total   = {h: 0 for h in HEADS}  
    tloss = 0.0
    bar = tqdm(loader, desc=f"{'Train' if train else 'Validation'} {epoch:02d}", ncols=120)

    with torch.set_grad_enabled(train):
        for em, lbl in bar:
            em = em.to(DEVICE)
            lbl = {h: l.to(DEVICE) for h, l in lbl.items()}

            out = model(em)
            loss, correct, total = criterion(out, lbl, cls_w, TASK_WEIGHTS)

            if train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optim.step()
                if global_step < warm:
                    scale = (global_step + 1) / max(1, warm)
                    for pg, base in zip(optim.param_groups, sched.base_lrs):
                        pg["lr"] = base * scale
                else:
                    sched.step()
                global_step += 1

            tloss += loss.item()
            for h in HEADS:
                # sum_metrics[h] += metrics[h]
                sum_correct[h] += correct[h]
                sum_total[h] += total[h]

            lr_val = optim.param_groups[0]["lr"] if train else 0.0
            bar.set_postfix(l=f"{loss.item():.2f}", lr=f"{lr_val:.2e}")

    n = len(loader) # batch num
    acc = {h: (100.0 * sum_correct[h] / sum_total[h]) for h in HEADS}
    return tloss / n, acc, mean_acc(acc), global_step


def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{ts}_train_log.txt")

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger, log_path

def train_transformer():
    logger, LOG_PATH = setup_logger()
    logger.info(f"[log] write to: {LOG_PATH}")
    tr, vl, te, num_cls, freq = make_loaders(BATCH, WORKERS)

    cls_w = {}
    for h in HEADS:
        if HEAD_TYPES[h]=="cls":
            cls_w[h] = build_weight(freq[h])  

    # net=MultiTaskTF(num_cls,D_MODEL,N_HEAD,ENC_LAY,DEC_LAY,DROPOUT).to(DEVICE)
    net = MultiTaskCNN(num_cls, D_MODEL, DROPOUT).to(DEVICE)
    optim=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(   
            optim, T_max=len(tr)*EPOCHS, eta_min=1e-6) 
    warm=len(tr) 
    g=0;
    best_vl_loss=float('inf'); 
    wait=0 


    for ep in range(1,EPOCHS+1):
        tr_loss, tr_acc, tr_mean_acc, g = run_epoch(
            net, tr, cls_w, True, optim, sched, warm, g, ep)
        vl_loss, vl_acc, vl_mean_acc, _ = run_epoch(
            net, vl, cls_w, False, epoch=ep)

        def fmt(h):
            return f"{tr_acc[h]:.1f}/{vl_acc[h]:.1f}"

        log = " | ".join(f"{h}:{fmt(h)}" for h in HEADS)
        logger.info(
            f"Epoch {ep:02d} loss(tr/vl):{tr_loss:.4f}/{vl_loss:.4f} || "
            f"mean acc(tr/vl):{tr_mean_acc:.4f}/{vl_mean_acc:.4f} || {log}"
        )

      
        if vl_loss < best_vl_loss - 1e-5:
            best_vl_loss = vl_loss
            wait = 0
            torch.save({
                        "state_dict": net.state_dict(),
                        "num_cls": num_cls,
                        "cls_w": {h: cls_w[h].detach().cpu() for h in HEADS},
                    }, "best_model_cnn.pt")
            logger.info(f"  [✓] saved (by val loss {vl_loss:.6f})")
        else:
            wait += 1
            logger.info(f"  no improve on val loss {wait}/{PATIENCE} (best {best_vl_loss:.6f})")

        if wait >= PATIENCE:
            logger.info("Early stop")
            break

        gc.collect()


    best_path = "best_model_cnn.pt"
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        net.load_state_dict(ckpt["state_dict"])
        logger.info(f"[test] loaded best checkpoint: {best_path}")
    else:
        logger.info(f"[test] best checkpoint not found, evaluate current weights.")

    te_loss, te_acc, te_mean_acc, _ = run_epoch(net, te, cls_w, train=False, epoch=0)

    def fmt_test(h):
        return f"{te_acc[h]:.1f}"

    log_te = " | ".join(f"{h}:{fmt_test(h)}" for h in HEADS)
    logger.info(f"[TEST] loss{te_loss:.4f}  mean acc{te_mean_acc:.4f} || {log_te}")



if __name__=="__main__":
    train_transformer()

    

    