
#import os
#import warnings
#warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchcrf import CRF
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel


DATA_PATH = "/kaggle/input/recognition/ner.csv"  
BEST_PATH = "best_bert_bigru_crf.pt"

BATCH_SIZE = 16
MAX_WORD_LEN = 18        
MAX_WORDS = 140          
BERT_MODEL = "distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv(DATA_PATH, sep=",", encoding="latin1", engine="python", on_bad_lines="skip")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


sents = []
for idx, g in df.groupby("sentence_idx"):
    sent = []
    for _, row in g.iterrows():
        w = str(row["word"]) if pd.notna(row["word"]) else ""
        t = row["tag"] if pd.notna(row["tag"]) else "PAD"
        sent.append((w, t))
    sents.append(sent)

print("Loaded", len(sents), "sentences. Example:", sents[0][:10])


all_chars = {c for sent in sents for w, _ in sent for c in w}
char2idx = {"PAD": 0, "UNK": 1}
char2idx.update({c: i for i, c in enumerate(sorted(all_chars), start=2)})

tags = sorted(set(df["tag"].dropna().values))
tag2idx = {"PAD": 0}
tag2idx.update({t: i + 1 for i, t in enumerate(tags)})
idx2tag = {i: t for t, i in tag2idx.items()}

print("Char vocab size:", len(char2idx), "Num tags:", len(tag2idx))


#  Helper encoders (char + label arrays)

def encode_chars_for_sentence(words, max_words=MAX_WORDS, max_word_len=MAX_WORD_LEN):
    chars = []
    for w in words[:max_words]:
        ch_ids = [char2idx.get(c, char2idx["UNK"]) for c in w[:max_word_len]]
        if len(ch_ids) < max_word_len:
            ch_ids += [char2idx["PAD"]] * (max_word_len - len(ch_ids))
        chars.append(ch_ids)
    while len(chars) < max_words:
        chars.append([char2idx["PAD"]] * max_word_len)
    return np.array(chars, dtype=np.int64)

def encode_labels_for_sentence(tags_seq, max_words=MAX_WORDS):
    labs = [tag2idx.get(t, tag2idx["PAD"]) for t in tags_seq[:max_words]]
    while len(labs) < max_words:
        labs.append(tag2idx["PAD"])
    mask = [1 if i < len(tags_seq[:max_words]) else 0 for i in range(max_words)]
    return np.array(labs, dtype=np.int64), np.array(mask, dtype=np.int64)


raw_words_list = [[w for w, t in sent] for sent in sents]
raw_tags_list  = [[t for w, t in sent] for sent in sents]


train_words, val_words, train_tags, val_tags = train_test_split(
    raw_words_list, raw_tags_list, test_size=0.1, random_state=42
)

train_char_arrays = [encode_chars_for_sentence(w) for w in train_words]
train_label_arrays = [encode_labels_for_sentence(t)[0] for t in train_tags]
train_masks = [encode_labels_for_sentence(t)[1] for t in train_tags]

val_char_arrays = [encode_chars_for_sentence(w) for w in val_words]
val_label_arrays = [encode_labels_for_sentence(t)[0] for t in val_tags]
val_masks = [encode_labels_for_sentence(t)[1] for t in val_tags]

all_train_label_flat = np.concatenate([arr[arr != tag2idx["PAD"]].ravel() for arr in train_label_arrays])
unique, counts = np.unique(all_train_label_flat, return_counts=True)
tag_freq = {int(u): int(c) for u, c in zip(unique, counts)}
inv_tag_freq = {t: 1.0 / c for t, c in tag_freq.items()}

sent_weights = []
for labs in train_label_arrays:
    nonpad = labs[labs != tag2idx["PAD"]]
    if len(nonpad) > 0:
        w = float(np.mean([inv_tag_freq.get(int(x), 0.0) for x in nonpad]))
    else:
        w = 0.0
    sent_weights.append(w)
sent_weights = np.array(sent_weights, dtype=np.float64)
if sent_weights.max() > 0:
    sent_weights /= sent_weights.max()
sent_weights = 0.05 + 0.95 * sent_weights
sampler = WeightedRandomSampler(weights=torch.tensor(sent_weights, dtype=torch.double),
                                num_samples=len(sent_weights), replacement=True)

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=True)

class NERDatasetRaw(Dataset):
    def __init__(self, words_list, tags_list, char_arrays, label_arrays, masks):
        self.words_list = words_list
        self.tags_list = tags_list
        self.char_arrays = char_arrays
        self.label_arrays = label_arrays
        self.masks = masks
    def __len__(self): return len(self.words_list)
    def __getitem__(self, idx):
        return {
            "words": self.words_list[idx],
            "chars": self.char_arrays[idx],
            "labels": self.label_arrays[idx],
            "mask": self.masks[idx],
        }

def collate_fn(batch):
    words_batch = [b["words"][:MAX_WORDS] for b in batch]
    labels_batch = [b["labels"][:MAX_WORDS] for b in batch]
    chars_batch = [b["chars"][:MAX_WORDS] for b in batch]
    masks_batch = [b["mask"][:MAX_WORDS] for b in batch]

    enc = tokenizer(words_batch,
                    is_split_into_words=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_WORDS * 3)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    batch_size = len(words_batch)
   
    seq_len_sub = input_ids.size(1)
    word_first_idx = torch.zeros((batch_size, MAX_WORDS), dtype=torch.long)
    for i in range(batch_size):
        word_ids = enc.word_ids(batch_index=i)  
        first = {}
        for sub_i, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            if w_id not in first:
                first[w_id] = sub_i
        nwords = min(len(words_batch[i]), MAX_WORDS)
        for w in range(nwords):
            if w in first:
                word_first_idx[i, w] = first[w]
            else:
                word_first_idx[i, w] = 0

    chars_tensor = torch.tensor(np.stack(chars_batch), dtype=torch.long)
    labels_tensor = torch.tensor(np.stack(labels_batch), dtype=torch.long)
    word_mask_tensor = torch.tensor(np.stack(masks_batch), dtype=torch.bool)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_first_idx": word_first_idx,
        "char_ids": chars_tensor,
        "labels": labels_tensor,
        "word_mask": word_mask_tensor
    }

train_dataset = NERDatasetRaw(train_words, train_tags, train_char_arrays, train_label_arrays, train_masks)
val_dataset = NERDatasetRaw(val_words, val_tags, val_char_arrays, val_label_arrays, val_masks)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

class BERT_CharBiGRU_WordBiGRU_CRF(nn.Module):
    def __init__(self, bert_model_name, char_vocab_size, char_emb_dim, char_hidden_dim,
                 word_gru_hidden, num_tags, pad_char_idx=0, dropout=0.5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size

        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=pad_char_idx)
        self.char_gru = nn.GRU(char_emb_dim, char_hidden_dim, batch_first=True, bidirectional=True)

        self.word_gru = nn.GRU(bert_hidden + 2*char_hidden_dim, word_gru_hidden,
                               batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*word_gru_hidden, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

        self.register_buffer("class_weights", torch.ones(num_tags, dtype=torch.float))

    def forward(self, input_ids, attention_mask, word_first_idx, char_ids):
        B = input_ids.size(0)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  

        H = bert_out.size(-1)
        idx_exp = word_first_idx.unsqueeze(-1).expand(-1, -1, H)  
        bert_word_repr = torch.gather(bert_out, dim=1, index=idx_exp)  

        B, W, C = char_ids.size()
        flat_chars = char_ids.view(B * W, C)           
        char_emb = self.char_emb(flat_chars)            
        _, char_hidden = self.char_gru(char_emb)        
        char_repr = torch.cat([char_hidden[0], char_hidden[1]], dim=-1)  
        char_repr = char_repr.view(B, W, -1)           

        combined = torch.cat([bert_word_repr, char_repr], dim=-1)
        combined = self.dropout(combined)

        gru_out, _ = self.word_gru(combined)  
        gru_out = self.dropout(gru_out)

        emissions = self.fc(gru_out)  
        return emissions

    def loss(self, input_ids, attention_mask, word_first_idx, char_ids, tags, mask):
        emissions = self.forward(input_ids, attention_mask, word_first_idx, char_ids) 
        nll = -self.crf(emissions, tags, mask=mask, reduction='none')  

        seq_weights = []
        for b in range(tags.size(0)):
            valid_tags = tags[b][mask[b] == 1]
            if valid_tags.numel() > 0:
                seq_weights.append(self.class_weights[valid_tags].mean())
            else:
                seq_weights.append(torch.tensor(1.0, device=tags.device))
        seq_weights = torch.stack(seq_weights) 
        loss = (nll * seq_weights).mean()
        return loss

    def predict(self, input_ids, attention_mask, word_first_idx, char_ids, mask):
        emissions = self.forward(input_ids, attention_mask, word_first_idx, char_ids)
        preds = self.crf.decode(emissions, mask=mask)  
        return preds


model = BERT_CharBiGRU_WordBiGRU_CRF(
    bert_model_name=BERT_MODEL,
    char_vocab_size=len(char2idx),
    char_emb_dim=30,
    char_hidden_dim=50,
    word_gru_hidden=128,
    num_tags=len(tag2idx),
    pad_char_idx=char2idx["PAD"],
).to(DEVICE)


all_train_labels_flat = np.concatenate([arr[arr != tag2idx["PAD"]].ravel() for arr in train_label_arrays])
counts = np.bincount(all_train_labels_flat, minlength=len(tag2idx))
inv = np.ones_like(counts, dtype=np.float32)
inv[counts > 0] = 1.0 / counts[counts > 0]
inv = inv / (inv.mean() + 1e-12)
model.class_weights = torch.tensor(inv, dtype=torch.float32).to(DEVICE)


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)


def evaluate_epoch(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            att_mask = batch["attention_mask"].to(device)
            word_first_idx = batch["word_first_idx"].to(device)
            char_ids = batch["char_ids"].to(device)
            labels = batch["labels"].to(device)
            word_mask = batch["word_mask"].to(device).bool()

            preds = model.predict(input_ids, att_mask, word_first_idx, char_ids, mask=word_mask)
            for p_seq, lab_seq, m_seq in zip(preds, labels, word_mask):
                true = [idx2tag[int(x)] for x, mk in zip(lab_seq.tolist(), m_seq.tolist()) if mk]
                pred = [idx2tag[int(x)] for x in p_seq[:len(true)]]
                all_labels.append(true)
                all_preds.append(pred)

    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    return prec, rec, f1, report, all_labels, all_preds


EPOCHS = 40
PATIENCE = 5
best_f1 = -1.0
patience_ctr = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        att_mask = batch["attention_mask"].to(DEVICE)
        word_first_idx = batch["word_first_idx"].to(DEVICE)
        char_ids = batch["char_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        word_mask = batch["word_mask"].to(DEVICE).bool()

        optimizer.zero_grad()
        loss = model.loss(input_ids, att_mask, word_first_idx, char_ids, labels, mask=word_mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    prec, rec, f1, report, all_labels, all_preds = evaluate_epoch(model, val_loader, DEVICE)
    scheduler.step(f1)

    print(f"\nEpoch {epoch} | Train Loss {avg_loss:.4f} | Val P {prec:.4f} R {rec:.4f} F1 {f1:.4f}")
    print(report)


    flat_true = [t for seq in all_labels for t in seq if t != "O"]
    flat_pred = [p for seq in all_preds  for p in seq if p != "O"]
    if len(flat_true) == len(flat_pred) and len(flat_true) > 0:
        labels_cm = sorted(list(set(flat_true + flat_pred)))
        cm = confusion_matrix(flat_true, flat_pred, labels=labels_cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_cm)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, xticks_rotation=45)
        plt.title(f"Confusion matrix (epoch {epoch})")
        plt.show()

   
    if f1 > best_f1:
        best_f1 = f1
        patience_ctr = 0
        torch.save(model.state_dict(), BEST_PATH)
        print("Model improved. Saved checkpoint.")
    else:
        patience_ctr += 1
        print(f"No improvement. Patience {patience_ctr}/{PATIENCE}")
        if patience_ctr >= PATIENCE:
            print("Early stopping triggered.")
            break


model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
model.to(DEVICE)
prec, rec, f1, report, all_labels, all_preds = evaluate_epoch(model, val_loader, DEVICE)
print("\nFinal validation report:\n")
print(report)
print("Final F1:", f1)


def encode_chars_for_sentence_simple(words, max_words=MAX_WORDS, max_word_len=MAX_WORD_LEN):
    chars = []
    for w in words[:max_words]:
        ch_ids = [char2idx.get(c, char2idx["UNK"]) for c in w[:max_word_len]]
        if len(ch_ids) < max_word_len:
            ch_ids += [char2idx["PAD"]] * (max_word_len - len(ch_ids))
        chars.append(ch_ids)
    while len(chars) < max_words:
        chars.append([char2idx["PAD"]] * max_word_len)
    return np.array(chars, dtype=np.int64)

@torch.no_grad()
def predict_sentence(tokens):
    enc = tokenizer([tokens], is_split_into_words=True, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(DEVICE)
    att_mask = enc["attention_mask"].to(DEVICE)
    word_ids = enc.word_ids(batch_index=0)
    first = {}
    for sub_i, w_id in enumerate(word_ids):
        if w_id is None:
            continue
        if w_id not in first:
            first[w_id] = sub_i
    max_words = min(len(tokens), MAX_WORDS)
    word_first_idx = [first.get(i, 0) for i in range(max_words)]
    while len(word_first_idx) < MAX_WORDS:
        word_first_idx.append(0)
    word_first_idx = torch.tensor([word_first_idx], dtype=torch.long).to(DEVICE)
    char_ids = torch.tensor([encode_chars_for_sentence_simple(tokens)], dtype=torch.long).to(DEVICE)
    word_mask = torch.tensor([[1 if i < max_words else 0 for i in range(MAX_WORDS)]], dtype=torch.bool).to(DEVICE)
    preds = model.predict(input_ids, att_mask, word_first_idx, char_ids, mask=word_mask)[0]
    pred_tags = [idx2tag[int(p)] for p in preds[:len(tokens)]]
    return list(zip(tokens, pred_tags))


print(predict_sentence(["Akshay", "is", "working", "in", "Senscript", "located", "in", "Paris", "."]))
