import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
import torch.nn as nn
import pickle

BEST_PATH = "best_bert_bigru_crf.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_MODEL = "distilbert-base-uncased"
MAX_WORDS = 140
MAX_WORD_LEN = 18

with open("mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

if isinstance(mappings, dict):
    char2idx = mappings["char2idx"]
    tag2idx = mappings["tag2idx"]
    idx2tag = mappings["idx2tag"]
else:
    char2idx, tag2idx, idx2tag = mappings


if "PAD" not in char2idx:
    char2idx["PAD"] = len(char2idx)
if "UNK" not in char2idx:
    char2idx["UNK"] = len(char2idx)


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


state_dict = torch.load(BEST_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.eval()
print(" Model loaded successfully!")


tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

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

    
    word_mask = torch.tensor([[1 if i < max_words else 0 for i in range(MAX_WORDS)]],
                             dtype=torch.bool).to(DEVICE)

    preds = model.predict(input_ids, att_mask, word_first_idx, char_ids, mask=word_mask)[0]
    pred_tags = [idx2tag[int(p)] for p in preds[:len(tokens)]]
    return list(zip(tokens, pred_tags))


sentence = "Mahatma Gandhi was born in India and he is a friend of Martin Luther King . He was born on 2nd October 1869 in Porbandar . " 
tokens = sentence.split()
print(predict_sentence(tokens))


#The Olympics will be held in Tokyo at 8 am inagrauted by Micheal Jackson