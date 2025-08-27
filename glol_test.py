
import torch
import torch.nn as nn
import pickle

from torchcrf import CRF


BEST_PATH   = "best_glove_bigru_crf.pt"
MAPPING_PATH = "mappings.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_WORD_LEN   = 18     
CHAR_EMB_DIM   = 30
CHAR_HIDDEN    = 50
WORD_HIDDEN    = 128
GLOVE_DIM      = 100   
DROPOUT        = 0.5


with open(MAPPING_PATH, "rb") as f:
    mappings = pickle.load(f)

char2idx = mappings["char2idx"]
tag2idx  = mappings["tag2idx"]
idx2tag  = mappings["idx2tag"]


idx2tag = {int(k): v for k, v in idx2tag.items()}


class GloVe_CharBiGRU_WordBiGRU_CRF(nn.Module):
    def __init__(self, vocab_size, glove_dim, char_vocab_size, char_emb_dim, char_hidden_dim,
                 word_hidden_dim, num_tags, pad_word_idx=0, pad_char_idx=0, dropout=0.5):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, glove_dim, padding_idx=pad_word_idx)

        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=pad_char_idx)
        self.char_gru = nn.GRU(char_emb_dim, char_hidden_dim, batch_first=True, bidirectional=True)

        self.word_gru = nn.GRU(glove_dim + 2*char_hidden_dim, word_hidden_dim,
                               batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*word_hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, word_ids, char_ids):
        w_emb = self.word_emb(word_ids)  

        B, L, C = char_ids.size()
        flat = char_ids.view(B*L, C)
        c_emb = self.char_emb(flat)
        _, h = self.char_gru(c_emb)
        c_repr = torch.cat([h[0], h[1]], dim=-1)  
        c_repr = c_repr.view(B, L, -1)

        x = torch.cat([w_emb, c_repr], dim=-1)
        x = self.dropout(x)

        y, _ = self.word_gru(x)
        y = self.dropout(y)
        emissions = self.fc(y)
        return emissions

    def predict(self, word_ids, char_ids, mask):
        emissions = self.forward(word_ids, char_ids)
        return self.crf.decode(emissions, mask=mask)


vocab_size = mappings.get("vocab_size", 27422)  
num_tags = len(tag2idx)

model = GloVe_CharBiGRU_WordBiGRU_CRF(
    vocab_size=vocab_size,
    glove_dim=GLOVE_DIM,
    char_vocab_size=len(char2idx),
    char_emb_dim=CHAR_EMB_DIM,
    char_hidden_dim=CHAR_HIDDEN,
    word_hidden_dim=WORD_HIDDEN,
    num_tags=num_tags,
    pad_word_idx=0,
    pad_char_idx=0,
    dropout=DROPOUT
).to(DEVICE)

state_dict = torch.load(BEST_PATH, map_location=DEVICE)


if "class_weights" in state_dict:
    print(" Removing 'class_weights' from checkpoint")
    del state_dict["class_weights"]

model.load_state_dict(state_dict, strict=False)
model.eval()
print(" Model and mappings loaded.")

def encode_chars_for_sentence(words, max_word_len=MAX_WORD_LEN):
    chars = []
    for w in words:
        ch = [char2idx.get(c, char2idx["UNK"]) for c in w[:max_word_len]]
        if len(ch) < max_word_len:
            ch += [char2idx["PAD"]] * (max_word_len - len(ch))
        chars.append(ch)
    return torch.tensor([chars], dtype=torch.long)  

@torch.no_grad()
def predict_sentence(words):
    
    word_ids = torch.tensor([[1 for _ in words]], dtype=torch.long).to(DEVICE)

    char_ids = encode_chars_for_sentence(words).to(DEVICE)
    mask = torch.ones((1, len(words)), dtype=torch.bool).to(DEVICE)

    preds = model.predict(word_ids, char_ids, mask=mask)[0]
    tags = [idx2tag[int(p)] for p in preds[:len(words)]]
    return list(zip(words, tags))

if __name__ == "__main__":
    print("\nType a sentence to test NER (or 'quit' to exit)\n")
    while True:
        text = input(">>> ")
        if text.strip().lower() in ["quit", "exit"]:
            break
        words = text.strip().split()
        result = predict_sentence(words)
        print(result)
