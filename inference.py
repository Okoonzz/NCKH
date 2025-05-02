import json
import torch
from data.sequence_vector import load_report, build_text_event_sequence
from data.vocab import encode_sequence
from model.xlstm_classifier import xLSTMClassifier

# Load vocab từ file và xác định vocab_size
with open('vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)
context_length = 50
vocab_size = len(vocab)

# Khởi tạo model và load checkpoint
model = xLSTMClassifier(vocab_size=vocab_size, embed_dim=128, context_length=context_length)
model.load_state_dict(torch.load('model.pt', map_location='cpu'))
model.eval()

def predict(report_path):
    report = load_report(report_path)
    seq = build_text_event_sequence(report)
    idxs = encode_sequence(seq, vocab, context_length)
    xb = torch.tensor([idxs])
    with torch.no_grad():
        logit = model(xb)
        prob = torch.sigmoid(logit).item()
    return prob

def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'reports/report.json'
    prob = predict(path)
    label = 'RANSOMWARE' if prob >= 0.5 else 'BENIGN'
    print(f"Probability ransomware: {prob:.4f} -> {label}")

if __name__ == '__main__':
    main()