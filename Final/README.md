# Ransomware Analyzer (Streamlit)

UI pipeline end-to-end:
1) Upload **Cuckoo report JSON** (or upload `.exe` -> submit to Cuckoo API).
2) Extract features -> build GraphML -> build PyG graph + sequence ids.
3) Inference with **xLSTM + GCN + gated fusion**.
4) XAI: **LIME** (sequence) + **GNNExplainer** (graph).
5) Optional LLM report (OpenAI-compatible).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Artifacts you must provide
Put your trained artifacts under `artifacts/` (or change paths in sidebar):
- `artifacts/best_mm_da_seq1500.pt`
- `artifacts/seq_vocab.json`
- `artifacts/graph_vocab.json`

## Cuckoo
If you want `.exe` upload:
- Set `CUCKOO_API` in sidebar (e.g., `http://192.168.111.158:8090`)
- Ensure your Streamlit machine can reach the Cuckoo API.

## Notes
- `torch-geometric` installation depends on your OS/Python/PyTorch (may need special wheels).
