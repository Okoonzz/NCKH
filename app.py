# app.py
import streamlit as st
import tempfile, json, os, time, pathlib, requests, torch
from torch_geometric.data import Batch
from torch_geometric.utils import add_self_loops
import torch

# ---- các module ----
from tmp import extract_features
from test import build_ransomware_graph, sanitize_and_save_graph
from gen_pt import graphml_to_pyg_data
from multimodal import GCNEncoder, xLSTMEncoder, MultiModalClassifier
# ----------------------------------

# streamlit run app.py

# ---- cấu hình chung ----
CUCKOO_API = "http://192.168.111.158:8090"
SEQ_LEN, API_LIMIT = 2000, 1000
THRESH = 0.5                         # ngưỡng phân loại
USE_GPU = True                      # đổi True nếu đã set CUDA_HOME
DEVICE  = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def load_vocab_model():
    vocab = json.load(open("vocab.json", encoding="utf-8"))
    genc  = GCNEncoder(in_feats=768).to(DEVICE)
    senc  = xLSTMEncoder(vocab_size=len(vocab),
                         embed=128, seq_len=SEQ_LEN, blocks=1).to(DEVICE)
    model = MultiModalClassifier(genc, senc).to(DEVICE)
    model.load_state_dict(torch.load("best_model_api1000_seq2000.pth",
                                     map_location=DEVICE))
    model.eval()
    return vocab, model

def submit_cuckoo(buf, filename):
    files = {"file": (filename, buf)}
    r = requests.post(f"{CUCKOO_API}/tasks/create/file", files=files)
    r.raise_for_status()
    return r.json()["task_id"]

def wait_cuckoo_report(tid):
    while True:
        r = requests.get(f"{CUCKOO_API}/tasks/view/{tid}"); r.raise_for_status()
        if r.json()["task"]["status"] == "reported": return
        time.sleep(5)

def download_report(tid):
    r = requests.get(f"{CUCKOO_API}/tasks/report/{tid}"); r.raise_for_status()
    return r.json()

def tokens_from_feat(fj):
    toks = [f"api:{c['api']}" for c in fj.get("api_call_sequence", [])[:API_LIMIT]]
    for ft, vs in fj.get("behavior_summary", {}).items():
        toks += [f"feature:{ft}:{v}" for v in vs]
    toks += [f"dropped_file:{d}" for d in fj.get("dropped_files", [])]
    toks += [f"signature:{s['name']}" for s in fj.get("signatures", [])]
    toks += [f"process:{p['name']}" for p in fj.get("processes", [])]
    for proto, ents in fj.get("network", {}).items():
        for e in ents:
            dst = (e.get("dst") or e.get("dst_ip", "")) if isinstance(e, dict) else str(e)
            port = (e.get("dst_port") or e.get("port", "")) if isinstance(e, dict) else ""
            toks.append(f"network:{proto}:{dst}:{port}")
    return toks

def build_pyg(feat_path):
    with tempfile.TemporaryDirectory() as td:
        gml = pathlib.Path(td) / "tmp.graphml"
        G   = build_ransomware_graph(str(feat_path))
        sanitize_and_save_graph(G, str(gml))
        data = graphml_to_pyg_data(str(gml))

    # --- Vá: đảm bảo edge_index có dạng (2, E) ---
    if data.edge_index.numel() == 0:               # hoàn toàn không có cạnh
        n = data.num_nodes
        if n == 0:
            raise ValueError("Graph không có node nào!")
        # tạo self-loop cho mỗi node: (0..n-1) → (0..n-1)
        loops = torch.arange(n, dtype=torch.long)
        data.edge_index = torch.stack([loops, loops], dim=0)
    elif data.edge_index.dim() == 1:               # shape (E,) thành (2,E)
        E = data.edge_index.size(0)
        data.edge_index = data.edge_index.unsqueeze(0).repeat(2, 1)

    return data

def infer(feat_json, feat_path, vocab, model):
    # PyG Data
    pyg_data = build_pyg(feat_path)
    batch    = Batch.from_data_list([pyg_data]).to(DEVICE)
    # seq tensor
    idxs = [vocab.get(t, vocab["<UNK>"]) for t in tokens_from_feat(feat_json)]
    idxs = (idxs + [vocab["<PAD>"]] * SEQ_LEN)[:SEQ_LEN]
    seq  = torch.tensor(idxs, dtype=torch.long, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(batch, seq)).item()
    return prob, int(prob > THRESH)

# ================= Streamlit UI =================
st.set_page_config(page_title="Ransomware Analyzer", page_icon="🛡️", layout="centered")
st.title("🛡️ Ransomware Analyzer")

analysis = st.selectbox("Chọn loại phân tích:", ["Ransomware and Benign"])
filetype = st.selectbox("Chọn kiểu file đầu vào:", ["report (Cuckoo JSON)", "executable (.exe)"])

uploaded = st.file_uploader("Tải lên file", type=["json", "exe"], accept_multiple_files=False)

if uploaded:
    vocab, model = load_vocab_model()
    with st.spinner("Đang xử lý..."):
        if filetype.startswith("report"):
            # ----------- report JSON trực tiếp -----------
            feat_json = json.load(uploaded)
        else:
            # ----------- upload exe -> cuckoo ------------
            tid = submit_cuckoo(uploaded, uploaded.name)
            st.info(f"Đã gửi Cuckoo task {tid}, đang chờ ...")
            wait_cuckoo_report(tid)
            raw_json  = download_report(tid)
            reports_dir = pathlib.Path("cuckoo_reports")
            reports_dir.mkdir(exist_ok=True)
            raw_path = reports_dir / f"report_{uploaded.name}.json"
            raw_path.write_text(json.dumps(raw_json, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            feat_json = extract_features(raw_json)

        # ghi ra tệp tạm cho hàm graph
        with tempfile.NamedTemporaryFile(
                suffix=".json",
                delete=False,
                mode="w",
                encoding="utf-8") as tf:
            json.dump(feat_json, tf)
            tmp_path = pathlib.Path(tf.name)
        prob, pred = infer(feat_json, tmp_path, vocab, model)
        os.unlink(tmp_path)

    if pred == 1:
        st.warning(f"⚠️  KẾT QUẢ: **Ransomware**  (xác suất = {prob:.4f})")
    else:
        st.success(f"✅  KẾT QUẢ: **Benign**  (xác suất = {prob:.4f})")
    st.json({"probability": prob, "label": "ransomware" if pred else "benign"})
    st.download_button(
        "Tải report JSON",
        data=json.dumps(raw_json, ensure_ascii=False, indent=2),
        file_name=f"report_{uploaded.name}.json",
        mime="application/json"
    )
