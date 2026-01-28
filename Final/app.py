# app.py
import os
import io
import json
import hashlib
import re
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import streamlit as st
import pandas as pd

import torch
from torch_geometric.data import Batch

# ====== IMPORT directly from modules/ ======
from modules import extract_features as m_extract
from modules import build_graphml as m_graphml
from modules import gen_pt_seq as m_seq
from modules import gen_pt_label as m_label
from modules import model as m_model
from modules import xai_llm as m_xai


# =========================
# Paths (artifacts/)
# =========================
ART_DIR = Path("artifacts")
DEFAULT_MODEL_PT      = ART_DIR / "best_mm_da_seq1500.pt"
DEFAULT_SEQ_VOCAB     = ART_DIR / "seq_vocab.json"
DEFAULT_GRAPH_VOCAB   = ART_DIR / "graph_vocab.json"
DEFAULT_GRAPH_ID2TOK  = ART_DIR / "graph_vocab_id2token.json"

SEQ_LEN = 1500


# =========================
# Small helpers
# =========================
def _safe_json_load(uploaded) -> Dict[str, Any]:
    raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
    return json.loads(raw.decode("utf-8", errors="replace"))


def _risk_from_prob(p: float) -> Tuple[str, str]:
    # Returns (risk_level, emoji)
    if p < 0.40:
        return "low", "🟢"
    if p < 0.70:
        return "medium", "🟡"
    if p < 0.90:
        return "high", "🟠"
    return "critical", "🔴"

def _extract_json_fenced_block(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Extract the first ```json ... ``` fenced block from text.
    Returns (parsed_json_or_None, remaining_text_without_that_block).
    """
    if not text:
        return None, ""

    # Prefer fenced JSON
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        # Fallback: any fenced block
        m = re.search(r"```\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if not m:
        return None, text

    json_str = m.group(1)
    rest = (text[:m.start()] + text[m.end():]).strip()

    try:
        return json.loads(json_str), rest
    except Exception:
        return None, text

def _render_bullets(items: List[str]) -> None:
    if not items:
        st.write("—")
        return
    st.markdown("\n".join([f"- {str(x)}" for x in items]))

def _sigmoid_prob(logit_1d: torch.Tensor) -> float:
    return float(torch.sigmoid(logit_1d).detach().cpu().item())

def _infer_in_feats_from_ckpt(ckpt_obj: Any, fallback: int = 768) -> int:
    sd = None
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                sd = ckpt_obj[k]
                break
        if sd is None and all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            sd = ckpt_obj
    if sd is None:
        return fallback
    for key in ["graph_enc.conv1.lin.weight", "module.graph_enc.conv1.lin.weight"]:
        if key in sd and hasattr(sd[key], "shape"):
            return int(sd[key].shape[1])
    for k, v in sd.items():
        if k.endswith("graph_enc.conv1.lin.weight") and hasattr(v, "shape"):
            return int(v.shape[1])
    return fallback

def make_jsonable(obj):
    # numpy
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)

    # torch
    if torch.is_tensor(obj):
        if obj.ndim == 0:
            return float(obj.detach().cpu().item())
        return obj.detach().cpu().tolist()

    # containers
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]

    return obj

def add_download(downloads: List[Dict[str, Any]], label: str, data: Any, file_name: str, mime: str):
    if data is None:
        return
    if isinstance(data, (bytes, bytearray)):
        b = bytes(data)
    else:
        b = str(data).encode("utf-8", errors="replace")
    downloads.append({"label": label, "data": b, "file_name": file_name, "mime": mime})

def _dominant_branch(
    w_seq: float, w_graph: float,
    prob_seq_only: float, prob_graph_only: float
) -> Tuple[str, str, str, float, float]:
    dominant_gate = "seq" if w_seq >= w_graph else "graph"
    conf_seq = abs(prob_seq_only - 0.5)
    conf_graph = abs(prob_graph_only - 0.5)
    dominant_ablation = "seq" if conf_seq >= conf_graph else "graph"

    if dominant_gate == dominant_ablation:
        dominant_final = dominant_gate
    else:
        dominant_final = f"mixed(gate={dominant_gate}, ablation={dominant_ablation})"
    return dominant_gate, dominant_ablation, dominant_final, conf_seq, conf_graph



def _extract_hash_from_report(raw_report: Dict[str, Any]) -> Optional[str]:
    """Try to extract a file hash embedded in a Cuckoo report (sha256/sha1/md5)."""
    if not isinstance(raw_report, dict):
        return None

    def _get_path(d: Any, keys: Tuple[str, ...]) -> Any:
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    candidate_paths = [
        ("target", "file", "sha256"),
        ("target", "file", "sha1"),
        ("target", "file", "md5"),
        ("file", "sha256"),
        ("file", "sha1"),
        ("file", "md5"),
        ("sha256",),
        ("sha1",),
        ("md5",),
    ]

    for keys in candidate_paths:
        v = _get_path(raw_report, keys)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return None

# =========================
# Cuckoo helpers
# =========================
def cuckoo_submit_file(api_base: str, exe_path: str) -> int:
    import requests
    url = api_base.rstrip("/") + "/tasks/create/file"
    with open(exe_path, "rb") as f:
        r = requests.post(url, files={"file": f}, timeout=60)
    r.raise_for_status()
    j = r.json()
    if "task_id" in j:
        return int(j["task_id"])
    if "task_ids" in j and j["task_ids"]:
        return int(j["task_ids"][0])
    raise RuntimeError(f"Unexpected submit response: {j}")

def cuckoo_wait_report(api_base: str, task_id: int, poll_s: float, timeout_s: int):
    import requests
    url_view = api_base.rstrip("/") + f"/tasks/view/{task_id}"
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError("Cuckoo timeout waiting for report")
        r = requests.get(url_view, timeout=30)
        r.raise_for_status()
        j = r.json()
        status = (j.get("task") or {}).get("status")
        if status == "reported":
            return
        time.sleep(poll_s)

def cuckoo_get_report_json(api_base: str, task_id: int) -> Dict[str, Any]:
    import requests
    url = api_base.rstrip("/") + f"/tasks/report/{task_id}/json"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.json()


# =========================
# Cached load: model + vocabs
# =========================
@st.cache_resource
def load_artifacts(
    model_pt_path: str,
    seq_vocab_path: str,
    graph_vocab_path: str,
    graph_id2tok_path: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok2id, id2tok, pad_id, unk_id = m_xai.load_seq_vocab(seq_vocab_path)
    graph_id2tok = m_xai.load_graph_id2token(graph_id2tok_path, graph_vocab_path)

    ckpt_obj = torch.load(model_pt_path, map_location="cpu", weights_only=False)
    in_feats = _infer_in_feats_from_ckpt(ckpt_obj, fallback=768)

    graph_enc = m_model.GCNEncoder(in_feats=in_feats, hidden=64, drop=0.3).to(device)
    seq_enc   = m_model.xLSTMEncoder(vocab_size=len(tok2id), embed=128, seq_len=SEQ_LEN, blocks=1).to(device)
    model     = m_model.MultiModalClassifier(graph_enc, seq_enc, fusion_h=128).to(device)

    try:
        m_xai.robust_load_state_dict(model, model_pt_path, device)
    except Exception:
        sd = ckpt_obj
        if isinstance(ckpt_obj, dict):
            for k in ["state_dict", "model_state_dict", "model"]:
                if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                    sd = ckpt_obj[k]
                    break
        if isinstance(sd, dict):
            sd2 = {}
            for k, v in sd.items():
                new_k = k[7:] if k.startswith("module.") else k
                sd2[new_k] = v
            model.load_state_dict(sd2, strict=False)

    model.eval()
    return {
        "device": device,
        "model": model,
        "tok2id": tok2id,
        "id2tok": id2tok,
        "pad_id": int(pad_id),
        "unk_id": int(unk_id),
        "graph_id2tok": graph_id2tok,
        "in_feats": int(in_feats),
        "d_seq": int(seq_enc.output_dim),
        "d_graph": int(graph_enc.output_dim),
    }


# =========================
# Branch-only probs (not depending on gate)
# =========================
def branch_only_probs(model, g_batch: Batch, s_batch: torch.Tensor) -> Tuple[float, float]:
    with torch.no_grad():
        h_g = model.graph_enc(g_batch.x, g_batch.edge_index, g_batch.batch)
        h_s = model.seq_enc(s_batch)

        B = min(h_g.size(0), h_s.size(0))
        h_g = h_g[:B]
        h_s = h_s[:B]

        zeros_g = torch.zeros_like(h_g)
        zeros_s = torch.zeros_like(h_s)

        out_seq = model.classifier(torch.cat([h_s, zeros_g], dim=-1))
        if isinstance(out_seq, (tuple, list)):
            out_seq = out_seq[0]
        out_seq = out_seq.reshape(-1)

        out_graph = model.classifier(torch.cat([zeros_s, h_g], dim=-1))
        if isinstance(out_graph, (tuple, list)):
            out_graph = out_graph[0]
        out_graph = out_graph.reshape(-1)

        prob_seq_only   = _sigmoid_prob(out_seq[0])
        prob_graph_only = _sigmoid_prob(out_graph[0])

    return prob_seq_only, prob_graph_only


# ======================================================
# Streamlit UI
# ======================================================
st.set_page_config(page_title="Ransomware Analyzer", page_icon="🛡️", layout="wide")
st.title("🛡️ Ransomware Analyzer with Explainable AI and LLM")

# Keep downloads across reruns
if "downloads" not in st.session_state:
    st.session_state["downloads"] = []

with st.sidebar:
    st.header("Artifacts (default: artifacts/)")
    model_pt = st.text_input("Model (.pt)", str(DEFAULT_MODEL_PT))
    seq_vocab = st.text_input("seq_vocab.json", str(DEFAULT_SEQ_VOCAB))
    graph_vocab = st.text_input("graph_vocab.json", str(DEFAULT_GRAPH_VOCAB))
    graph_id2tok = st.text_input("graph_id2token.json", str(DEFAULT_GRAPH_ID2TOK))

    st.divider()
    st.header("Option")
    run_lime = st.checkbox("Run LIME (sequence)", value=True)
    run_gnn = st.checkbox("Run GNNExplainer (graph)", value=True)
    enable_llm = st.checkbox("Create LLM report", value=False)

    llm_model = st.text_input("LLM model", value=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"), disabled=not enable_llm)
    try:
        openai_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        openai_key = os.getenv("OPENAI_API_KEY", "")

    st.divider()
    st.header("Cuckoo (for .exe upload)")
    cuckoo_api = st.text_input("CUCKOO_API", value=os.getenv("CUCKOO_API", "http://127.0.0.1:8090"))
    cuckoo_timeout = st.number_input("Timeout (seconds)", 60, 3600, 600, 30)
    cuckoo_poll = st.number_input("Poll interval (seconds)", 1, 30, 3, 1)

# Load artifacts
try:
    art = load_artifacts(model_pt, seq_vocab, graph_vocab, graph_id2tok)
except Exception as e:
    st.error(f"Can not load artifacts: {e}")
    st.stop()

device = art["device"]
model = art["model"]

analysis = st.selectbox("Analysis type:", ["Ransomware vs Benign"])
filetype = st.selectbox("Input type:", ["Cuckoo report (JSON)", "Executable (.exe)"])

uploaded = st.file_uploader("Upload file", type=["json", "exe"], accept_multiple_files=False)
run_btn = st.button("🔍 Analyze", type="primary", disabled=(uploaded is None))

if run_btn and uploaded is not None:
    # Reset downloads for this run
    downloads: List[Dict[str, Any]] = []
    st.session_state["downloads"] = []

    tmpdir = Path(tempfile.mkdtemp(prefix="rw_app_"))
    out_dir = tmpdir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # 1) Get RAW report
    # -----------------------
    raw_report: Optional[Dict[str, Any]] = None
    sample_name = Path(uploaded.name).stem

    file_bytes = uploaded.getvalue()

    # Fallback hash from uploaded bytes; if report contains a hash, we will prefer it below.
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    with st.spinner("Waiting..."):
        is_report = (filetype == "Cuckoo report (JSON)") or uploaded.name.lower().endswith(".json")
        if is_report:
            raw_report = _safe_json_load(uploaded)
        else:
            exe_path = tmpdir / uploaded.name
            exe_path.write_bytes(file_bytes)
            tid = cuckoo_submit_file(cuckoo_api, str(exe_path))
            st.info(f"Sent to Cuckoo: task **{tid}**, waiting for report...")
            cuckoo_wait_report(cuckoo_api, tid, float(cuckoo_poll), int(cuckoo_timeout))
            raw_report = cuckoo_get_report_json(cuckoo_api, tid)

    # Prefer hash embedded in the Cuckoo report, if available
    report_hash = _extract_hash_from_report(raw_report)
    if report_hash:
        file_hash = report_hash

    add_download(
        downloads,
        "⬇️ Download RAW report JSON",
        json.dumps(raw_report, ensure_ascii=False, indent=2).encode("utf-8"),
        f"raw_report_{sample_name}.json",
        "application/json",
    )

    # -----------------------
    # 2) Extract features
    # -----------------------
    with st.spinner("Extracting features from raw report..."):
        features = m_extract.extract_features(raw_report)

    feat_path = tmpdir / f"{sample_name}_features.json"
    feat_path.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")

    add_download(
        downloads,
        "⬇️ Download extracted features JSON",
        feat_path.read_bytes(),
        feat_path.name,
        "application/json",
    )

    with st.expander("Preview extracted features", expanded=False):
        st.json(features)

    # -----------------------
    # 3) Build graphml
    # -----------------------
    with st.spinner("Building GraphML..."):
        G = m_graphml.build_ransomware_graph(str(feat_path))
        graphml_path = tmpdir / f"{sample_name}.graphml"
        if hasattr(m_graphml, "sanitize_and_save_graph"):
            m_graphml.sanitize_and_save_graph(G, str(graphml_path))
        else:
            import networkx as nx
            nx.write_graphml(G, str(graphml_path))

    add_download(
        downloads,
        "⬇️ Download GraphML",
        graphml_path.read_bytes(),
        graphml_path.name,
        "application/xml",
    )

    # -----------------------
    # 4) GraphML -> PyG Data
    # -----------------------
    with st.spinner("Converting GraphML -> PyG Data..."):
        data = m_label.graphml_to_pyg_data(str(graphml_path))

    has_real_graph = True
    if (not hasattr(data, "x")) or (data.x is None) or (data.x.dim() != 2) or (data.x.size(0) == 0):
        has_real_graph = False
        data.x = torch.zeros((1, art["in_feats"]), dtype=torch.float32)
        data.edge_index = torch.empty((2, 0), dtype=torch.long)

    # -----------------------
    # 5) Features -> seq_ids
    # -----------------------
    with st.spinner("Building sequence ids..."):
        tokens = m_seq.build_tokens_from_features(features)
        seq_ids_list = m_xai.encode_tokens_to_ids(
            tokens,
            art["tok2id"],
            pad_id=art["pad_id"],
            unk_id=art["unk_id"],
            max_len=SEQ_LEN
        )
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.long)

    # -----------------------
    # 6) Inference + gate + branch-only
    # -----------------------
    g = Batch.from_data_list([data]).to(device)
    s = seq_ids.unsqueeze(0).to(device)

    g.has_graph = torch.tensor([1 if has_real_graph else 0], dtype=torch.long, device=device)

    with st.spinner("Inference..."):
        with torch.no_grad():
            logits, h_fused, br = model(g, s, return_branch=True, return_features=True)
            prob_full = _sigmoid_prob(logits[0])
            pred_full = 1 if prob_full >= 0.5 else 0

        w_seq = float(br["w_seq"][0].item())
        w_graph = float(br["w_graph"][0].item())

        prob_seq_only, prob_graph_only = branch_only_probs(model, g, s)
        dominant_gate, dominant_ablation, dominant_final, conf_seq, conf_graph = _dominant_branch(
            w_seq, w_graph, prob_seq_only, prob_graph_only
        )

    st.subheader("✅ Prediction")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pred", "ransomware" if pred_full == 1 else "benign")
    col2.metric("Prob(ransomware)", f"{prob_full:.6f}")
    col3.metric("Gate (w_seq / w_graph)", f"{w_seq:.3f} / {w_graph:.3f}")
    col4.metric("Branch stronger", dominant_gate)

    st.caption(
        f"Dominant(gate)={dominant_gate} | Dominant(ablation)={dominant_ablation} | "
        f"prob_seq_only={prob_seq_only:.4f} (conf={conf_seq:.3f}) | "
        f"prob_graph_only={prob_graph_only:.4f} (conf={conf_graph:.3f})"
    )
    # --- Visual risk indicator (easier to scan) ---
    risk_level, risk_emoji = _risk_from_prob(prob_full)
    with st.container(border=True):
        st.markdown(f"### {risk_emoji} Overall risk: **{risk_level.upper()}**")
        st.progress(min(max(float(prob_full), 0.0), 1.0))
        st.caption("Risk is derived from the model probability and supporting evidence; review details below.")



        result_df = pd.DataFrame([{
            "file": uploaded.name,
            "pred_label": pred_full,
            "pred_name": "ransomware" if pred_full == 1 else "benign",
            "prob_full": prob_full,
            "w_seq": w_seq,
            "w_graph": w_graph,
            "prob_seq_only": prob_seq_only,
            "prob_graph_only": prob_graph_only,
            "dominant_gate": dominant_gate,
            "dominant_ablation": dominant_ablation,
            "dominant_final": dominant_final,
            "has_graph": int(has_real_graph),
            "n_seq_tokens": len(tokens),
            "n_graph_nodes": int(data.num_nodes) if hasattr(data, "num_nodes") else None,
            "n_graph_edges": int(data.edge_index.size(1)) if hasattr(data, "edge_index") else None,
        }])

        add_download(
            downloads,
            "⬇️ Download overview report (CSV)",
            result_df.to_csv(index=False).encode("utf-8"),
            f"result_{sample_name}.csv",
            "text/csv",
        )

        # -----------------------
    
    # -----------------------
    # 7-8) Explainability (XAI)
    # -----------------------
    st.subheader("🔎 Explainability (XAI)")
    tab_seq, tab_graph = st.tabs(["Sequence (LIME)", "Graph (GNNExplainer)"])

    with tab_seq:
            # 7) LIME
                # -----------------------
                lime_benign, lime_ransom, instance_tokens = [], [], []
                top5_b, top5_r = [], []
                if run_lime:
                    st.subheader("🔎 XAI — LIME (Sequence)")
                    try:
                        with st.spinner("Running LIME..."):
                            lime_benign, lime_ransom, instance_tokens = m_xai.explain_seq_lime_both(
                                model=model,
                                ds_item_seq_ids=seq_ids,
                                id2tok_seq=art["id2tok"],
                                tok2id_seq=art["tok2id"],
                                feat_dim=art["in_feats"],
                                max_len=SEQ_LEN,
                                pad_id=art["pad_id"],
                                unk_id=art["unk_id"],
                                device=device,
                                top_k=20,
                                num_samples=600,
                                batch_size=16,
                                max_tokens_for_text=500
                            )

                        top5_b = m_xai.topk_push_to_class(lime_benign, k=5)
                        top5_r = m_xai.topk_push_to_class(lime_ransom, k=5)

                        cL, cR = st.columns(2)
                        with cL:
                            st.write("Top tokens → **Benign**")
                            st.table([{"token": t, "weight": float(w)} for t, w in top5_b])
                        with cR:
                            st.write("Top tokens → **Ransomware**")
                            st.table([{"token": t, "weight": float(w)} for t, w in top5_r])

                        import matplotlib.pyplot as plt
                        orig_show = plt.show
                        plt.show = lambda *args, **kwargs: None
                        m_xai.plot_lime_top5_two_sides(
                            top5_b, top5_r,
                            p_benign=(1.0 - prob_seq_only),
                            p_ransom=prob_seq_only,
                            k=5,
                            save_path=None,
                            sample_name=sample_name
                        )
                        fig = plt.gcf()
                        plt.show = orig_show
                        st.pyplot(fig, clear_figure=True)

                        df_lb = pd.DataFrame(lime_benign, columns=["token", "weight"])
                        df_lr = pd.DataFrame(lime_ransom, columns=["token", "weight"])
                        add_download(downloads, "⬇️ Download LIME benign CSV", df_lb.to_csv(index=False).encode("utf-8"), f"{sample_name}_lime_benign.csv", "text/csv")
                        add_download(downloads, "⬇️ Download LIME ransomware CSV", df_lr.to_csv(index=False).encode("utf-8"), f"{sample_name}_lime_ransomware.csv", "text/csv")

                    except Exception as e:
                        st.warning(f"LIME failed: {e}")

                # -----------------------
    


    with tab_graph:
            # 8) GNNExplainer
                # -----------------------
                gnn_out = {}
                top_nodes_named, top_edges_named = [], []

                if run_gnn:
                    st.subheader("🧠 XAI — GNNExplainer (Graph)")

                    try:
                        with st.spinner("Running GNNExplainer..."):      
                            with torch.enable_grad():          
                                model.eval()
                                model.zero_grad(set_to_none=True)

                                gnn_out = m_xai.explain_graph_gnn(
                                    model=model,
                                    data_single=data,
                                    seq_len=SEQ_LEN,
                                    epochs=120,
                                    top_ratio=0.10,
                                    device=device
                                )

                    except Exception as e:
                        st.warning(f"GNNExplainer failed (explaining): {e}")
                        gnn_out = {}

                    finally:
                        # Trả model về trạng thái sạch cho lần chạy tiếp theo
                        model.eval()
                        model.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()


                    if gnn_out:
                        graph_id2tok_map = art["graph_id2tok"]

                        def idx2name_fn(node_idx: int) -> str:
                            try:
                                vid = int(data.node_ids[int(node_idx)].item())
                                return graph_id2tok_map.get(vid, f"VID_{vid}")
                            except Exception:
                                return f"node_{int(node_idx)}"

                        try:
                            for nidx in (gnn_out.get("top_nodes") or []):
                                nidx_i = int(nidx)
                                score = float(gnn_out["node_score"][nidx_i]) if "node_score" in gnn_out else 0.0
                                top_nodes_named.append((nidx_i, idx2name_fn(nidx_i), score))

                            top_edge_indices = (gnn_out.get("top_edge_indices") or [])
                            edge_mask = gnn_out.get("edge_mask", None)

                            for r, (u, v) in enumerate(gnn_out.get("top_edges") or [], 1):
                                eidx = int(top_edge_indices[r - 1]) if (r - 1) < len(top_edge_indices) else None
                                emask = 0.0
                                if eidx is not None and edge_mask is not None and len(edge_mask):
                                    emask = float(edge_mask[eidx])
                                top_edges_named.append({
                                    "rank": r,
                                    "src_idx": int(u), "dst_idx": int(v),
                                    "src_name": idx2name_fn(int(u)),
                                    "dst_name": idx2name_fn(int(v)),
                                    "edge_mask": emask,
                                })
                        except Exception as e:
                            st.warning(f"Graph post-processing failed: {e}")

                        df_g_nodes = pd.DataFrame(top_nodes_named, columns=["node_idx", "node_name", "score"])
                        df_g_edges = pd.DataFrame(top_edges_named)

                        st.markdown("**Top nodes (named)**")
                        st.dataframe(df_g_nodes.head(50), use_container_width=True)

                        st.markdown("**Top edges (named)**")
                        st.dataframe(df_g_edges.head(50), use_container_width=True)

                        add_download(downloads, "⬇️ Download Graph nodes CSV", df_g_nodes.to_csv(index=False).encode("utf-8"), f"{sample_name}_graph_nodes.csv", "text/csv")
                        add_download(downloads, "⬇️ Download Graph edges CSV", df_g_edges.to_csv(index=False).encode("utf-8"), f"{sample_name}_graph_edges.csv", "text/csv")

                        try:
                            import matplotlib.pyplot as plt
                            orig_show = plt.show
                            plt.show = lambda *args, **kwargs: None

                            out_png = out_dir / f"{sample_name}_gnn.png"
                            m_xai.draw_gnnexplainer_subgraph_with_table(
                                data_i=data,
                                gnn_out=gnn_out,
                                idx2name_fn=idx2name_fn,
                                save_path=str(out_png),
                                title=f"GNNExplainer — {sample_name}",
                                name_max_len=None,
                                max_table_rows=60
                            )

                            plt.show = orig_show

                            if out_png.exists():
                                st.image(str(out_png), caption="GNNExplainer subgraph + table", use_container_width=True)
                                add_download(downloads, "⬇️ Download GNN image (PNG)", out_png.read_bytes(), out_png.name, "image/png")

                        except Exception as e:
                            st.warning(f"GNN drawing failed (but CSV is available): {e}")

                # -----------------------
    
            # -----------------------
    # 9) LLM (General + Technical)
    # -----------------------
    if enable_llm:
        st.subheader("📝 LLM reports (General + Technical)")

        if not openai_key:
            st.warning("LLM is enabled but OPENAI_API_KEY is missing.")
        else:
            os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["LLM_MODEL_NAME"] = llm_model

            try:
                payload = {
                    "file_info": {
                        "name": sample_name,
                        "sha256": file_hash,
                    },
                    "model_output": {
                        "pred_label": int(pred_full),
                        "pred_name": "ransomware" if pred_full == 1 else "benign",
                        "prob_full": float(prob_full),
                        "w_seq": float(w_seq),
                        "w_graph": float(w_graph),
                        "prob_seq_only": float(prob_seq_only),
                        "prob_graph_only": float(prob_graph_only),
                        "conf_seq": float(conf_seq),
                        "conf_graph": float(conf_graph),
                        "has_graph": int(has_real_graph),
                        "n_graph_nodes": int(data.num_nodes),
                        "n_graph_edges": int(data.edge_index.size(1)),
                    },
                    "sequence_lime": {
                        "push_benign": lime_benign[:20] if 'lime_benign' in locals() and lime_benign else [],
                        "push_ransomware": lime_ransom[:20] if 'lime_ransom' in locals() and lime_ransom else [],
                        "top5_benign": top5_b if 'top5_b' in locals() else [],
                        "top5_ransomware": top5_r if 'top5_r' in locals() else [],
                    },
                    "graph": {
                        "top_nodes_named": top_nodes_named[:25] if 'top_nodes_named' in locals() and top_nodes_named else [],
                        "top_edges_named": top_edges_named[:25] if 'top_edges_named' in locals() and top_edges_named else [],
                    }
                }

                payload_clean = make_jsonable(payload)

                # Call 2 prompts (kept separate in modules/xai_llm.py)
                with st.spinner("Generating General-user report..."):
                    general_out = m_xai.generate_llm_general_user_report(payload_clean, model_name=llm_model, temperature=0.2)

                with st.spinner("Generating Technical report..."):
                    technical_md = m_xai.generate_llm_markdown_explanation(payload_clean, model_name=llm_model, temperature=0.2)

                # --- Layout: side-by-side, both visible ---
                left, right = st.columns([1, 1], gap="large")

                # GENERAL
                with left:
                    st.markdown("## 👤 General user view")
                    general_json, general_md = _extract_json_fenced_block(general_out)

                    if isinstance(general_json, dict):
                        with st.container(border=True):
                            rl = str(general_json.get("risk_level", "")).lower().strip()
                            hd = str(general_json.get("headline", "")).strip()
                            cf = general_json.get("confidence", None)

                            st.markdown(f"### {hd}" if hd else "### Summary")
                            c1, c2 = st.columns(2)
                            c1.metric("Risk level", rl.upper() if rl else "—")
                            if isinstance(cf, (int, float)):
                                c2.metric("Confidence", f"{float(cf):.3f}")
                            else:
                                c2.metric("Confidence", "—")

                            wt = str(general_json.get("what_it_means", "")).strip()
                            if wt:
                                st.markdown("**What this means**")
                                st.write(wt)

                    else:
                        st.warning("Could not parse the General JSON block. Showing raw output below.")
                        st.markdown(general_out)

                    if isinstance(general_json, dict):
                        with st.container(border=True):
                            st.markdown("### Top signals")
                            _render_bullets(general_json.get("top_signals", []))

                        with st.container(border=True):
                            st.markdown("### What you should do now")
                            _render_bullets(general_json.get("next_steps", []))

                        with st.expander("More (when to seek help + limitations)", expanded=False):
                            st.markdown("**When to seek help**")
                            _render_bullets(general_json.get("when_to_seek_help", []))
                            lim = str(general_json.get("limitations", "")).strip()
                            if lim:
                                st.markdown("**Limitations**")
                                st.write(lim)

                    if general_md:
                        with st.expander("General markdown report", expanded=False):
                            st.markdown(general_md)

                    add_download(downloads, "⬇️ Download General (raw)", general_out.encode("utf-8"), f"{sample_name}_llm_general.txt", "text/plain")
                    if isinstance(general_json, dict):
                        add_download(downloads, "⬇️ Download General (json)", json.dumps(general_json, ensure_ascii=False, indent=2).encode("utf-8"), f"{sample_name}_llm_general.json", "application/json")

                # TECHNICAL
                with right:
                    st.markdown("## 🛡️ Technical view")
                    with st.container(border=True):
                        st.markdown(technical_md)

                    add_download(downloads, "⬇️ Download Technical (md)", technical_md.encode("utf-8"), f"{sample_name}_llm_technical.md", "text/markdown")

                # Combined download
                combined = (
                    "=== GENERAL (raw) ===\n\n" + general_out.strip() +
                    "\n\n=== TECHNICAL (md) ===\n\n" + technical_md.strip()
                )
                add_download(downloads, "⬇️ Download LLM (combined)", combined.encode("utf-8"), f"{sample_name}_llm_combined.txt", "text/plain")

            except Exception as e:
                st.warning(f"LLM failed: {e}")

    # Save downloads for downloads section
        st.session_state["downloads"] = downloads


# =========================
# Downloads
# =========================
if st.session_state.get("downloads"):
    st.divider()
    st.subheader("⬇️ Downloads")
    for i, d in enumerate(st.session_state["downloads"]):
        st.download_button(
            label=d["label"],
            data=d["data"],
            file_name=d["file_name"],
            mime=d["mime"],
            key=f"dl_{i}_{d['file_name']}",
        )