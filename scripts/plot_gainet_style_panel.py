#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D


def _extract_highlight_from_mcs(m1: Chem.Mol, m2: Chem.Mol) -> tuple[list[int], list[int], list[int], list[int]]:
    mcs = rdFMCS.FindMCS([m1, m2], timeout=4, completeRingsOnly=True)
    if not mcs.smartsString:
        return [], [], [], []
    patt = Chem.MolFromSmarts(mcs.smartsString)
    if patt is None:
        return [], [], [], []

    match1 = list(m1.GetSubstructMatch(patt))
    match2 = list(m2.GetSubstructMatch(patt))
    if not match1 or not match2:
        return [], [], [], []

    def match_bonds(mol: Chem.Mol, match: list[int]) -> list[int]:
        out = []
        for b in patt.GetBonds():
            a1 = match[b.GetBeginAtomIdx()]
            a2 = match[b.GetEndAtomIdx()]
            mb = mol.GetBondBetweenAtoms(a1, a2)
            if mb is not None:
                out.append(mb.GetIdx())
        return out

    return match1, match_bonds(m1, match1), match2, match_bonds(m2, match2)


def _extract_highlight_from_smarts(mol: Chem.Mol, smarts: str) -> tuple[list[int], list[int]]:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return [], []
    match = list(mol.GetSubstructMatch(patt))
    if not match:
        return [], []
    bonds = []
    for b in patt.GetBonds():
        a1 = match[b.GetBeginAtomIdx()]
        a2 = match[b.GetEndAtomIdx()]
        mb = mol.GetBondBetweenAtoms(a1, a2)
        if mb is not None:
            bonds.append(mb.GetIdx())
    return match, bonds


def _mol_image(
    mol: Chem.Mol,
    legend: str = "",
    highlight_atoms: list[int] | None = None,
    highlight_bonds: list[int] | None = None,
    width: int = 520,
    height: int = 340,
    line_width: float = 1.8,
) -> np.ndarray:
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.addAtomIndices = False
    options.bondLineWidth = line_width
    options.padding = 0.03
    options.legendFontSize = 20

    highlight_atoms = highlight_atoms or []
    highlight_bonds = highlight_bonds or []
    atom_colors = {idx: (0.96, 0.20, 0.20, 0.75) for idx in highlight_atoms}
    bond_colors = {idx: (0.90, 0.15, 0.15, 0.9) for idx in highlight_bonds}

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        legend=legend,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    return np.asarray(img)


def _row_label(i: int) -> str:
    return f"({chr(ord('a') + i)})"


def _annotation_text(row: pd.Series) -> str:
    p1 = float(row.get("pred1", np.nan))
    p2 = float(row.get("pred2", np.nan))
    avg = float(row.get("avg_pred", np.nan))
    act = float(row.get("actual", np.nan))
    return (
        f"Interaction Prediction 1: {p1:.4f}\n"
        f"Interaction Prediction 2: {p2:.4f}\n"
        f"Average Interaction Prediction: {avg:.4f}\n"
        f"Actual: {act:.1f}"
    )


def _safe_seq_to_smiles(seq: str, max_len_draw: int = 25) -> str | None:
    seq = (seq or "").strip().upper()
    if not seq:
        return None
    # Protein dizisi çok uzunsa çizim için fragman kullan.
    if len(seq) > max_len_draw:
        seq = seq[:max_len_draw]
    mol = Chem.MolFromFASTA(seq)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _rows_from_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def _select_rows(df: pd.DataFrame, mode: str, threshold: float, max_rows: int) -> pd.DataFrame:
    d = df.copy()
    d["pred_label"] = (d["avg_pred"] >= threshold).astype(int)

    if mode == "tp":
        d = d[(d["pred_label"] == 1) & (d["actual"] == 1)]
    elif mode == "tn":
        d = d[(d["pred_label"] == 0) & (d["actual"] == 0)]
    elif mode == "fp":
        d = d[(d["pred_label"] == 1) & (d["actual"] == 0)]
    elif mode == "fn":
        d = d[(d["pred_label"] == 0) & (d["actual"] == 1)]
    elif mode == "pos":
        d = d[d["avg_pred"] >= threshold]
    elif mode == "neg":
        d = d[d["avg_pred"] < threshold]

    d["confidence"] = np.abs(d["avg_pred"] - 0.5)
    d = d.sort_values("confidence", ascending=False)
    return d.head(max_rows)


def build_panel_df_from_real(
    pairs_jsonl: str | None,
    pred_csv: str,
    max_seq_len_draw: int,
    mode: str,
    threshold: float,
    max_rows: int,
) -> pd.DataFrame:
    pred = pd.read_csv(pred_csv)
    if "pair_id" not in pred.columns:
        raise ValueError("pred CSV must contain pair_id")

    if "pred1" not in pred.columns:
        if "pred_prob" in pred.columns:
            pred["pred1"] = pred["pred_prob"].astype(float)
        elif "prediction" in pred.columns:
            pred["pred1"] = pred["prediction"].astype(float)
        else:
            raise ValueError("pred CSV must contain pred1 or pred_prob or prediction")

    if "pred2" not in pred.columns:
        pred["pred2"] = pred["pred1"]

    if "avg_pred" not in pred.columns:
        pred["avg_pred"] = (pred["pred1"].astype(float) + pred["pred2"].astype(float)) / 2.0

    if "actual" not in pred.columns:
        if "label" in pred.columns:
            pred["actual"] = pred["label"]
        else:
            pred["actual"] = np.nan

    pairs = None
    if pairs_jsonl:
        pairs = pd.DataFrame(_rows_from_jsonl(pairs_jsonl))
        if "pair_id" not in pairs.columns:
            raise ValueError("pairs jsonl must contain pair_id")
        df = pairs.merge(pred, on="pair_id", how="inner", suffixes=("", "_pred"))
    else:
        df = pred.copy()

    # names
    if "name1" not in df.columns:
        if "peptide_name" in df.columns:
            df["name1"] = df["peptide_name"]
        else:
            df["name1"] = df["pair_id"].astype(str) + "_A"

    if "name2" not in df.columns:
        if "protein_name" in df.columns:
            df["name2"] = df["protein_name"]
        else:
            df["name2"] = df["pair_id"].astype(str) + "_B"

    # smiles derivation (real data path)
    if "smiles1" not in df.columns:
        if "peptide_smiles" in df.columns:
            df["smiles1"] = df["peptide_smiles"]
        elif "peptide_seq" in df.columns:
            df["smiles1"] = df["peptide_seq"].fillna("").map(lambda s: _safe_seq_to_smiles(str(s), max_len_draw=max_seq_len_draw))
        else:
            df["smiles1"] = None

    if "smiles2" not in df.columns:
        if "protein_smiles" in df.columns:
            df["smiles2"] = df["protein_smiles"]
        elif "protein_fragment_seq" in df.columns:
            df["smiles2"] = df["protein_fragment_seq"].fillna("").map(lambda s: _safe_seq_to_smiles(str(s), max_len_draw=max_seq_len_draw))
        elif "protein_seq" in df.columns:
            df["smiles2"] = df["protein_seq"].fillna("").map(lambda s: _safe_seq_to_smiles(str(s), max_len_draw=max_seq_len_draw))
        else:
            df["smiles2"] = None

    # keep rows with drawable molecules
    df = df[df["smiles1"].notna() & df["smiles2"].notna()].copy()

    required = ["name1", "smiles1", "name2", "smiles2", "pred1", "pred2", "avg_pred", "actual"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column after merge/derivation: {c}")

    return _select_rows(df, mode=mode, threshold=threshold, max_rows=max_rows)


def make_panel(df: pd.DataFrame, output_png: str | Path) -> None:
    n = len(df)
    if n == 0:
        raise ValueError("No valid rows selected for plotting")

    fig_h = 3.4 * n
    fig = plt.figure(figsize=(18, fig_h), facecolor="#e9e9e9")
    gs = fig.add_gridspec(nrows=n, ncols=1, left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.08)

    for i, (_, row) in enumerate(df.iterrows()):
        m1 = Chem.MolFromSmiles(str(row["smiles1"]))
        m2 = Chem.MolFromSmiles(str(row["smiles2"]))
        if m1 is None or m2 is None:
            continue

        Chem.rdDepictor.Compute2DCoords(m1)
        Chem.rdDepictor.Compute2DCoords(m2)

        h1_atoms, h1_bonds, _, _ = _extract_highlight_from_mcs(m1, m2)
        custom_smarts = row.get("highlight_smarts", np.nan)
        if isinstance(custom_smarts, str) and custom_smarts.strip():
            c_atoms, c_bonds = _extract_highlight_from_smarts(m1, custom_smarts.strip())
            if c_atoms:
                h1_atoms, h1_bonds = c_atoms, c_bonds

        img_left = _mol_image(m1)
        img_mid = _mol_image(m2)
        img_right = _mol_image(m1, highlight_atoms=h1_atoms, highlight_bonds=h1_bonds, line_width=2.2)

        ax = fig.add_subplot(gs[i, 0])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.imshow(img_left, extent=(0.03, 0.32, 0.14, 0.86), aspect="auto")
        ax.imshow(img_mid, extent=(0.36, 0.65, 0.14, 0.86), aspect="auto")
        ax.imshow(img_right, extent=(0.72, 0.98, 0.14, 0.86), aspect="auto")

        ax.annotate("", xy=(0.71, 0.50), xytext=(0.66, 0.50), arrowprops=dict(arrowstyle="->", lw=2.2, color="black"))
        ax.text(0.005, 0.94, _row_label(i), fontsize=18, fontweight="bold")
        ax.text(0.50, 0.91, _annotation_text(row), fontsize=11, ha="center", va="top")
        ax.text(0.175, 0.06, str(row["name1"]), fontsize=10, ha="center")
        ax.text(0.505, 0.06, str(row["name2"]), fontsize=10, ha="center")

    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create GAINET-style interaction panel")
    p.add_argument("--input-csv", help="Direct panel csv with smiles+pred columns")
    p.add_argument("--pairs-jsonl", help="Real pairs metadata jsonl (optional)")
    p.add_argument("--pred-csv", help="Prediction CSV (pair_id + pred columns)")
    p.add_argument("--output-png", required=True)
    p.add_argument("--max-rows", type=int, default=5)
    p.add_argument("--mode", choices=["all", "tp", "tn", "fp", "fn", "pos", "neg"], default="all")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-seq-len-draw", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        if "avg_pred" not in df.columns:
            df["avg_pred"] = (df["pred1"].astype(float) + df["pred2"].astype(float)) / 2.0
        if "actual" not in df.columns:
            df["actual"] = np.nan
        df = _select_rows(df, mode=args.mode, threshold=args.threshold, max_rows=args.max_rows)
    else:
        if not args.pred_csv:
            raise ValueError("Either --input-csv OR --pred-csv must be provided")
        df = build_panel_df_from_real(
            pairs_jsonl=args.pairs_jsonl,
            pred_csv=args.pred_csv,
            max_seq_len_draw=args.max_seq_len_draw,
            mode=args.mode,
            threshold=args.threshold,
            max_rows=args.max_rows,
        )

    make_panel(df, args.output_png)
    print(f"[saved] {args.output_png}")
    print(f"[rows] {len(df)}")


if __name__ == "__main__":
    main()
