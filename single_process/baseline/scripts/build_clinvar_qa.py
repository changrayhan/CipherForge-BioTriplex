#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the ClinVar variant-pathogenicity QA dataset (plaintext baseline).

Input sources (either one):
  1. --from_parquet : pre-parsed ClinVar parquet (just-dna-seq/clinvar schema:
     chrom, start, ref, alt, rsid, variation_id, allele_id, gene, genes,
     clin_sig, clin_sig_raw, review_status, review_stars, condition,
     molecular_consequence, variant_type, origin). Preferred: NCBI raw VCF is
     unreachable in this environment, and this table is the parsed equivalent
     (ClinVar VCF 2026-06-27, Apache-2.0).
  2. --vcf          : raw ClinVar VCF (GRCh38); INFO: CLNSIG / CLNREVSTAT /
     MC / ORIGIN / GENEINFO / CLNHGVS.

Filter: biallelic SNV -> missense_variant -> germline -> P/LP vs B/LB ->
        review-status gate (--min_review_stars) -> gene present ->
        dedup by chr:pos:ref:alt.
Split : genes -> train/val/test 80/10/10; per-class caps; per-(gene,label) cap.
Output: BioTriplex-style QA JSONL + splits.json + stats.json.

NOTE: the `condition` / phenotype column is deliberately NOT used as input
(label leak). Variant field uses genomic notation (chr:pos ref>alt) because
the parquet table has no transcript HGVS column.
"""
import argparse
import gzip
import json
import random
import re
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

P_SET = {"pathogenic", "likely pathogenic"}
B_SET = {"benign", "likely benign"}
QUESTION = "Is this genetic variant pathogenic for a human disease?"


def parse_info(info):
    d = {}
    for item in info.split(";"):
        if not item:
            continue
        if "=" in item:
            k, v = item.split("=", 1)
            d[k] = urllib.parse.unquote(v).strip('"')
        else:
            d[item] = True
    return d


def clinsig_label(raw):
    toks = {t.strip() for t in re.split(r"[|&/]", str(raw).replace("_", " ").lower()) if t.strip()}
    if not toks:
        return None
    if toks <= P_SET:
        return 1
    if toks <= B_SET:
        return 0
    return None


def revstat_rank(raw):
    s = str(raw).lower()
    if "no_assertion" in s or "no_classification" in s:
        return 0
    if "practice_guideline" in s:
        return 5
    if "reviewed_by_expert_panel" in s:
        return 4
    if "multiple_submitters" in s:
        return 3
    if "criteria_provided" in s:
        return 2
    if "single_submitter" in s:
        return 1
    return 0


def clinvar_stars(raw):
    """Map CLNREVSTAT to ClinVar star tiers:
    3 = practice guideline / reviewed by expert panel,
    2 = criteria provided, multiple submitters,
    1 = criteria provided, single submitter,
    0 = no assertion / no classification / unknown."""
    s = str(raw).lower()
    if "practice_guideline" in s or "reviewed_by_expert_panel" in s:
        return 3
    if "multiple_submitters" in s:
        return 2
    if "criteria_provided" in s:
        return 1
    return 0


def is_snv(ref, alt):
    return len(ref) == 1 and len(alt) == 1 and ref.upper() in "ACGT" and alt.upper() in "ACGT"


def is_missense(mc):
    if not mc:
        return False
    text = str(mc).replace("[", "").replace("]", "")
    for part in text.split(","):
        for tok in part.split("|"):
            if tok.strip().startswith("missense_variant"):
                return True
    return False


def strip_brackets(s):
    return str(s).strip().strip("[]").strip('"')


def is_germline(origin):
    """ClinVar ORIGIN is a bitmask (1=germline, 2=somatic, 4=de_novo, ...).
    The parquet stores it as int / digit string / bracketed string / object
    array; values may combine several origins with '|'."""
    if origin is None:
        return False
    for tok in re.split(r"[|&]", str(origin)):
        tok = tok.strip().strip("[]").strip('"')
        if tok == "germline":
            return True
        if tok.lstrip("-").isdigit() and (int(tok) & 1):
            return True
    return False


def cell(v):
    """Unwrap numpy object-array cells (parquet quirk of just-dna-seq/clinvar)
    into a plain string; join multi-element arrays with '|'."""
    if v is None:
        return ""
    if hasattr(v, "ndim"):
        import numpy as np
        a = np.asarray(v)
        if a.ndim == 0:
            return str(a.item())
        return "|".join(str(x) for x in a.tolist())
    return str(v)


def first_gene(geneinfo):
    if not geneinfo:
        return None
    g = str(geneinfo).split(":")[0].strip()
    return g or None


def first_hgvs(hgvs):
    if not hgvs:
        return None
    h = str(hgvs).split("|")[0].strip()
    return h or None


def iter_vcf_records(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 8:
                continue
            yield cols


def add_row(best, conflict, chrom, pos, ref, alt, gene, hgvs, label, rank, vid):
    key = (chrom, pos, ref, alt)
    row = {
        "id": f"clinvar-{chrom}:{pos}:{ref}>{alt}",
        "chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
        "gene": gene, "hgvs": hgvs, "label": label,
        "revstat_rank": rank, "rs": str(vid) if str(vid).startswith("rs") else "",
    }
    if key in best:
        if best[key][1] != label:
            conflict.add(key)
        elif rank > best[key][0]:
            best[key] = (rank, label, row)
    else:
        best[key] = (rank, label, row)


def load_vcf_rows(vcf, counters, min_review_stars=1):
    best, conflict = {}, set()
    print(f"[1/3] parsing {vcf} ...")
    for cols in iter_vcf_records(vcf):
        chrom, pos, vid, ref, alts = cols[0], cols[1], cols[2], cols[3], cols[4]
        info = parse_info(cols[7])
        counters["records"] += 1
        alt_list = alts.split(",")
        if len(alt_list) > 1:
            counters["multi_alt_records"] += 1
        for alt in alt_list:
            if not is_snv(ref, alt):
                counters["skipped_not_snv"] += 1
                continue
            if not is_missense(info.get("MC", "")):
                counters["skipped_not_missense"] += 1
                continue
            origins = {o.strip() for o in str(info.get("ORIGIN", "")).split("|")}
            if "germline" not in origins:
                counters["skipped_not_germline"] += 1
                continue
            label = clinsig_label(info.get("CLNSIG", ""))
            if label is None:
                counters["skipped_clinsig"] += 1
                continue
            rank = revstat_rank(info.get("CLNREVSTAT", ""))
            if rank < min_review_stars:
                counters["skipped_revstat"] += 1
                continue
            gene = first_gene(info.get("GENEINFO", ""))
            if not gene:
                counters["skipped_no_gene"] += 1
                continue
            hgvs = first_hgvs(info.get("CLNHGVS", ""))
            if not hgvs:
                counters["skipped_no_hgvs"] += 1
                continue
            add_row(best, conflict, chrom, pos, ref, alt, gene, hgvs, label, rank, vid)
    return best, conflict, counters


def load_parquet_rows(path, counters, min_review_stars=1):
    # WSL/pandas-3.0 compat: pandas.read_parquet hangs in this environment;
    # pyarrow direct read is equivalent (same DataFrame produced downstream).
    import pyarrow.parquet as pq

    needed = [
        "chrom", "start", "ref", "alt", "rsid", "id", "rs",
        "clnsig", "clnrevstat", "mc", "origin", "geneinfo", "clnhgvs",
        "variant_type", "molecular_consequence", "gene", "clin_sig", "review_stars",
    ]
    schema = pq.read_schema(str(path))
    cols = [c for c in needed if c in schema.names]
    df = pq.read_table(str(path), columns=cols).to_pandas()
    counters["records"] = len(df)
    best, conflict = {}, set()
    vcf_style = "geneinfo" in set(df.columns)
    print(f"[1/3] parsing parquet {path} ({len(df)} records) ...")
    for rec in df.itertuples(index=False):
        chrom = str(rec.chrom)
        pos = str(int(rec.start if not vcf_style else rec.start))
        ref = str(rec.ref)
        alt = str(rec.alt)
        counters["alleles"] += 1
        if not is_snv(ref, alt):
            counters["skipped_not_snv"] += 1
            continue
        if vcf_style:
            mc = strip_brackets(cell(getattr(rec, "mc", "")))
            if not is_missense(mc):
                counters["skipped_not_missense"] += 1
                continue
            if not is_germline(cell(getattr(rec, "origin", ""))):
                counters["skipped_not_germline"] += 1
                continue
            label = clinsig_label(strip_brackets(cell(getattr(rec, "clnsig", ""))))
            if label is None:
                counters["skipped_clinsig"] += 1
                continue
            status = strip_brackets(cell(getattr(rec, "clnrevstat", "")))
            rank = revstat_rank(status)
            if min_review_stars >= 2:
                # High-quality gate: multiple submitters / expert panel / guideline.
                if clinvar_stars(status) < min_review_stars:
                    counters["skipped_revstat"] += 1
                    continue
            elif rank < 1:
                # Original standard gate (matches the 178,896-variant baseline).
                counters["skipped_revstat"] += 1
                continue
            gene = first_gene(strip_brackets(cell(getattr(rec, "geneinfo", ""))))
            if not gene:
                counters["skipped_no_gene"] += 1
                continue
            hgvs = first_hgvs(strip_brackets(cell(getattr(rec, "clnhgvs", ""))))
            if not hgvs:
                counters["skipped_no_hgvs"] += 1
                continue
            vid = cell(getattr(rec, "rs", "") or getattr(rec, "id", "") or "")
            add_row(best, conflict, chrom, pos, ref, alt, gene, hgvs, label, rank, vid)
        else:
            vt = cell(getattr(rec, "variant_type", "") or "")
            if "single_nucleotide_variant" not in vt:
                counters["skipped_variant_type"] += 1
                continue
            mc = cell(getattr(rec, "molecular_consequence", "") or "")
            if not is_missense(mc):
                counters["skipped_not_missense"] += 1
                continue
            if not is_germline(getattr(rec, "origin", None)):
                counters["skipped_not_germline"] += 1
                continue
            label = clinsig_label(cell(getattr(rec, "clin_sig", "") or ""))
            if label is None:
                counters["skipped_clinsig"] += 1
                continue
            stars = int(getattr(rec, "review_stars", 0) or 0)
            if stars < min_review_stars:
                counters["skipped_revstat"] += 1
                continue
            gene = cell(getattr(rec, "gene", "") or "").strip()
            if not gene:
                counters["skipped_no_gene"] += 1
                continue
            hgvs = f"{chrom}:{pos} {ref}>{alt}"
            add_row(best, conflict, chrom, pos, ref, alt, gene, hgvs, label, stars, cell(getattr(rec, "rsid", "") or ""))
    return best, conflict, counters


def sample_rows(rows, rng, per_class_cap, max_per_gene_label):
    gene_label = Counter()
    kept = []
    for r in rows:
        key = (r["gene"], r["label"])
        if gene_label[key] >= max_per_gene_label:
            continue
        gene_label[key] += 1
        kept.append(r)
    buckets = {0: [], 1: []}
    for r in kept:
        buckets[r["label"]].append(r)
    out = []
    for label in (0, 1):
        pool = buckets[label]
        rng.shuffle(pool)
        out.extend(pool[:per_class_cap])
    return out


def build_dataset(best, conflict, counters, out_dir, args, source):
    rows = [r for k, (_rk, _lb, r) in best.items() if k not in conflict]
    counters["dedup_conflict_dropped"] = len(conflict)
    counters["final_variants"] = len(rows)
    counters["label_counts"] = dict(Counter(r["label"] for r in rows))
    print(f"  final variants: {len(rows)}  labels={counters['label_counts']}  conflicts={len(conflict)}")

    rng = random.Random(args.seed)
    genes = sorted({r["gene"] for r in rows})
    rng.shuffle(genes)
    n = len(genes)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    gene2split = {}
    for g in genes[:n_train]:
        gene2split[g] = "train"
    for g in genes[n_train:n_train + n_val]:
        gene2split[g] = "val"
    for g in genes[n_train + n_val:]:
        gene2split[g] = "test"

    buckets = {"train": [], "val": [], "test": []}
    for r in rows:
        buckets[gene2split[r["gene"]]].append(r)
    caps = {"train": args.max_train_per_class, "val": args.max_eval_per_class, "test": args.max_eval_per_class}
    sampled = {s: sample_rows(rs, rng, caps[s], args.max_per_gene_label) for s, rs in buckets.items()}

    def qa_line(r):
        return {
            "id": r["id"],
            "question": QUESTION,
            "input": f"Gene: {r['gene']} | Variant: {r['hgvs']} | Consequence: missense variant | Origin: germline",
            "output": "Yes" if r["label"] == 1 else "No",
            "meta": {
                "chrom": r["chrom"], "pos": r["pos"], "ref": r["ref"], "alt": r["alt"],
                "gene": r["gene"], "label": r["label"],
                "review_status_rank": r["revstat_rank"], "rs": r["rs"],
            },
        }

    print("[2/3] writing splits ...")
    for split in ("train", "val", "test"):
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in sampled[split]:
                fh.write(json.dumps(qa_line(r), ensure_ascii=False) + "\n")

    split_counts = {
        s: {
            "rows": len(sampled[s]),
            "labels": dict(Counter(r["label"] for r in sampled[s])),
            "genes": len({r["gene"] for r in sampled[s]}),
        }
        for s in ("train", "val", "test")
    }
    with open(out_dir / "splits.json", "w", encoding="utf-8") as fh:
        json.dump({"seed": args.seed, "gene2split": gene2split, "counts": split_counts}, fh, ensure_ascii=False, indent=2)
    stats = {
        "source": source,
        "counters": dict(counters),
        "splits": split_counts,
        "top_genes": Counter(r["gene"] for r in rows).most_common(20),
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    print("[3/3] done -> " + str(out_dir))
    for s in ("train", "val", "test"):
        print(f"  {s}: {split_counts[s]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vcf", default="", help="raw ClinVar VCF (GRCh38); auto-detect data/raw if empty")
    ap.add_argument("--from_parquet", default="", help="pre-parsed ClinVar parquet (just-dna-seq/clinvar schema)")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--max_train_per_class", type=int, default=5000)
    ap.add_argument("--max_eval_per_class", type=int, default=5000)
    ap.add_argument("--max_per_gene_label", type=int, default=2000)
    ap.add_argument("--min_review_stars", type=int, default=1,
                    help="parquet review_stars floor (1=single submitter, 2=multiple submitters/expert panel, 3=expert panel/practice guideline); use 2 for high-quality data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    counters = Counter()
    if args.from_parquet:
        src = Path(args.from_parquet)
        best, conflict, counters = load_parquet_rows(src, counters, args.min_review_stars)
    else:
        if args.vcf:
            src = Path(args.vcf)
        else:
            cands = sorted((base / "data" / "raw").glob("clinvar_*.vcf.gz"))
            if not cands:
                sys.exit("No clinvar_*.vcf.gz under data/raw and --from_parquet not given")
            src = cands[-1]
        best, conflict, counters = load_vcf_rows(src, counters, args.min_review_stars)

    out_dir = Path(args.out_dir) if args.out_dir else base / "data" / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dataset(best, conflict, counters, out_dir, args, source=str(src))


if __name__ == "__main__":
    main()
