"""Deterministic synthetic inputs for the quickstart.

Nothing here is real: skill names, occupations, postings, descriptions and
relevance judgments are generated from templates with a fixed seed. The data
exercise every stage of the pipeline (cleaning, co-occurrence thresholds,
isolated and unobserved skills, grouped splits, withheld judgments) but say
nothing about recommendation quality on real labor markets.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .encoders import HASHED_NGRAM_MODEL_ID, HASHED_NGRAM_REVISION
from .pipeline_io import write_json

DEFAULT_SEED = 20261011
TAXONOMY = [
    ("Welding and joining", [
        ("Arc welding", ["Steel arc welding", "Aluminum arc welding"]),
        ("Beam welding", ["Laser welding", "Electron beam welding"]),
        ("Solid-state joining", ["Friction stir welding", "Brazing and soldering"])]),
    ("Casting and forming", [
        ("Casting", ["Sand casting", "Die casting"]),
        ("Forging", ["Hot forging", "Cold forging"]),
        ("Sheet forming", ["Stamping", "Deep drawing"])]),
    ("Surface and heat treatment", [
        ("Plating", ["Electroplating", "Electroless plating"]),
        ("Coating", ["Thermal spray coating", "Powder coating"]),
        ("Heat treatment", ["Quenching and tempering", "Case hardening"])]),
    ("Digital manufacturing", [
        ("Process control", ["PLC programming", "Sensor data acquisition"]),
        ("Quality inspection", ["Vision inspection", "Nondestructive testing"]),
        ("Manufacturing analytics", ["Process data analysis", "Predictive maintenance"])]),
]
ASPECTS = ["process setup", "parameter tuning", "fixture design", "defect inspection", "equipment maintenance",
           "safety procedures", "material selection", "automation programming", "quality documentation", "cost estimation"]
SKILLS_PER_DETAIL = len(ASPECTS)
TITLES = ["Method and apparatus for {detail}", "System for {detail} with inline monitoring",
          "{Detail} process using adaptive control", "Apparatus for improving {detail} efficiency"]
FILLERS = ["Embodiments reduce cycle time and defect rates.",
           "The configuration improves repeatability under varying material conditions.",
           "Measurements are recorded for traceability and process control.",
           "The arrangement lowers energy use while maintaining product quality."]
SIZES = {"postings": 24000, "patents": 60, "occupations": 20, "years": [2021, 2022, 2023, 2024],
         "patent_years": [2019, 2020, 2021, 2022, 2023]}
GOLD_FILES = {"patents": "patents.parquet", "labels": "labels.parquet",
              "candidate_pool": "candidate_pool.parquet", "task_status": "task_status.parquet"}


def lower_detail(name):
    return name.lower().replace("plc", "PLC")


def skill_table():
    rows, details = [], []
    for m, (main, subs) in enumerate(TAXONOMY, 1):
        for s, (sub, detail_names) in enumerate(subs, 1):
            for d, detail in enumerate(detail_names, 1):
                details.append({"main_group": main, "sub_group": sub, "detail_group": detail})
                for k, aspect in enumerate(ASPECTS, 1):
                    rows.append({"skill_code": f"TS-{m}{s}{d}-{k:02d}", "skill_name": f"{lower_detail(detail)} {aspect}",
                                 "main_group": main, "sub_group": sub, "detail_group": detail})
    return pd.DataFrame(rows), pd.DataFrame(details)


def rare_sets(detail_count):
    """Skill indices that never appear in postings, and ones that appear below any threshold."""
    unobserved = {di * SKILLS_PER_DETAIL + 9 for di in range(14, detail_count)}
    low = {di * SKILLS_PER_DETAIL + 8 for di in range(4, detail_count)}
    return unobserved, low


def occupation_table(details):
    rows = []
    for i in range(SIZES["occupations"]):
        primary = details.iloc[i]
        rows.append({"occupation_code": f"{1001 + i}", "occupation_name": f"{lower_detail(primary.detail_group)} technician",
                     "group": primary.main_group, "subgroup": primary.sub_group,
                     "home_details": [i, (i + 6) % len(details), (i + 13) % len(details)]})
    return pd.DataFrame(rows)


def posting_tables(rng, skills, occupations, unobserved, low):
    n, years = SIZES["postings"], SIZES["years"]
    common = np.array([i for i in range(len(skills)) if i not in unobserved and i not in low])
    pools = []
    for _, occupation in occupations.iterrows():
        members, weights = [], []
        for share, detail in zip([.6, .25, .15], occupation.home_details):
            group = [i for i in range(detail * SKILLS_PER_DETAIL, (detail + 1) * SKILLS_PER_DETAIL) if i in set(common)]
            members.extend(group)
            weights.extend([share / len(group)] * len(group))
        pools.append((np.array(members), np.array(weights) / sum(weights)))
    occupation_weights = 1. / (np.arange(len(occupations)) + 1) ** .7
    occupation_ids = rng.choice(len(occupations), n, p=occupation_weights / occupation_weights.sum())
    counts = rng.integers(3, 9, n)
    extra = rng.random(n) < .12
    codes = skills.skill_code.to_numpy()
    rows = []
    for i in range(n):
        members, weights = pools[occupation_ids[i]]
        chosen = list(rng.choice(members, min(counts[i], len(members)), replace=False, p=weights))
        if extra[i]:
            chosen.append(int(rng.choice(common)))
        rows.append(chosen)
    for skill in sorted(low):
        for target in rng.choice(n, 2, replace=False):
            rows[int(target)].append(skill)
    source = np.where(rng.random(n) < .6, "job_site_a", "job_site_b")
    frame = pd.DataFrame({"source": source, "posting_id": [f"P{i:06d}" for i in range(n)],
                          "year": [years[i % len(years)] for i in range(n)], "month": rng.integers(1, 13, n),
                          "occupation": occupations.occupation_code.to_numpy()[occupation_ids],
                          "skills": [",".join(dict.fromkeys(codes[r])) for r in rows]})
    # Deliberate imperfections that the cleaning stage must handle.
    repeat = rng.random(n) < .02
    frame.loc[repeat, "skills"] = frame.loc[repeat, "skills"].map(lambda s: s + "," + s.split(",")[0])
    duplicates = frame.sample(n=int(n * .01), random_state=int(rng.integers(0, 2 ** 31)))
    bad_month = frame.sample(n=int(n * .0025), random_state=int(rng.integers(0, 2 ** 31))).copy()
    bad_month["posting_id"] = [f"X{i:06d}" for i in range(len(bad_month))]
    bad_month["month"] = 13
    future = frame.sample(n=int(n * .0025), random_state=int(rng.integers(0, 2 ** 31))).copy()
    future["posting_id"] = [f"F{i:06d}" for i in range(len(future))]
    future["year"] = years[-1] + 1
    missing = frame.sample(n=int(n * .003), random_state=int(rng.integers(0, 2 ** 31))).copy()
    missing["posting_id"] = None
    noisy = pd.concat([frame, duplicates, bad_month, future, missing], ignore_index=True)
    noisy["file_year"] = noisy.year.clip(upper=years[-1])
    files = {}
    for year in years:
        part = noisy[noisy.file_year == year].drop(columns="file_year").reset_index(drop=True)
        files[f"postings_{year}.parquet"] = part.astype({"year": "int64", "month": "int64"})
    return files


def patent_tables(rng, skills, details, unobserved, low):
    judgeable = [[i for i in range(di * SKILLS_PER_DETAIL, (di + 1) * SKILLS_PER_DETAIL)
                  if i not in unobserved and i not in low] for di in range(len(details))]
    patents, judgments = [], []
    years = SIZES["patent_years"]
    for i in range(SIZES["patents"]):
        year = years[i % len(years)]
        primary = i % len(details)
        sibling = primary ^ 1
        cross = (primary + 7) % len(details)
        far = (primary + 12) % len(details)
        core = sorted(rng.choice(judgeable[primary], int(rng.integers(2, 5)), replace=False))
        indirect = sorted(rng.choice(judgeable[sibling], int(rng.integers(2, 4)), replace=False))
        indirect.append(int(rng.choice(judgeable[cross])))
        negatives = sorted(rng.choice(judgeable[far], int(rng.integers(1, 4)), replace=False))
        withheld = []
        if i % 10 == 0:
            remaining = [s for s in judgeable[sibling] if s not in indirect]
            withheld = [int(rng.choice(remaining))]
        detail = details.iloc[primary].detail_group
        names = [skills.skill_name.iloc[s] for s in core]
        core_text = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
        title = TITLES[i % len(TITLES)].format(detail=lower_detail(detail), Detail=detail)
        abstract = (f"The invention relates to {lower_detail(detail)} in manufacturing. The disclosed method involves "
                    f"{core_text}. The process is combined with {lower_detail(details.iloc[sibling].detail_group)} and "
                    f"draws on {lower_detail(details.iloc[cross].detail_group)}. {FILLERS[i % len(FILLERS)]}")
        sample_id = f"TOY-{i + 1:04d}"
        family = "FAM-0001" if i in (6, 7) else "FAM-0002" if i in (20, 21) else ""
        patents.append({"sample_id": sample_id, "application_id": f"APP-{year}-{i + 1:04d}", "application_year": year,
                        "technology_description": f"{title}. {abstract}", "family_id": family})
        for grade, members in [(2, core), (1, indirect), (0, negatives), (None, withheld)]:
            for s in members:
                judgments.append({"sample_id": sample_id, "perspective": "skills", "skill_code": skills.skill_code.iloc[s],
                                  "skill_name": skills.skill_name.iloc[s], "relevance_score": grade,
                                  "eligible_numeric_label": grade is not None, "decision_method": "toy_generator_v1",
                                  "label_status": "judged" if grade is not None else "withheld"})
    pool = pd.DataFrame(judgments)
    pool["relevance_score"] = pool.relevance_score.astype("Int64")
    labels = pool[pool.eligible_numeric_label].reset_index(drop=True)
    tasks = pool.groupby(["sample_id", "perspective"]).agg(judged_candidates=("eligible_numeric_label", "sum"),
                                                          withheld_candidates=("eligible_numeric_label", lambda s: int((~s).sum()))).reset_index()
    tasks.insert(2, "task_status", "judged")
    return pd.DataFrame(patents), labels, pool, tasks


def app_config(seed):
    return {
        "version": "toy_v1", "market_id": "TOY", "seed": seed,
        "encoder": {"backend": "hashed_ngram", "model_id": HASHED_NGRAM_MODEL_ID, "revision": HASHED_NGRAM_REVISION,
                    "dimension": 1024, "max_seq_length": 4096, "batch_size": 64, "device": "cpu", "cpu_threads": 4},
        "skills": {"path": "inputs/skills.csv", "id_column": "skill_code", "name_column": "skill_name",
                   "taxonomy_columns": ["main_group", "sub_group", "detail_group"],
                   "text_template": "Skill: {skill_name}\nCategory: {taxonomy_path}", "taxonomy_separator": " > "},
        "postings": {"files": [f"inputs/postings/postings_{year}.parquet" for year in SIZES["years"]],
                     "columns": {"source": "source", "posting_id": "posting_id", "year": "year", "month": "month",
                                 "skills": "skills", "occupation": "occupation"},
                     "skill_separator": ",", "occupation_code_width": 4,
                     "start_year": SIZES["years"][0], "end_year": SIZES["years"][-1],
                     "threads": 2, "memory_limit": "1GB", "batch_size": 50000},
        "graph": {"thresholds": {"min_skill_postings": 1, "min_cooccurrence": 5, "weight_metric": "cosine",
                                 "min_weight": 0.05, "min_active_years": 1},
                  "grid": {"min_skill_postings": [1, 5], "min_cooccurrence": [1, 5, 10, 20], "min_weight": [0.0, 0.05, 0.1, 0.2]}},
        "occupations": {"path": "inputs/occupations.csv", "code_column": "occupation_code", "name_column": "occupation_name",
                        "path_columns": ["group", "subgroup"], "code_width": 4,
                        "embedding_template": "Occupation: {name}\nCategory: {path}",
                        "min_occupation_skill_postings": 5, "threshold_grid": [1, 2, 5, 10, 20]},
        "patents": {"dir": "inputs/patents", "files": dict(GOLD_FILES),
                    "columns": {"application": "application_id", "year": "application_year",
                                "text": "technology_description", "family": "family_id"},
                    "split_fractions": {"train": 0.6, "validation": 0.2, "test": 0.2}, "split_seed": 42},
        "model": {"variant": "lara", "device": "cpu", "cpu_threads": 4, "hidden_dim": 64, "decoder_dim": 32,
                  "attention_heads": 2, "query_neighbors": 10, "epochs": 30, "score_chunk_size": 8192},
        "evaluation": {"ks": [5, 10, 20, 50, 100]},
    }


def content_sha256(frame):
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def generate(seed=DEFAULT_SEED):
    """Return every toy table in memory, keyed by relative output path."""
    rng = np.random.default_rng(seed)
    skills, details = skill_table()
    unobserved, low = rare_sets(len(details))
    occupations = occupation_table(details)
    tables = {"inputs/skills.csv": skills,
              "inputs/occupations.csv": occupations.drop(columns="home_details")}
    for name, frame in posting_tables(rng, skills, occupations, unobserved, low).items():
        tables[f"inputs/postings/{name}"] = frame
    patents, labels, pool, tasks = patent_tables(rng, skills, details, unobserved, low)
    for key, frame in [("patents", patents), ("labels", labels), ("candidate_pool", pool), ("task_status", tasks)]:
        tables[f"inputs/patents/{GOLD_FILES[key]}"] = frame
    return tables


def write_toy_inputs(output, seed=DEFAULT_SEED):
    """Write the toy tables, the application config and a content manifest into ``output``."""
    output = Path(output)
    if (output / "toy_manifest.json").exists():
        raise FileExistsError(f"Toy inputs already exist in {output}")
    tables = generate(seed)
    manifest = {"seed": seed, "generator": "lara_skills.toy", "synthetic": True, "tables": {}}
    for relative, frame in tables.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
        else:
            path.write_bytes(frame.to_parquet(index=False))
        manifest["tables"][relative] = {"rows": int(len(frame)), "columns": list(frame.columns),
                                        "content_sha256": content_sha256(frame)}
    config = app_config(seed)
    write_json(output / "lara_app.json", config)
    manifest["app_config"] = "lara_app.json"
    write_json(output / "toy_manifest.json", manifest)
    return manifest


SAMPLE_DESCRIPTION = ("Apparatus for laser welding of aluminum battery housings. The invention relates to laser welding "
                      "in manufacturing and involves laser welding parameter tuning, laser welding fixture design and "
                      "laser weld defect inspection combined with electron beam welding process setup.")
