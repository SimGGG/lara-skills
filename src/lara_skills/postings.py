"""Configurable posting normalization and binary skill co-occurrence counts."""

import io
import json
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse

from .pipeline_io import digest, output_run, write_json, write_parquet


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def sql_column(value):
    return '"' + value.replace('"', '""') + '"'


def records(con, query):
    result = con.execute(query)
    return [dict(zip([d[0] for d in result.description], row)) for row in result.fetchall()]


def clean_postings(root, cfg):
    root = Path(root)
    destination = root / cfg["clean_output"]
    with output_run(root, destination, "clean_postings", cfg) as (out, manifest):
        inputs = [root / p for p in cfg["input_files"]]
        manifest["inputs"] = [{"path": str(p.relative_to(root)), "sha256": digest(p),
                               "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in inputs]
        index_path = root / cfg["skill_index"]
        dictionary = pd.read_parquet(io.BytesIO(index_path.read_bytes()))
        if dictionary.skill_code.isna().any() or not dictionary.skill_code.is_unique:
            raise ValueError("Invalid skill dictionary identifiers")
        manifest["skill_index_sha256"] = digest(index_path)
        c = {k: sql_column(v) for k, v in cfg["columns"].items() if v is not None}
        # DuckDB's spill and database files stay in a managed local temporary directory.
        with tempfile.TemporaryDirectory(prefix="skill_postings_") as work:
            con = duckdb.connect(str(Path(work) / "working.duckdb"))
            try:
                con.execute(f"SET threads={int(cfg['threads'])}")
                con.execute(f"SET memory_limit={sql_string(cfg['memory_limit'])}")
                con.execute(f"SET temp_directory={sql_string(Path(work) / 'spill')}")
                con.execute("SET preserve_insertion_order=false")
                con.register("dictionary", dictionary[["skill_code"]])
                summaries = []
                for file_order, path in enumerate(inputs):
                    print(f"Normalizing {path.name}", flush=True)
                    yr = f"trim(cast({c['year']} AS VARCHAR))"
                    mo = f"trim(cast({c['month']} AS VARCHAR))"
                    occ = f"nullif(trim(cast({c['occupation']} AS VARCHAR)),'')" if "occupation" in c else "NULL::VARCHAR"
                    if cfg.get("occupation_code_width"):
                        occ = f"lpad({occ},{int(cfg['occupation_code_width'])},'0')"
                    con.execute(f"""CREATE OR REPLACE TABLE chunk AS SELECT
                        {file_order}::INTEGER AS input_file_index, file_row_number AS input_row,
                        nullif(trim(cast({c['source']} AS VARCHAR)),'') AS source,
                        nullif(trim(cast({c['posting_id']} AS VARCHAR)),'') AS source_posting_id,
                        CASE WHEN regexp_full_match({yr},'[0-9]{{4}}') THEN try_cast({yr} AS INTEGER) END AS year,
                        CASE WHEN regexp_full_match({mo},'[0-9]{{1,2}}') THEN try_cast({mo} AS INTEGER) END AS month,
                        {occ} AS occupation_code,
                        list_filter(list_transform(string_split(coalesce(cast({c['skills']} AS VARCHAR),''),
                        {sql_string(cfg['skill_separator'])}), x->trim(x)),x->x<>'') AS original_skills
                        FROM read_parquet({sql_string(path)},file_row_number=true)""")
                    con.execute("ALTER TABLE chunk ADD COLUMN skills VARCHAR[]")
                    con.execute("UPDATE chunk SET skills=list_sort(list_distinct(original_skills))")
                    valid_period = f"coalesce(year BETWEEN {int(cfg['start_year'])} AND {int(cfg['end_year'])} AND month BETWEEN 1 AND 12,false)"
                    valid_id = "source IS NOT NULL AND source_posting_id IS NOT NULL"
                    row = records(con, f"""SELECT count(*) AS raw_rows,
                        count(*) FILTER(WHERE NOT ({valid_period})) AS excluded_period,
                        count(*) FILTER(WHERE ({valid_period}) AND NOT ({valid_id})) AS excluded_missing_id,
                        count(*) FILTER(WHERE ({valid_period}) AND ({valid_id})) AS eligible_rows,
                        sum(len(original_skills)-len(skills)) AS duplicate_skill_tokens_removed
                        FROM chunk""")[0]
                    row["input_file"] = cfg["input_files"][file_order]
                    summaries.append(row)
                    selection = f"SELECT * EXCLUDE(original_skills) FROM chunk WHERE ({valid_period}) AND ({valid_id})"
                    if file_order == 0:
                        con.execute("CREATE TABLE eligible AS " + selection)
                    else:
                        con.execute("INSERT INTO eligible " + selection)
                    print(json.dumps(row), flush=True)
                con.execute("DROP TABLE chunk")
                con.execute("""CREATE TABLE duplicate_groups AS SELECT source,source_posting_id,count(*) AS copies,
                    count(DISTINCT struct_pack(yr:=year,mo:=month,occ:=occupation_code,sk:=skills)) AS versions
                    FROM eligible GROUP BY source,source_posting_id HAVING count(*)>1""")
                conflicts = con.execute("SELECT count(*) FROM duplicate_groups WHERE versions>1").fetchone()[0]
                if conflicts:
                    raise ValueError(f"{conflicts} duplicate posting keys have conflicting period/occupation/skills")
                market = sql_string(cfg["market_id"])
                con.execute(f"""CREATE TABLE cleaned AS SELECT {market} AS market_id,
                    cast(to_json([{market},source,source_posting_id]) AS VARCHAR) AS posting_id,
                    source, source_posting_id, year, month,
                    printf('%04d-%02d',year,month) AS period, 'month' AS date_precision,
                    occupation_code, skills, input_file_index,input_row
                    FROM eligible QUALIFY row_number() OVER (
                    PARTITION BY source,source_posting_id ORDER BY input_file_index,input_row)=1""")
                con.execute("DROP TABLE eligible")
                con.execute("""CREATE TABLE frequency AS SELECT code AS skill_code,count(*) AS posting_count
                    FROM cleaned,unnest(skills) AS u(code) GROUP BY code""")
                unknown = records(con, "SELECT f.* FROM frequency f ANTI JOIN dictionary d USING(skill_code)")
                if unknown:
                    raise ValueError(f"Unknown skill codes: {unknown[:20]}")
                summary = records(con, """SELECT count(*) AS postings,
                    count(*) FILTER(WHERE len(skills)>0) AS postings_with_skills,
                    count(*) FILTER(WHERE len(skills)>=2) AS postings_with_pairs,
                    sum(len(skills)) AS posting_skill_links,
                    sum(len(skills)*(len(skills)-1)//2) AS pair_events,
                    max(len(skills)) AS max_skills_per_posting FROM cleaned""")[0]
                if not summary["postings"] or not summary["posting_skill_links"]:
                    raise ValueError("No usable postings/skills remain after filtering")
                summary["raw_rows"] = sum(r["raw_rows"] for r in summaries)
                summary["eligible_rows"] = sum(r["eligible_rows"] for r in summaries)
                summary["duplicate_rows_removed"] = summary["eligible_rows"] - summary["postings"]
                summary["duplicate_groups"] = con.execute("SELECT count(*) FROM duplicate_groups").fetchone()[0]
                summary["duplicate_conflicts"] = conflicts
                summary["unknown_skills"] = len(unknown)
                summary["observed_skills"] = con.execute("SELECT count(*) FROM frequency").fetchone()[0]
                summary["per_input_file"] = summaries
                con.execute(f"COPY duplicate_groups TO {sql_string(out / 'duplicate_groups.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)")
                con.execute(f"COPY (SELECT * FROM frequency ORDER BY skill_code) TO {sql_string(out / 'skill_frequency.parquet')} (FORMAT PARQUET)")
                by_source = con.execute("""SELECT year,source,count(*) AS postings,
                    count(*) FILTER(WHERE len(skills)>0) AS postings_with_skills,
                    sum(len(skills)) AS posting_skill_links,
                    sum(len(skills)*(len(skills)-1)//2) AS pair_events
                    FROM cleaned GROUP BY year,source ORDER BY year,source""").df()
                by_source.to_csv(out / "year_source_summary.csv", index=False, encoding="utf-8-sig")
                con.execute(f"COPY (SELECT len(skills) AS skill_count,count(*) AS postings FROM cleaned GROUP BY len(skills) ORDER BY skill_count) TO {sql_string(out / 'skills_per_posting.csv')} (HEADER,DELIMITER ',')")
                for year in range(cfg["start_year"], cfg["end_year"] + 1):
                    print(f"Saving standardized postings for {year}", flush=True)
                    con.execute(f"COPY (SELECT * FROM cleaned WHERE year={year}) TO {sql_string(out / f'postings_{year}.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 125000)")
                manifest["summary"] = summary
                write_json(out / "quality_summary.json", summary)
                write_json(out / "config.json", cfg)
            finally:
                con.close()
        # Exact content hashes were taken before reading; detect source changes during processing.
        for path, provenance in zip(inputs, manifest["inputs"]):
            if path.stat().st_size != provenance["bytes"] or path.stat().st_mtime_ns != provenance["mtime_ns"]:
                raise RuntimeError(f"Input changed during processing: {path}")
        print(json.dumps(summary), flush=True)
    return destination


def batch_incidence(skills, vocabulary):
    """Build binary posting/skill incidence, rejecting unrecognized codes."""
    offsets = skills.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    # list_flatten respects sliced Arrow list offsets.
    flat = pc.list_flatten(skills)
    indices = pc.index_in(flat, value_set=vocabulary)
    if indices.null_count:
        raise ValueError("Unrecognized skill in normalized postings")
    idx = indices.to_numpy(zero_copy_only=False).astype(np.int32)
    matrix = sparse.csr_matrix((np.ones(len(idx), dtype=np.int64), idx, offsets - offsets[0]),
                               shape=(len(skills), len(vocabulary)))
    matrix.sum_duplicates()
    matrix.data.fill(1)
    matrix.sort_indices()
    return matrix


def aggregate_cooccurrence(root, cfg):
    root = Path(root)
    clean = root / cfg["clean_output"]
    clean_manifest = json.loads((clean / "manifest.json").read_text(encoding="utf-8"))
    if clean_manifest["status"] != "completed":
        raise ValueError("Cleaned dataset is not complete")
    with output_run(root, root / cfg["statistics_output"], "skill_cooccurrence", cfg) as (out, manifest):
        frequency_path = clean / "skill_frequency.parquet"
        frequency = pd.read_parquet(io.BytesIO(frequency_path.read_bytes())).sort_values("skill_code").reset_index(drop=True)
        vocabulary = pa.array(frequency.skill_code)
        n = len(frequency)
        annual = []
        years = list(range(cfg["start_year"], cfg["end_year"] + 1))
        # One dense count accumulator avoids repeatedly materializing hundreds of millions of pairs.
        total = np.zeros((n, n), dtype=np.int64)
        year_freq = {}
        year_edges = {}
        expected_all_pairs = 0
        for year in years:
            path = clean / f"postings_{year}.parquet"
            if digest(path) != clean_manifest["files"][path.name]["sha256"]:
                raise ValueError(f"Cleaned posting file changed: {path}")
            count = np.zeros((n, n), dtype=np.int64)
            postings, links, events = 0, 0, 0
            for batch_no, batch in enumerate(pq.ParquetFile(path).iter_batches(batch_size=cfg["batch_size"], columns=["skills"]), 1):
                matrix = batch_incidence(batch.column(0), vocabulary)
                lengths = np.diff(matrix.indptr).astype(np.int64)
                product = (matrix.T @ matrix).tocoo()
                count[product.row, product.col] += product.data
                postings += matrix.shape[0]
                links += int(lengths.sum())
                events += int((lengths * (lengths - 1) // 2).sum())
                if batch_no % 4 == 0:
                    print(f"Co-occurrence {year}: {postings:,} postings", flush=True)
            freq = count.diagonal().copy()
            if int(freq.sum()) != links or int((count.sum() - freq.sum()) // 2) != events:
                raise ValueError("Co-occurrence event accounting failed")
            year_freq[year] = freq
            row, col = np.nonzero(np.triu(count, k=1))
            edges = pd.DataFrame({"source_node": row.astype(np.int32), "target_node": col.astype(np.int32),
                                  "cooccurrence": count[row, col]})
            write_parquet(edges, out / f"pairs_{year}.parquet")
            year_edges[year] = edges
            annual.append({"year": year, "postings": postings, "posting_skill_links": links,
                           "pair_events": events, "unique_pairs": len(edges), "observed_skills": int((freq > 0).sum())})
            total += count
            expected_all_pairs += events
            print(json.dumps(annual[-1]), flush=True)
        if not np.array_equal(total.diagonal(), frequency.posting_count.to_numpy()):
            raise ValueError("Counts do not match independent SQL skill frequency")
        row, col = np.nonzero(np.triu(total, k=1))
        counts = total[row, col]
        if int(counts.sum()) != expected_all_pairs or expected_all_pairs != clean_manifest["summary"]["pair_events"]:
            raise ValueError("Pair events do not match independent cleaned-data totals")
        freq = frequency.posting_count.to_numpy()
        f1, f2 = freq[row], freq[col]
        pairs = pd.DataFrame({"source_node": row.astype(np.int32), "target_node": col.astype(np.int32),
                              "cooccurrence": counts,
                              "cosine": counts / np.sqrt(f1.astype(float) * f2),
                              "jaccard": counts / (f1 + f2 - counts)})
        if np.any(counts > np.minimum(f1, f2)) or np.any(row >= col):
            raise ValueError("Invalid pair support or self-loop")
        key = row.astype(np.int64) * n + col
        active = np.zeros(len(pairs), dtype=np.int16)
        for year, edges in year_edges.items():
            annual_key = edges.source_node.to_numpy().astype(np.int64) * n + edges.target_node.to_numpy()
            active[np.searchsorted(key, annual_key)] += 1
        pairs["active_years"] = active
        nodes = frequency.copy()
        nodes.insert(0, "node_id", np.arange(n, dtype=np.int32))
        for year in years:
            nodes[f"postings_{year}"] = year_freq[year]
        dictionary = pd.read_parquet(io.BytesIO((root / cfg["skill_index"]).read_bytes()))
        # Keep every dictionary column (identifier, name, taxonomy levels, embedding row) whatever its name.
        keep = [c for c in dictionary.columns if c not in {"embedding_text", "text_sha256", "token_count"}]
        nodes = nodes.merge(dictionary[keep], on="skill_code", how="left", validate="one_to_one")
        if nodes.embedding_row.isna().any():
            raise ValueError("Missing embedding mapping")
        write_parquet(nodes, out / "nodes.parquet")
        nodes.to_csv(out / "nodes.csv", index=False, encoding="utf-8-sig")
        write_parquet(pairs, out / "pairs.parquet")
        pd.DataFrame(annual).to_csv(out / "annual_summary.csv", index=False, encoding="utf-8-sig")
        distribution = pairs.cooccurrence.value_counts().sort_index().rename_axis("cooccurrence").reset_index(name="pairs")
        distribution.to_csv(out / "pair_support_distribution.csv", index=False)
        summary = {"nodes": n, "unique_pairs": len(pairs), "pair_events": expected_all_pairs,
                   "postings": clean_manifest["summary"]["postings"], "posting_skill_links": int(freq.sum()),
                   "self_loops": 0, "unknown_skills": 0, "count_dtype": "int64",
                   "frequency_and_pair_totals_verified": True, "years": years,
                   "pair_support_quantiles": {str(q): float(pairs.cooccurrence.quantile(q)) for q in [0, .25, .5, .75, .9, .95, .99, 1]}}
        write_json(out / "summary.json", summary)
        manifest.update({"clean_manifest_sha256": digest(clean / "manifest.json"),
                         "skill_index_sha256": digest(root / cfg["skill_index"]), "summary": summary})
        print(json.dumps(summary), flush=True)
    return root / cfg["statistics_output"]
