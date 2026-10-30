#!/usr/bin/env python3
"""
resolve_codes.py -- OMOP concept resolution for the TTE study (ty0_1).

Maps the study's vocabulary-neutral code content (codesets_tte.py + lookup_table CSVs) to OMOP
STANDARD concept sets using the source's own vocabulary tables (concept /
concept_relationship / concept_ancestor / drug_strength) via duckdb. OMOP event records
store standard concepts (condition_concept_id = SNOMED, drug_concept_id = RxNorm), so
everything resolves to those:

  EXPOSURE (3 classes, 17 ingredients): exact RxNorm Ingredient name -> concept_ancestor
      descendants (standard Drug) -> drug_concept_id sets. A drug concept reached from >=2
      STUDY classes (combination products) is excluded from every class arm and listed in
      dual_class_concept.csv; it REMAINS in the washout set.
  MED CLASSES (26): huilin_med name terms, SUBSTRING match on Ingredient concept names ->
      descendants -> drug_concept_id sets.
  DX SETS (outcomes x6, NCOs x90, T2D/T1D/ESRD): dotted ICD-10-CM PREFIX match on ICD10CM
      concept_code -> 'Maps to' SNOMED standard -> + concept_ancestor descendants (closure,
      so EHR rows coded straight into finer SNOMED still match).
  CLUSTERS (441 clusters / 66k ICD-10-CM codes): EXACT concept_code match -> 'Maps to'
      standard ONLY (no descendant closure: the 66k-code dictionary is already
      leaf-comprehensive; a SNOMED closure would explode it and blur cluster boundaries).

The dx_route=source path of build_cohort matches ICD-10 prefixes on source values and does
not use the dx sets resolved here (it still uses the drug maps and the OUT_*/NCO_* ids).

OUTPUT (intermediate/codes_resolved/):
  exposure_concept_map.csv   drug_class, drug_concept_id
  washout_concept.csv        drug_concept_id            (union of class sets + duals)
  dual_class_concept.csv     drug_concept_id, classes
  med_class_concept_map.csv  med_class, drug_concept_id
  dx_concept_map.csv         set_id, concept_id          (OUT_* / NCO_* / T2D / T1D / ESRD)
  cluster_concept_map.csv    set_id(=COV_*), concept_id
  resolution_summary.csv

Usage: python ty0_preprocessing/resolve_codes.py --source ynhh
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "8"
import argparse
import logging
import re

import pandas as pd

import codesets_tte as cs
from datasources.registry import get_source
from utils import setup_logging, time_execution


def parse_args():
    p = argparse.ArgumentParser(description="TTE OMOP concept resolution")
    p.add_argument("--source", default="ynhh", help="datasources/registry.py source")
    p.add_argument("--root", default=None, help="Override the OMOP root (else from the registry)")
    p.add_argument("--out_dir", default="intermediate/codes_resolved")
    return p.parse_args()


def connect(root):
    import duckdb
    from glob import glob
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{os.environ.get('DUCKDB_MEM', '4GB')}'")
    con.execute("PRAGMA threads=4")
    tmp = os.environ.get("TMPDIR") or "/tmp"
    con.execute(f"SET temp_directory='{tmp}/duckdb_t0_spill'")
    for name in ["concept", "concept_relationship", "concept_ancestor", "drug_strength"]:
        d = os.path.join(root, name)
        files = sorted(f for f in glob(os.path.join(d, "*.parquet"))
                       if not os.path.basename(f).startswith("._"))
        if not files:
            raise FileNotFoundError(f"No parquet for OMOP vocab table '{name}' under {d}")
        con.execute(f"CREATE OR REPLACE VIEW {name} AS "
                    f"SELECT * FROM read_parquet({files!r}, union_by_name=true)")
    return con


def resolve_ingredient_drugs(con, name_rows, substring=False):
    """(set_id, term) rows -> (set_id, drug_concept_id) via RxNorm Ingredient -> descendants.
    substring=True matches term as a substring of the ingredient name (med classes);
    False = exact ingredient name (exposure classes)."""
    con.execute("CREATE OR REPLACE TEMP TABLE gt(set_id VARCHAR, term VARCHAR)")
    con.executemany("INSERT INTO gt VALUES (?,?)", name_rows)
    match = ("lower(c.concept_name) LIKE '%' || g.term || '%'" if substring
             else "lower(c.concept_name) = g.term")
    return con.execute(f"""
        WITH ing AS (
            SELECT DISTINCT g.set_id, c.concept_id AS ing_id
            FROM gt g JOIN concept c
              ON c.concept_class_id='Ingredient' AND c.vocabulary_id IN ('RxNorm','RxNorm Extension')
             AND {match}
        ),
        drugs AS (
            SELECT i.set_id, ca.descendant_concept_id AS drug_id
            FROM ing i JOIN concept_ancestor ca ON ca.ancestor_concept_id=i.ing_id
            UNION
            SELECT set_id, ing_id FROM ing
        )
        SELECT DISTINCT d.set_id, d.drug_id AS drug_concept_id
        FROM drugs d JOIN concept c ON c.concept_id=d.drug_id
        WHERE c.domain_id='Drug' AND c.standard_concept='S'
    """).df()


# Set pairs the protocol declares MUTUALLY EXCLUSIVE. Their source definitions share zero
# ICD-10 codes (T1D = E10, T2D = E11), so any resolved-concept overlap is provably a mapping
# artifact and never lost signal: shared concepts are dropped from BOTH sets, then disjointness
# is HARD-ASSERTED (a shared concept cannot discriminate the pair; better to fail the build).
DISJOINT_PAIRS = [("T1D", "T2D")]


def resolve_dx_prefix(con, rows):
    """(set_id, dotted_icd10_prefix) -> (set_id, concept_id). Layered resolution policy:

    1. ICD10CM/ICD10 prefix -> 'Maps to' standard concepts.
    2. SINGLE-target codes (clean 1:1 mapping) keep their descendant closure -- needed because
       ~54% of YNHH condition rows are SNOMED-source-coded and can carry concepts finer than
       any Maps-to image.
    3. COMBINATION codes (>1 target) contribute their targets WITHOUT closure, and each target
       is kept only if it is a DESCENDANT-OR-SELF of one of the SAME set's single-mapped
       concepts: a combination code's extra targets are generic qualifiers ("Complication due
       to diabetes mellitus"), and closure through them merges sibling sets (T1D/T2D reach
       ~99.8% identity) while narrow NCOs swallow broad qualifier families. Descendant-of-own-
       singles keeps the type-specific halves ("Gangrene due to type 2 DM" IS a descendant of
       "Type 2 diabetes mellitus") and drops every qualifier. A set with no single-mapped
       concepts keeps its combination targets unfiltered (logged).
    4. Root guard: closure is NOT walked through a single target that is a strict ancestor of
       another OUT_* set's mapped concepts (or of the T1D/T2D counterpart) -- F34/F39 map 1:1
       onto the hierarchy root "Mood disorder", whose closure would swallow Depression and
       Bipolar whole. Such roots stay DIRECT hits: rows coded exactly there still match, the
       subtree does not.
    Returns columns [set_id, concept_id]."""
    con.execute("CREATE OR REPLACE TEMP TABLE dxq(set_id VARCHAR, source_code VARCHAR)")
    con.executemany("INSERT INTO dxq VALUES (?,?)", rows)
    return con.execute("""
        WITH src AS (
            SELECT DISTINCT q.set_id, c.concept_id AS src_id
            FROM dxq q JOIN concept c
              ON c.vocabulary_id IN ('ICD10CM','ICD10')
             AND (c.concept_code = q.source_code OR c.concept_code LIKE q.source_code || '%')
        ),
        mapped AS (
            SELECT DISTINCT s.set_id, s.src_id, cr.concept_id_2 AS std_id
            FROM src s JOIN concept_relationship cr
              ON cr.concept_id_1 = s.src_id AND cr.relationship_id = 'Maps to'
             AND cr.invalid_reason IS NULL
        ),
        deg AS (SELECT set_id, src_id, COUNT(DISTINCT std_id) AS n_targets
                FROM mapped GROUP BY 1, 2),
        single AS (SELECT DISTINCT m.set_id, m.std_id
                   FROM mapped m JOIN deg d USING (set_id, src_id) WHERE d.n_targets = 1),
        combo AS (SELECT DISTINCT m.set_id, m.std_id
                  FROM mapped m JOIN deg d USING (set_id, src_id) WHERE d.n_targets > 1),
        -- policy 3: combination targets kept only under the set's own single-mapped concepts;
        -- sets with NO singles keep their combos (cannot apply the rule without an anchor)
        no_single AS (SELECT DISTINCT set_id FROM mapped
                      EXCEPT SELECT DISTINCT set_id FROM single),
        combo_kept AS (
            SELECT DISTINCT c.set_id, c.std_id FROM combo c
            WHERE EXISTS (SELECT 1 FROM single s
                          JOIN concept_ancestor ca
                            ON ca.ancestor_concept_id = s.std_id
                           AND ca.descendant_concept_id = c.std_id
                          WHERE s.set_id = c.set_id)
               OR c.set_id IN (SELECT set_id FROM no_single)
        ),
        -- policy 4: single targets that are strict ancestors of another OUTCOME set's mapped
        -- concepts (or of the T1D/T2D counterpart) are hierarchy roots for this study: no closure
        roots AS (
            SELECT DISTINCT s.set_id, s.std_id FROM single s
            WHERE EXISTS (
                SELECT 1 FROM concept_ancestor ca
                JOIN mapped m2 ON m2.std_id = ca.descendant_concept_id
                WHERE ca.ancestor_concept_id = s.std_id
                  AND ca.ancestor_concept_id <> ca.descendant_concept_id
                  AND m2.set_id <> s.set_id
                  AND ((s.set_id LIKE 'OUT_%' AND m2.set_id LIKE 'OUT_%')
                       OR (s.set_id = 'T1D' AND m2.set_id = 'T2D')
                       OR (s.set_id = 'T2D' AND m2.set_id = 'T1D')))
        ),
        closure AS (
            SELECT s.set_id, ca.descendant_concept_id AS std_id
            FROM single s JOIN concept_ancestor ca ON ca.ancestor_concept_id = s.std_id
            WHERE NOT EXISTS (SELECT 1 FROM roots r
                              WHERE r.set_id = s.set_id AND r.std_id = s.std_id)
        ),
        allrows AS (SELECT * FROM single UNION SELECT * FROM combo_kept UNION SELECT * FROM closure)
        SELECT DISTINCT a.set_id, a.std_id AS concept_id
        FROM allrows a JOIN concept c ON c.concept_id = a.std_id AND c.standard_concept = 'S'
    """).df()


def resolve_dx_exact_mapsto(con, rows):
    """(set_id, dotted_icd10_code) -> (set_id, concept_id): EXACT ICD10CM code -> Maps to
    standard only (cluster dictionary; no closure)."""
    con.execute("CREATE OR REPLACE TEMP TABLE clq(set_id VARCHAR, source_code VARCHAR)")
    con.executemany("INSERT INTO clq VALUES (?,?)", rows)
    return con.execute("""
        SELECT DISTINCT q.set_id, cr.concept_id_2 AS concept_id
        FROM clq q
        JOIN concept c
          ON c.vocabulary_id IN ('ICD10CM','ICD10') AND c.concept_code = q.source_code
        JOIN concept_relationship cr
          ON cr.concept_id_1 = c.concept_id AND cr.relationship_id = 'Maps to'
         AND cr.invalid_reason IS NULL
        JOIN concept s ON s.concept_id = cr.concept_id_2 AND s.standard_concept = 'S'
    """).df()


def sanitize(name):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")


def redot(match_key: str) -> str:
    """codesets_tte normalizes to decimal-stripped keys; OMOP ICD10CM concept_code is dotted.
    Reinsert the dot after the 3rd character (ICD-10 convention) for codes longer than 3."""
    k = match_key.strip().upper()
    return k if len(k) <= 3 else f"{k[:3]}.{k[3:]}"


@time_execution
def main():
    log_file = setup_logging(level=logging.INFO)
    logging.info("RESOLVE CODES (OMOP) STARTING...")
    args = parse_args()
    root = args.root or get_source(args.source).get("root")
    logging.info("OMOP root: %s (source=%s)", root, args.source)
    os.makedirs(args.out_dir, exist_ok=True)
    con = connect(root)
    summary = []

    # ---- exposure classes (exact ingredient names) ----
    rows = [(cls, ing.lower()) for cls, ings in cs.EXPOSURE_INGREDIENTS.items() for ing in ings]
    expo = resolve_ingredient_drugs(con, rows, substring=False)
    expo["drug_concept_id"] = expo["drug_concept_id"].astype("int64").astype(str)
    per = expo.groupby("drug_concept_id")["set_id"].agg(lambda s: sorted(set(s)))
    dual = per[per.map(len) >= 2]
    dual_df = pd.DataFrame({"drug_concept_id": dual.index,
                            "classes": ["|".join(c) for c in dual.values]})
    washout = pd.DataFrame({"drug_concept_id": sorted(expo.drug_concept_id.unique())})
    expo_clean = (expo[~expo.drug_concept_id.isin(dual.index)]
                  .rename(columns={"set_id": "drug_class"}))
    logging.info("Exposure: %d drug concepts, %d dual-class excluded from arms (kept in washout)",
                 expo.drug_concept_id.nunique(), len(dual_df))
    for cls, g in expo_clean.groupby("drug_class"):
        logging.info("  %-8s %d concepts", cls, len(g))
        summary.append({"set": "exposure", "name": cls, "n": len(g),
                        "status": "ok" if len(g) else "UNMATCHED"})

    # ---- med classes (substring terms) ----
    med_rows = [(m, t) for m, terms in cs.med_classes().items() for t in terms]
    med = resolve_ingredient_drugs(con, med_rows, substring=True)
    med["drug_concept_id"] = med["drug_concept_id"].astype("int64").astype(str)
    med = med.rename(columns={"set_id": "med_class"}).drop_duplicates()
    for m, g in med.groupby("med_class"):
        summary.append({"set": "med_class", "name": m, "n": len(g),
                        "status": "ok" if len(g) else "UNMATCHED"})
    logging.info("Med classes: %d classes, %d drug concepts", med.med_class.nunique(),
                 med.drug_concept_id.nunique())

    # ---- dx sets: outcomes + NCOs + eligibility (prefix + closure) ----
    dx_rows = []
    for name, keys in cs.outcome_sets().items():
        dx_rows += [(f"OUT_{sanitize(name)}", redot(k)) for k in keys]
    for name, keys in cs.nco_sets().items():
        dx_rows += [(f"NCO_{sanitize(name)}", redot(k)) for k in keys]
    for name, keys in cs.covariate_exclusion_sets().items():
        dx_rows += [(name, redot(k)) for k in keys]        # T2D / T1D / ESRD
    dx = resolve_dx_prefix(con, dx_rows)
    dx["concept_id"] = dx["concept_id"].astype("int64").astype(str)

    # ---- declared-disjoint pairs: drop shared concepts from BOTH sides, then HARD-ASSERT ----
    # Licensed because the source definitions share zero ICD-10 codes (T1D=E10, T2D=E11): a
    # shared resolved concept is a mapping artifact that cannot discriminate the pair.
    for a, b in DISJOINT_PAIRS:
        sa = set(dx.loc[dx.set_id == a, "concept_id"])
        sb = set(dx.loc[dx.set_id == b, "concept_id"])
        shared = sa & sb
        if shared:
            names = con.execute(
                "SELECT concept_id, concept_name FROM concept WHERE concept_id IN (%s)"
                % ",".join(shared)).df()
            for _, r in names.iterrows():
                logging.warning("  DISJOINT-PAIR %s/%s: dropping shared concept %s '%s' from both",
                                a, b, r.concept_id, r.concept_name)
            dx = dx[~(dx.set_id.isin([a, b]) & dx.concept_id.isin(shared))]
        sa = set(dx.loc[dx.set_id == a, "concept_id"])
        sb = set(dx.loc[dx.set_id == b, "concept_id"])
        assert not (sa & sb), f"{a}/{b} still overlap after curation -- resolver bug, refusing to continue"
        logging.info("  DISJOINT-PAIR %s/%s: %d / %d concepts, 0 shared (asserted)", a, b, len(sa), len(sb))

    # ---- containment audit ----
    # WARN, not fail: some containment is genuine (a complication NCO nested in its disease NCO);
    # near-100% containment between two OUTCOME sets is a resolver bug and must be investigated.
    _sets = {k: set(g.concept_id) for k, g in dx.groupby("set_id")}
    import itertools
    for a, b in itertools.combinations(sorted(_sets), 2):
        ov = _sets[a] & _sets[b]
        if not ov:
            continue
        frac = len(ov) / min(len(_sets[a]), len(_sets[b]))
        if frac > 0.5:
            lvl = logging.ERROR if (a.startswith("OUT_") and b.startswith("OUT_")) else logging.WARNING
            logging.log(lvl, "  CONTAINMENT %s vs %s: %d shared = %.0f%% of the smaller set%s",
                        a, b, len(ov), 100 * frac,
                        " -- OUTCOME sets must be independent, investigate" if lvl == logging.ERROR else "")

    # Domain audit: OMOP routes Observation-domain standard concepts to the `observation`
    # table, so a set resolving there matches nothing in condition_occurrence. Record domain_id
    # in the map and WARN per set on any non-Condition share so this surfaces here, not
    # silently at build_cohort. The id list is registered as int64 (some pandas/duckdb pairs
    # cannot register the post-.astype(str) column).
    con.register("dx_resolved_tmp", dx[["concept_id"]].drop_duplicates().astype({"concept_id": "int64"}))
    dom = con.execute("""
        SELECT CAST(c.concept_id AS VARCHAR) AS concept_id, c.domain_id
        FROM concept c JOIN dx_resolved_tmp t ON c.concept_id = t.concept_id
    """).df()
    dom["concept_id"] = dom["concept_id"].astype(str)
    dx = dx.merge(dom, on="concept_id", how="left")
    n_by_set = dx.groupby("set_id").size()
    noncond = dx[dx.domain_id != "Condition"].groupby("set_id").size()
    for sid in sorted({r[0] for r in dx_rows}):
        n = int(n_by_set.get(sid, 0))
        nc = int(noncond.get(sid, 0))
        status = "ok" if n else "UNMATCHED"
        if n and nc:
            status = f"DOMAIN-WARN {nc}/{n} non-Condition"
            logging.warning("  DOMAIN-WARN dx set %s: %d/%d resolved concepts are NOT Condition-domain "
                            "(they live outside condition_occurrence; build_cohort's dx_long UNION obs_long "
                            "stream covers Observation-domain rows -- other domains will not match)",
                            sid, nc, n)
        summary.append({"set": "dx", "name": sid, "n": n, "status": status})
        if n == 0:
            logging.warning("  UNMATCHED dx set: %s", sid)
    logging.info("Dx sets: %d sets, %d (set, concept) rows", dx.set_id.nunique(), len(dx))

    # ---- clusters (exact + maps-to) ----
    cl_rows = [(f"COV_{sanitize(cl)}", redot(code)) for code, cl in cs.cluster_master()]
    clusters = resolve_dx_exact_mapsto(con, cl_rows)
    clusters["concept_id"] = clusters["concept_id"].astype("int64").astype(str)
    con.register("cl_resolved_tmp", clusters[["concept_id"]].drop_duplicates().astype({"concept_id": "int64"}))
    dom_cl = con.execute("""
        SELECT CAST(c.concept_id AS VARCHAR) AS concept_id, c.domain_id
        FROM concept c JOIN cl_resolved_tmp t ON c.concept_id = t.concept_id
    """).df()
    dom_cl["concept_id"] = dom_cl["concept_id"].astype(str)
    clusters = clusters.merge(dom_cl, on="concept_id", how="left")
    ncl = int((clusters.domain_id != "Condition").sum())
    if ncl:
        logging.warning("  DOMAIN-WARN clusters: %d/%d resolved concepts are NOT Condition-domain",
                        ncl, len(clusters))
    logging.info("Clusters: %d of 441 clusters resolved, %d (cluster, concept) rows",
                 clusters.set_id.nunique(), len(clusters))
    summary.append({"set": "clusters", "name": "_total", "n": len(clusters),
                    "status": f"{clusters.set_id.nunique()}/441 clusters"})

    expo_clean.to_csv(os.path.join(args.out_dir, "exposure_concept_map.csv"), index=False)
    washout.to_csv(os.path.join(args.out_dir, "washout_concept.csv"), index=False)
    dual_df.to_csv(os.path.join(args.out_dir, "dual_class_concept.csv"), index=False)
    med.to_csv(os.path.join(args.out_dir, "med_class_concept_map.csv"), index=False)
    dx.to_csv(os.path.join(args.out_dir, "dx_concept_map.csv"), index=False)
    clusters.to_csv(os.path.join(args.out_dir, "cluster_concept_map.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "resolution_summary.csv"), index=False)
    logging.info("RESOLVE CODES (OMOP) COMPLETED -> %s. Log: %s", args.out_dir, log_file)


if __name__ == "__main__":
    main()
