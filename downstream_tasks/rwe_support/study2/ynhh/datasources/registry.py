#!/usr/bin/env python3
"""
datasources/registry.py -- the data-source registry + adapter factory.

One place to declare an OMOP data source: its name, where it lives, and site-specific
params. `--source <name>` looks up this table; adding a source is a dict entry (+ an overlay
yaml), not new pipeline code.
"""

import os

# Resolve paths relative to the module root (this file's parent's parent).
_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASOURCES = {
    # -- YNHH: the OMOP source (root from YNHH_OMOP_ROOT in env.sh) --
    # Layout of the YNHH OMOP export (summarize your own with datasources/catalog_parquet_db.py);
    # export quirks the OMOPAdapter absorbs:
    # person.year_of_birth all NULL (DOBYR from date_of_birth/birth_datetime), *_date columns
    # NULL (reads coalesce date<-datetime), no observation_period table (observability from
    # the visit_occurrence footprint), no death table (person.death_datetime).
    "ynhh": {
        "data_model": "omop",
        "root": os.environ.get("YNHH_OMOP_ROOT", ""),
        "strata": ["ynhh"],
        "inpatient_visit_concept_ids": [9201, 262],
        # Condition-status keep-set (verified on YNHH inpatient dx): 32902 Primary, 32890 Admission,
        # 4174609 Hospital-acquired (the nearest secondary analogue there; 32908 Secondary is
        # absent but kept for portability).
        "dx_status_concept_ids": [32902, 32908, 32890, 4174609],
        "config_overlay": "config/overlays/ynhh.yaml",
        "spark_profile": "bigmem",          # large export (millions of persons, >1B conditions)
        "status": "active",
    },
}


def get_source(name: str) -> dict:
    if name not in DATASOURCES:
        raise KeyError(f"Unknown data source '{name}'. Known: {sorted(DATASOURCES)}")
    return DATASOURCES[name]


def overlay_path(name: str):
    """Absolute path to the source's config overlay yaml, or None."""
    ov = get_source(name).get("config_overlay")
    if not ov:
        return None
    return ov if os.path.isabs(ov) else os.path.join(_MODULE_ROOT, ov)


def default_stratum(name: str) -> str:
    """The default analysis cell for a source: for OMOP this is the source name itself."""
    return get_source(name)["strata"][0]


def make_adapter(source: str, spark, *, stratum=None,
                 codes_resolved_root="intermediate/codes_resolved", cfg=None, restrict_ids=None):
    """Construct the adapter for `source` (OMOP only): the registry supplies the root + site
    params. Adapter classes are imported lazily."""
    entry = get_source(source)
    model = entry["data_model"]
    if model == "omop":
        from datasources.omop import OMOPAdapter
        root = entry.get("root")
        if root is None:
            raise NotImplementedError(
                f"Data source '{source}' is a placeholder: set DATASOURCES['{source}']['root'] to its "
                f"OMOP parquet directory (and confirm inpatient_visit_concept_ids + the config overlay).")
        return OMOPAdapter(spark, source_name=source, root=root,
                           inpatient_visit_concept_ids=entry.get("inpatient_visit_concept_ids", [9201, 262]),
                           dx_status_concept_ids=entry.get("dx_status_concept_ids", [32902, 32908, 4203942]),
                           cfg=cfg, codes_resolved_root=codes_resolved_root,
                           restrict_ids=restrict_ids)
    raise ValueError(f"Unknown data_model '{model}' for source '{source}'")
