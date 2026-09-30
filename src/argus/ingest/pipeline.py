from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from argus.ingest.geoip import enrich
from argus.ingest.parsers import parse_csv, parse_json, parse_xml
from argus.ingest.schema import CANONICAL_FIELDS
from argus.ingest.validate import validate_and_coerce

PARSERS = {"csv": parse_csv, "json": parse_json, "xml": parse_xml}


def run_ingest(input_path: Path, fmt: str, rejects_path: Path) -> pd.DataFrame:
    parser = PARSERS[fmt]
    valid_rows = []

    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rejects_path, "w", encoding="utf-8") as rejects_f:
        for index, raw_row in enumerate(parser(input_path)):
            coerced, reason = validate_and_coerce(raw_row)
            if reason is not None:
                rejects_f.write(
                    json.dumps(
                        {
                            "source_format": fmt,
                            "row_index": index,
                            "reason": reason,
                            "txid": raw_row.get("txid"),
                        }
                    )
                    + "\n"
                )
                continue

            enriched = enrich(coerced["src_ip"])
            if enriched is None:
                rejects_f.write(
                    json.dumps(
                        {
                            "source_format": fmt,
                            "row_index": index,
                            "reason": "unresolvable_geoip",
                            "txid": raw_row.get("txid"),
                        }
                    )
                    + "\n"
                )
                continue

            asn, geo_country = enriched
            coerced["asn"] = asn
            coerced["geo_country"] = geo_country
            valid_rows.append(coerced)

    return pd.DataFrame(valid_rows, columns=CANONICAL_FIELDS)


def write_canonical(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
