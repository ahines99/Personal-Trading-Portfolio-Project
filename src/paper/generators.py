from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd


def _coerce_tabular_weights(
    value: Any,
    *,
    weight_candidates: Iterable[str],
    ticker_key: str = "ticker",
) -> pd.DataFrame:
    """Normalize common target/prior inputs into [ticker, weight, price?] rows."""
    if value is None:
        return pd.DataFrame(columns=[ticker_key, "weight", "price"])

    if isinstance(value, pd.Series):
        frame = value.dropna().rename("weight").reset_index()
        frame.columns = [ticker_key, "weight"]
        return frame.assign(price=pd.NA)

    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        lowered = {str(col).lower(): col for col in frame.columns}
        if ticker_key in lowered:
            ticker_col = lowered[ticker_key]
            weight_col = None
            for candidate in weight_candidates:
                if candidate.lower() in lowered:
                    weight_col = lowered[candidate.lower()]
                    break
            if weight_col is None and "weight" in lowered:
                weight_col = lowered["weight"]
            if weight_col is None:
                raise ValueError("Could not find a weight column in the provided DataFrame")
            price_col = None
            for candidate in ("price", "last_price", "reference_price"):
                if candidate in lowered:
                    price_col = lowered[candidate]
                    break
            cols = [ticker_col, weight_col]
            if price_col is not None:
                cols.append(price_col)
            out = frame.loc[:, cols].copy()
            out.columns = [ticker_key, "weight"] + (["price"] if price_col is not None else [])
            if "price" not in out.columns:
                out["price"] = pd.NA
            return out

        if frame.empty:
            return pd.DataFrame(columns=[ticker_key, "weight", "price"])

        latest = frame.iloc[-1].dropna()
        out = latest.rename("weight").reset_index()
        out.columns = [ticker_key, "weight"]
        out["price"] = pd.NA
        return out

    if isinstance(value, Mapping):
        records = []
        for ticker, payload in value.items():
            if isinstance(payload, Mapping):
                weight = payload.get("weight", payload.get("target_weight", payload.get("prior_weight", 0.0)))
                price = payload.get("price", payload.get("last_price", pd.NA))
            elif is_dataclass(payload):
                data = asdict(payload)
                weight = data.get("weight", data.get("target_weight", data.get("prior_weight", 0.0)))
                price = data.get("price", data.get("last_price", pd.NA))
            else:
                weight = payload
                price = pd.NA
            records.append({ticker_key: ticker, "weight": weight, "price": price})
        return pd.DataFrame.from_records(records, columns=[ticker_key, "weight", "price"])

    raise TypeError(f"Unsupported weights input type: {type(value)!r}")


def _prepare_weights(value: Any, *, role: str) -> pd.DataFrame:
    frame = _coerce_tabular_weights(
        value,
        weight_candidates=("weight", f"{role}_weight"),
    ).copy()
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "weight", "price"])

    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame = frame[frame["ticker"] != ""]
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
    frame["price"] = pd.to_numeric(frame.get("price"), errors="coerce")
    frame = (
        frame.groupby("ticker", as_index=False)
        .agg({"weight": "sum", "price": "last"})
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    return frame


def compute_intended_trades(
    target_weights: Any,
    prior_holdings: Any,
    notional: float = 100_000,
    *,
    max_single_order_notional: Optional[float] = None,
    noise_threshold: float = 100.0,
) -> pd.DataFrame:
    """Diff target weights against prior holdings into an intended trade blotter."""
    if notional <= 0:
        raise ValueError("notional must be positive")
    if noise_threshold < 0:
        raise ValueError("noise_threshold cannot be negative")

    target = _prepare_weights(target_weights, role="target").rename(
        columns={"weight": "target_weight", "price": "target_price"}
    )
    prior = _prepare_weights(prior_holdings, role="prior").rename(
        columns={"weight": "prior_weight", "price": "prior_price"}
    )

    merged = target.merge(prior, on="ticker", how="outer")
    merged["target_weight"] = merged["target_weight"].fillna(0.0)
    merged["prior_weight"] = merged["prior_weight"].fillna(0.0)
    merged["reference_price"] = merged["target_price"].combine_first(merged["prior_price"])
    merged["prior_notional"] = merged["prior_weight"] * float(notional)
    merged["target_notional"] = merged["target_weight"] * float(notional)
    merged["delta_notional"] = merged["target_notional"] - merged["prior_notional"]

    if max_single_order_notional is not None:
        oversize = merged.loc[merged["delta_notional"].abs() > float(max_single_order_notional), "ticker"]
        if not oversize.empty:
            bad = ", ".join(oversize.tolist())
            raise ValueError(
                f"Trade notional exceeds max_single_order_notional={max_single_order_notional}: {bad}"
            )

    merged = merged.loc[merged["delta_notional"].abs() >= float(noise_threshold)].copy()
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "action",
                "prior_weight",
                "target_weight",
                "prior_notional",
                "target_notional",
                "delta_notional",
                "shares_to_trade",
            ]
        )

    merged["action"] = merged["delta_notional"].apply(lambda v: "BUY" if v > 0 else "SELL")
    merged["shares_to_trade"] = pd.NA
    price_mask = merged["reference_price"].notna() & (merged["reference_price"] > 0)
    merged.loc[price_mask, "shares_to_trade"] = (
        merged.loc[price_mask, "delta_notional"] / merged.loc[price_mask, "reference_price"]
    )

    merged["sort_bucket"] = merged["action"].map({"BUY": 0, "SELL": 1}).fillna(2)
    merged["sell_sort"] = merged["delta_notional"].abs()
    merged = merged.sort_values(
        by=["sort_bucket", "delta_notional", "sell_sort", "ticker"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )

    out = merged.loc[
        :,
        [
            "ticker",
            "action",
            "prior_weight",
            "target_weight",
            "prior_notional",
            "target_notional",
            "delta_notional",
            "shares_to_trade",
        ],
    ].reset_index(drop=True)

    numeric_cols = [
        "prior_weight",
        "target_weight",
        "prior_notional",
        "target_notional",
        "delta_notional",
        "shares_to_trade",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
