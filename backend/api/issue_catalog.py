"""Backend-owned issue tile definitions for operator clients."""
from __future__ import annotations


FAULT_LABELS = {
    "DRY_RUN": "Dry Run",
    "LOW_PF": "Low Power Factor",
    "PHASE_CURRENT_IMBALANCE": "Phase Current Imbalance",
    "VOLTAGE_IMBALANCE": "Voltage Imbalance",
    "PHASE_LOSS": "Possible Phase Loss",
    "VOLTAGE_PHASE_LOSS": "Possible Voltage Phase Loss",
    "UNDER_VOLTAGE": "Under Voltage",
    "OVER_VOLTAGE": "Over Voltage",
    "MOTOR_OVERLOAD": "Possible Motor Overload",
    "FLOW_RESTRICTION": "Possible Flow Restriction",
    "SHORT_CYCLING": "Short Cycling",
}


ISSUE_TILE_DEFINITIONS = (
    {
        "code": "DRY_RUN",
        "label": "Possible Dry Run",
        "affected_metrics": ["flow", "current", "power"],
    },
    {
        "code": "LOW_PF",
        "label": "Low Power Factor",
        "affected_metrics": ["pf"],
    },
    {
        "code": "PHASE_CURRENT_IMBALANCE",
        "label": "Phase Current Imbalance",
        "affected_metrics": ["current"],
    },
    {
        "code": "VOLTAGE_IMBALANCE",
        "label": "Voltage Imbalance",
        "affected_metrics": ["voltage"],
    },
    {
        "code": "PHASE_LOSS",
        "label": "Possible Phase Loss",
        "affected_metrics": ["current"],
    },
    {
        "code": "VOLTAGE_PHASE_LOSS",
        "label": "Possible Voltage Phase Loss",
        "affected_metrics": ["voltage"],
    },
    {
        "code": "UNDER_VOLTAGE",
        "label": "Under Voltage",
        "affected_metrics": ["voltage"],
    },
    {
        "code": "OVER_VOLTAGE",
        "label": "Over Voltage",
        "affected_metrics": ["voltage"],
    },
    {
        "code": "MOTOR_OVERLOAD",
        "label": "Possible Motor Overload",
        "affected_metrics": ["current", "power"],
    },
    {
        "code": "FLOW_RESTRICTION",
        "label": "Possible Flow Restriction",
        "affected_metrics": ["flow", "current", "power"],
    },
    {
        "code": "SHORT_CYCLING",
        "label": "Short Cycling",
        "affected_metrics": ["current"],
    },
    {
        "code": "UNUSUAL_PUMP_BEHAVIOUR",
        "label": "Unusual Pump Behaviour",
        "affected_metrics": [],
    },
)


def build_issue_tiles(payload: dict) -> list[dict]:
    """Return the complete tile catalog with active API faults marked."""
    state = str(payload.get("state") or "UNKNOWN").upper()
    faults = payload.get("detected_faults") or []
    active_faults = {
        str(fault.get("code") or "").upper(): fault
        for fault in faults
        if str(fault.get("code") or "").strip()
    }
    anomaly = state in {"ANOMALY", "PROVISIONAL_ANOMALY"}
    known_codes = {definition["code"] for definition in ISSUE_TILE_DEFINITIONS}
    tiles = []

    for definition in ISSUE_TILE_DEFINITIONS:
        code = definition["code"]
        fault = active_faults.get(code)
        active = bool(anomaly and fault)
        if code == "UNUSUAL_PUMP_BEHAVIOUR":
            active = bool(anomaly and not active_faults)

        tile = {
            "code": code,
            "label": definition["label"],
            "active": active,
            "affected_metrics": list(definition["affected_metrics"]),
        }
        if active and fault:
            tile["evidence"] = fault.get("evidence")
        tiles.append(tile)

    # A newly returned backend fault remains renderable even before the client
    # has seen its code. Adding it to ISSUE_TILE_DEFINITIONS later makes it a
    # permanently visible inactive tile too.
    for code, fault in active_faults.items():
        if code in known_codes or not anomaly:
            continue
        tiles.append({
            "code": code,
            "label": fault.get("label") or code.replace("_", " ").title(),
            "active": True,
            "affected_metrics": [],
            "evidence": fault.get("evidence"),
        })

    return tiles
