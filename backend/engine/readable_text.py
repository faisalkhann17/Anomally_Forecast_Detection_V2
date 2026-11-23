"""Plain-language names, units and counts for text shown to pump operators."""
from __future__ import annotations

# field -> (name shown to the operator, unit)
FIELDS: dict[str, tuple[str, str]] = {
    "current_r": ("Phase R current", "A"),
    "current_y": ("Phase Y current", "A"),
    "current_b": ("Phase B current", "A"),
    "current_avg": ("Motor current", "A"),
    "voltage_r": ("Phase R voltage", "V"),
    "voltage_y": ("Phase Y voltage", "V"),
    "voltage_b": ("Phase B voltage", "V"),
    "voltage_avg": ("Voltage", "V"),
    "pf_r": ("Phase R power factor", ""),
    "pf_y": ("Phase Y power factor", ""),
    "pf_b": ("Phase B power factor", ""),
    "pf_avg": ("Power factor", ""),
    "total_kw": ("Power", "kW"),
    "flowrate": ("Water flow", "m³/h"),
    "current_imbalance_pct": ("Current imbalance between phases", "%"),
    "voltage_imbalance_pct": ("Voltage imbalance between phases", "%"),
}


def field_name(field: str) -> str:
    """Operator name for a telemetry field, e.g. current_r -> Phase R current."""
    return FIELDS.get(field, (field.replace("_", " ").capitalize(), ""))[0]


def with_unit(value: float, field: str) -> str:
    """Value rounded for reading, followed by the field's unit."""
    unit = FIELDS.get(field, ("", ""))[1]
    digits = 2 if field.startswith("pf_") else 1
    text = f"{float(value):.{digits}f}"
    if not unit:
        return text
    return f"{text}{unit}" if unit == "%" else f"{text} {unit}"


def count(number: int | float, word: str) -> str:
    """'1 day', '3 days', '1 other pump', '2 other pumps'."""
    number = int(round(float(number)))
    return f"{number} {word}" if number == 1 else f"{number} {word}s"
