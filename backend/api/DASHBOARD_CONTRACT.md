# Dashboard response

`GET /api/pumps/{pump_id}/dashboard?refresh=true` returns the operator view by
default. State and inference are unchanged. Both the React dashboard and the
Streamlit fallback retain the fields they consume.

Always included: identity, timestamp, running status, final state/label, the
sensor values rendered by the KPI cards and telemetry table, baseline ranges,
and chart series.

`issue_tiles` is an always-present backend-owned catalog for the client issue
grid. Each entry contains `code`, `label`, `active`, and `affected_metrics`;
active faults also include their API evidence. The client renders this array
instead of maintaining its own fault-name list. `UNUSUAL_PUMP_BEHAVIOUR` is
active only when the final state is `ANOMALY` and no named fault was returned.
Unknown future fault codes are appended as active tiles automatically; adding a
definition in `backend/api/issue_catalog.py` makes a new tile permanently
visible even while inactive.

When the latest reported pump status is OFF, refresh still analyzes the latest
available ON session. `running_status` remains `OFF`, while state, faults,
warnings, snapshot values and trends describe that last ON session.
`analysis_context` is `LAST_ON_SESSION`, `last_on_timestamp` identifies the
analyzed reading, and `status_timestamp` identifies the latest actual status
row. ON pumps use `LIVE_ON_SESSION`.

State-specific fields are included only when they have UI content: named faults,
active early warnings, pending-baseline detail, and failed data-quality reasons.
Empty warning/fault arrays, PASS data-quality metadata, and null pending detail
are omitted.

`affected_readings` is a compact presentation projection of backend-returned
deviations/warnings. It tells the client which KPI card to mark as `ALERT` or
`WARNING` and its HIGH/LOW direction. It does not make a new anomaly decision.

Health score, generic deviations, duplicate summary/status flags, repeated
detailed issues, baseline metadata/statistics, model diagnostics, imputation
diagnostics and short-cycling internals are omitted from the
operator response. An `include_details` query parameter is intentionally
ignored; diagnostics cannot be enabled on the public operator endpoint.

Detailed diagnostics are available through:

```
GET /api/admin/pumps
GET /api/admin/pumps/{pump_id}/diagnostics?refresh=true
POST /api/admin/pumps/{pump_id}/reinitialize
```

The fleet list is cache-only; only the selected pump diagnostics endpoint
performs an on-demand refresh. Reinitialization is grouped under the admin API;
prediction remains directly accessible like the existing pump APIs.

Charts are the largest part of the response and remain enabled by default.
For a current-condition-only consumer, use `include_trends=false`:

```
/api/pumps/{pump_id}/dashboard?refresh=true&include_trends=false
```

These options only affect serialization. They do not change the 350-row live
context, model features, baseline selection, fault rules or final decision.
The operator and diagnostics payloads are projections of the same worker cache
entry, so neither API duplicates detection or threshold logic.
