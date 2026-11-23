# Changelog

All notable changes to the PumpWatch backend. Versions follow
[Semantic Versioning](https://semver.org/): MAJOR for breaking API or
configuration changes, MINOR for new features, PATCH for fixes.

## [1.0.0] - 2026-11-23

First tagged release.

### Live monitoring
- Engineering checks against engineer-approved limits: motor overload, phase
  loss, voltage phase loss, under/over voltage, current and voltage imbalance,
  low power factor, high power, dry run, flow restriction and short cycling.
- Early warnings when a reading moves 15% away from the pump's usual level
  (median of the last 7 days of running hours) or shifts suddenly in a run.
- Health score and plain-English issue descriptions for the dashboard.

### Trends and forecast
- Slow-wear checks on hourly running data with an estimated time to the limit.
- Water and energy performance (flow per kW, energy per m³).
- Timestral-123M hourly forecast turned into advisory limit-crossing risk.

### Engineer limits
- Profiles stored in MongoDB with change history; edits need approval.
- Every profile field is required; minimum delivery flow must be above 0.
- `GET /api/pumps/{id}/profile/flow-suggestion` suggests the minimum flow from
  the pump's running data (IQR outliers removed, lowest normal flow minus 30%).

### Storage
- Only problems are stored: open/resolved issue events in MongoDB.
