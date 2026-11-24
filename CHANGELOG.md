# Changelog

All notable changes to the PumpWatch backend. Versions follow
[Semantic Versioning](https://semver.org/): MAJOR for breaking API or
configuration changes, MINOR for new features, PATCH for fixes.

## [1.1.2] - 2026-11-25

### Fixed
- Live readings were about 5.5 hours old on the production server. Its devices write
  IST wall-clock time into InfluxDB labelled as UTC, so the stored `_time` runs
  5h30m ahead of real time, and windows such as "last 1 hour" up to `now()` returned
  readings from 5.5 hours earlier. New setting `INFLUX_TIME_OFFSET_MINUTES` (default
  0; production 330): every telemetry query window moves by the offset and every
  reading's time is converted back to real UTC, so live state, data age, charts,
  history and limit suggestions use real time. Packets whose device clock is even
  further ahead are left out.

## [1.1.1] - 2026-11-24

### Admin access
- `API_KEY_REQUIRED=false` lets the profile and admin routes work without an
  `X-API-Key` (the dashboard shows the admin setup to admins only). Changes made this
  way are recorded as `admin`. The default stays `true` (key required).

## [1.1.0] - 2026-11-24

### Production safety and load
- `INFLUX_WRITES_ENABLED=false` keeps the service read-only on InfluxDB; the hourly
  forecast history is kept in memory instead.
- `PROFILE_MONGODB_ALLOW_REMOTE=true` allows engineer profiles and issue history on a
  remote MongoDB server (local-only by default).
- The forecast and the slow-wear checks now run as one background job every 12 hours
  (`FORECAST_REFRESH_MINUTES=720`) and share one history read per pump.
- History is read one day per query, only the needed fields, one reading per
  5 minutes, with a short pause between queries; later cycles read only new data.
  Days that InfluxDB cannot return are skipped instead of failing the pump.

### Engineer limits
- `GET /api/pumps/{id}/profile/limit-suggestion`: suggests nameplate values and limits
  from the pump's last 30 days of running data, with warnings and the alarms the
  limits would have raised. Nothing is saved; the engineer reviews and saves.
- "Check minimum flow from data" no longer fails when meters send text readings such
  as "-".
- The dashboard "usual value / usual range" now shows the approved limits (approx).

### Dashboard
- While a pump is off, the dashboard returns its last running readings and health
  score (`last_run`); `last_on_timestamp` is the real last running time.

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
