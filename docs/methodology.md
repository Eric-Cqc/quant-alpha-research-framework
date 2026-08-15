# Methodology

The framework makes the research workflow explicit:

1. Generate or load a point-in-time panel.
2. Describe candidate signals with immutable `SignalSpec` records.
3. Validate fields, parameters, horizons, function syntax, alignment, and
   truncation invariance.
4. Remove exact duplicates using a canonical family-plus-parameter signature.
5. Build capped, dollar-neutral cross-sectional portfolios. A signal observed
   on date `t` earns the return on date `t+1`.
6. Score performance, apply risk gates, filter correlated return streams, and
   apply family caps.
7. Evaluate rolling train/validation/test windows and write lineage plus plots.

The candidate funnel is intentionally inspectable. Every reduction has a count
and a short reason in `funnel.csv`, while `lineage.json` records configuration,
data, stage, and selection hashes.
