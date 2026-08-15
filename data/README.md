# Data policy

This repository is synthetic-only. The demo creates a deterministic panel with
dates, synthetic ticker IDs, and the columns `open`, `high`, `low`, `close`,
and `volume`. No downloaded, licensed, restricted, or user-provided market
data is stored here.

To plug in public data, provide a pandas DataFrame with a sorted MultiIndex
named `(date, ticker)` and the five expected numeric columns. Keep any data
acquisition outside this repository, check its license, and do not redistribute
data that the license does not permit.
