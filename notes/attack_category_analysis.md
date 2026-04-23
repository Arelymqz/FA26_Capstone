Attack Category Detection Analysis

Detection performance varies significantly by attack category.

Results at threshold 0.50:

DoS detection rate: 82%
Probe detection rate: 75%
U2R detection rate: 8.5%
R2L detection rate: 1.1%

Key Observation

The classifier detects network-based attacks (DoS and Probe)
effectively but struggles with more subtle attacks such as
R2L and U2R.

This indicates that the current feature set and model
favor easily detectable traffic anomalies while failing
to identify attacks that resemble legitimate behavior.