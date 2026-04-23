Dataset: NSL-KDD

The NSL-KDD dataset is an improved version of the KDD Cup 1999 intrusion
detection dataset.

It contains network traffic records labeled as either normal or belonging
to one of several attack types.

Attack types are grouped into four main categories:

DoS  – denial of service
Probe – reconnaissance attacks
R2L  – remote-to-local attacks
U2R  – user-to-root attacks

Features include protocol information, connection statistics,
and service metadata.

Categorical features were converted using one-hot encoding,
and numerical features were scaled using StandardScaler.