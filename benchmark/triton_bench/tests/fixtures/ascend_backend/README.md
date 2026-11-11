# Real backend dump fixtures (recognizer tests only)

Source: diagnostic-only record `20261026-155402-979553`, result commit
`7c3011e6`, executed code `c52c24591115deac0bc69218b152ae805a69c471`.
Files are the byte-exact decompressed `probe-stderr` blobs in
`evidence/{Neg,Matmul}/vendor-whitelist/replay.json`.

| File | Uncompressed SHA256 |
| --- | --- |
| Neg.txt | 09b77a5def80e4a05ba142fdb5f8820693f08cd2f48cefffa32423e3f12679bc |
| Matmul.txt | dccaad7cef930e608216b562564094fc80a58f28f861d3ca67e32b2dba7a702f |

This replay compiled two archived inputs with two printing configurations. It
did **not** execute NPU kernels. These fixtures test the parser and negative
mutations only; no test result or diagnostic artifact grants compilation or
execution coverage. Formal coverage still requires a fresh frozen-suite run,
byte-exact adapter/binary binding, submitted source and TTIR provenance, and
the independent correctness/reference/synchronization gates.
