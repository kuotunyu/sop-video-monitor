"""sop-video-monitor: multi-camera SOP sequence monitoring (development build).

Package layout mirrors the design spec (2026-09-03, sections 5 and 8.1):

- ``splits``          subject-wise split freezing and the test-list SHA-256 contract
- ``sop_graph``       task precedence graph (OWL -> ``sop/graph.json``) and the three SOP checks
- ``stream``          ring buffer with WAIT / DROP_OLDEST / ADAPTIVE backpressure policies
- ``metrics.offline`` MoF, Edit, F1@{10,25,50} with subject bootstrap
- ``metrics.online``  Procedure Order Similarity, completion F1, detection delay
- ``industreal``      IndustReal labels, split freezing, segment -> per-frame alignment
- ``video``           PyAV probe / sequential decode
- ``features``        frozen DINOv2 frame embeddings cached per video
- ``baseline``        linear head, causal / centered smoothing, prediction tables, scoring
- ``audit``           on-disk data audit and ``data/manifest.json``
- ``cli``             typer entry point (``sop-monitor``)
"""

__version__ = "0.0.1"
