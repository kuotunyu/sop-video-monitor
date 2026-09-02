"""sop-video-monitor: multi-camera SOP sequence monitoring (W0 skeleton).

Package layout mirrors the design spec (2026-09-03, sections 5 and 8.1):

- ``splits``          subject-wise split freezing and the test-list SHA-256 contract
- ``sop_graph``       task precedence graph (OWL -> ``sop/graph.json``) and the three SOP checks
- ``stream``          ring buffer with WAIT / DROP_OLDEST / ADAPTIVE backpressure policies
- ``metrics.offline`` MoF, Edit, F1@{10,25,50} with subject bootstrap
- ``metrics.online``  Procedure Order Similarity, completion F1, detection delay
- ``cli``             typer entry point (``sop-monitor``)
"""

__version__ = "0.0.1"
