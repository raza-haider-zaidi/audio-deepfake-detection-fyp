"""Local URL Ingestion Helper.

A small, local-only Windows application that retrieves public YouTube
audio from the machine it runs on (where retrieval already works) and
exposes it to the deployed Streamlit application over a temporary,
authenticated Cloudflare Quick Tunnel -- see docs/local_url_helper.md for
the full architecture.

This package is intentionally separate from the Streamlit application
(`app/`): it has its own, much smaller dependency set (FastAPI/Uvicorn,
no ML/inference libraries) so it can be built into a standalone Windows
executable without bundling model code. It DOES reuse the existing,
already-tested public-video retrieval logic in `app.analysis.video_url`
and `app.analysis.media_ffmpeg` rather than duplicating it -- those two
modules have no dependency on Streamlit, ONNX Runtime, or any model code
(see local_helper/youtube.py).
"""
