# Vendored: bgutil-ytdlp-pot-provider

- **Upstream**: https://github.com/Brainicism/bgutil-ytdlp-pot-provider
- **Pinned tag**: `1.3.2`
- **Pinned commit**: `7511309af023b09788dc8f2efc96cc3671291e6c`
- **License**: GPL-3.0-only (see `LICENSE` in this directory, copied verbatim from upstream, unmodified)
- **What is vendored**: only the `server/` directory (the Deno/Node PO-token
  generation source, including its own `LICENSE`-covered `package.json`,
  `deno.json`, `deno.lock`, and `src/`), plus the top-level `LICENSE` and
  `README.md`. The upstream `plugin/` directory (the Python yt-dlp plugin)
  is intentionally NOT vendored -- it is installed instead via the pinned
  PyPI package `bgutil-ytdlp-pot-provider==1.3.2` in `requirements.txt`,
  since that half has no native-dependency/build concerns.
- **Why vendored rather than a deterministic download step**: Streamlit
  Community Cloud has no arbitrary build/postBuild hook (only
  `requirements.txt` for pip and `packages.txt` for apt), so there is no
  place to run `git clone --branch 1.3.2 ...` at deploy time. Vendoring a
  pinned commit in this repository is the deterministic alternative,
  matching this project's existing "pin a release, never track
  master/main" policy for every other third-party dependency.
- **How it is used**: `app/analysis/pot_provider.py` points yt-dlp's
  bgutil script-mode PO-token provider at this directory
  (`server_home=<this>/server`, an absolute path derived from the
  repository location) and lazily runs `deno install
  --allow-scripts=npm:canvas,npm:@swc/core --frozen` inside it, at most
  once per process, only when a plain (non-provider) retrieval attempt was
  declined by YouTube. See `docs/input_sources.md`, "Proof-of-Origin token
  support" for the full architecture and its verified/unverified status.
- **License compliance note (not legal advice)**: this project invokes the
  vendored server code as a separate OS subprocess (`deno run ...`) via
  `subprocess.run`, the same way it already invokes `ffmpeg` (also
  commonly GPL-licensed) -- it is not statically or dynamically linked
  into this project's Python process. The upstream copyright notice and
  full license text are preserved unmodified in this directory. This
  project's academic/non-commercial context and its own top-level license
  should still be reviewed against GPL-3.0's terms (particularly around
  distribution) before any public release that bundles this directory --
  that determination has NOT been made by this change and is left to the
  project owner.
- **Not vendored / not modified**: nothing in this directory has been
  edited from the pinned upstream commit. Any future update must re-pin an
  explicit upstream tag/commit, never track a moving branch.
