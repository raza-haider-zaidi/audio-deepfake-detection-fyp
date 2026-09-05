# Local URL Ingestion Helper

## Why URL retrieval is separated from cloud inference

Streamlit Community Cloud reliably deploys this application, but its
outbound datacenter IP address is blocked by YouTube's media servers
(`HTTP 403 Forbidden`) even with a Proof-of-Origin token provider
configured (see `docs/input_sources.md`, "Proof-of-Origin token
support", for that earlier attempt and its findings). This is a hosting
constraint, not a bug in this project's extraction logic — the exact same
retrieval code succeeds from an ordinary residential/office internet
connection.

Rather than keep fighting the datacenter IP with proxies or credentials
(explicitly out of scope — see "Public URLs only" below), Video URL
analysis is served by a small, separately-run **local helper** on a
machine where retrieval already works. The deployed Streamlit application
never contacts YouTube directly for this feature; it only talks to a
helper the user has explicitly connected for their own browser session.

## Architecture

```
Streamlit Cloud  --HTTPS-->  Cloudflare Quick Tunnel  --loopback-->  Local Helper  -->  YouTube
                                                                          |
                                                                    local ffmpeg
                                                                          |
                                                              normalized WAV audio
                                                                          |
                                                              <-- returned to Streamlit
```

- **Local helper role**: a small FastAPI service (`local_helper/`) that
  retrieves a public video's audio using the same, already-tested
  extraction logic as before (`app.analysis.video_url` /
  `app.analysis.media_ffmpeg`), extracts only the requested interval, and
  returns normalized 16 kHz mono WAV bytes.
- **Temporary Cloudflare tunnel**: `cloudflared tunnel --url
  http://127.0.0.1:<port>` (a Quick Tunnel) exposes the helper at a
  random `https://<random>.trycloudflare.com` address for the life of the
  helper process only — no account, no domain, no persistent public
  server. See `local_helper/tunnel.py`.
- **Session authentication**: a fresh, cryptographically random access
  token is generated every time the helper starts and is required
  (`Authorization: Bearer <token>`) on every endpoint except a minimal
  `/health` check. See `local_helper/security.py` and "Security" below.
- **No persistent media**: retrieved audio exists only in memory and in a
  temporary file deleted immediately after extraction (see
  `app/analysis/media_ffmpeg.py`, unchanged); nothing is written to a
  database, and the helper keeps no history between requests.
- **Public URLs only**: the helper accepts only `youtube.com`,
  `www.youtube.com`, `m.youtube.com`, `youtu.be`, and
  `youtube.com/shorts` links (the same allow-list
  `app.analysis.video_url` already enforced) — no generic URL fetching,
  no cookies, no Google account, no browser-cookie extraction, no proxy
  service, no third-party download API.
- **Upload fallback**: Audio File and Video File analysis are completely
  unaffected and remain available with no helper connection at all.
- **The analysis computer must stay on**: the helper (and its tunnel)
  only exist while its process is running — closing the helper window
  ends the Video URL feature for that session until it is reconnected.

## Connecting a helper session

1. On the analysis computer, start `Raza Audio URL Helper.exe`. It
   starts the local API, opens a Cloudflare Quick Tunnel, and displays a
   connection code once ready.
2. In the deployed Streamlit application, open the Video URL source and
   paste that connection code into the "Local URL Helper" card, then
   select Connect.
3. Streamlit decodes the code, checks its shape (HTTPS, a
   `trycloudflare.com` hostname, a supported version), and calls the
   helper's `/health` endpoint to confirm it is reachable before treating
   the session as connected.
4. The endpoint and token are kept only in that browser's Streamlit
   session state for the remainder of the session — never written to a
   file, a log, or Streamlit secrets, and never persisted across a
   reload.

If no helper is connected, or the connection drops, the Video URL card
shows: *"Video URL helper is not connected. Start the Audio Deepfake URL
Helper on the analysis computer, then connect it here."* with the
existing Audio File / Video File upload options offered as an
alternative — none of the other input sources are affected by helper
availability.

## Privacy

- **Audio File / Microphone / Voice Note / Video File**: media is
  processed by the Streamlit server for the current session and not
  intentionally persisted.
- **Video URL (local helper)**: the public video is retrieved by the
  connected helper computer; the selected audio interval is then
  temporarily transferred, over the authenticated tunnel connection, to
  the Streamlit application for analysis; the helper deletes its own
  temporary files immediately after each request. This is accurately
  described as data leaving the helper computer for analysis — **not**
  as "audio never leaves your device", which would be false for this
  source (the WAV interval is sent to Streamlit for inference, the same
  as every other source).

## Build/runtime verification status

The standalone Windows build (`scripts/build_local_helper.ps1`) has been
built and run end-to-end on a real Windows machine: the built
`Raza Audio URL Helper.exe`, run with its system PATH stripped down to
just `C:\Windows\System32` (simulating a machine with none of this
project's development tools installed), correctly bound its local API to
`127.0.0.1` only, launched its own bundled `tools\cloudflared.exe` (never
a system copy), reached "Ready", produced a working connection code via
"Copy Connection", and served real metadata/audio requests for a Shorts
URL, a normal watch URL, and a >30-second video with a non-zero start
offset over the actual public `https://*.trycloudflare.com` tunnel URL
(never `127.0.0.1`) -- ending in a real Spectra-AASIST3 prediction.
Closing the window terminated both the helper and `cloudflared.exe`
cleanly (no orphan processes); restarting produced a new tunnel URL and
token, and the old connection code stopped working immediately.

One issue was found and fixed during this verification: the `deno` JS
runtime (used by yt-dlp for some signature-challenge formats) is located
by the `deno` pip package via its installation environment's `Scripts/`
directory, which does not exist inside a frozen PyInstaller build --
`scripts/build_local_helper.ps1` now bundles the real `deno.exe` into
`tools/` alongside `cloudflared.exe`/`ffmpeg.exe`/`ffprobe.exe`, which
yt-dlp finds via its own PATH lookup (verified against yt-dlp's own
`_find_exe` source).

Not verified from a development sandbox: connecting an actual deployed
Streamlit Cloud session to a helper session (requires driving the real
deployed app's browser UI), and a second/incognito browser session
against that same deployed app.

## Development history note

An earlier iteration of this feature ran extraction directly from the
Streamlit Cloud process, including a Proof-of-Origin token provider
fallback for YouTube's anti-bot media-request checks (see
`docs/input_sources.md`). That approach is retained in the codebase for
local development and as the implementation the helper itself reuses, but
is no longer what the deployed application uses in production, since
Streamlit Community Cloud's datacenter egress IP was found to be
unsuitable for reliable YouTube media retrieval even with a token
provider.
