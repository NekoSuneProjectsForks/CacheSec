# CacheSec — Face Recognition Security System

A production-ready, self-hosted face recognition security system built for
**Raspberry Pi 5**. Continuously monitors a camera, recognises enrolled people,
alerts on unknowns via Discord, records video evidence, and provides a full
admin dashboard accessible locally or remotely through Cloudflare Tunnel.

---

## Features

| Feature | Details |
|---------|---------|
| Face recognition | InsightFace `buffalo_l` (ONNX, ARM64-optimised) |
| Web dashboard | Flask + Bootstrap 5 dark theme |
| Auth | bcrypt passwords, session auth, rate limiting, lockout |
| RBAC | admin / operator / viewer roles |
| Alerts | Discord webhook with snapshot image attachment and optional video upload |
| Recording | Auto-start/stop with min/max duration enforcement; local saving and microphone audio are configurable |
| Sound | GPIO PWM buzzer (access-denied tone on unknown) |
| Database | SQLite with WAL mode |
| Deployment | Gunicorn + systemd + Cloudflare Tunnel |
| Audit log | All admin actions tracked |

---

## Directory Structure

```
cachesec/
├── app.py                  # Flask entry point + bootstrap
├── config.py               # All config from .env
├── database.py             # SQLite connection + schema init
├── models.py               # DB query helpers
├── auth.py                 # Login, RBAC, session management
├── admin.py                # Dashboard blueprint (all routes)
├── camera.py               # Camera capture + detection loop
├── recognition.py          # InsightFace embedding + matching
├── recorder.py             # Video recording state machine
├── discord_notify.py       # Discord webhook integration
├── sound.py                # GPIO buzzer abstraction
├── sounds.py               # Original GPIO tone primitives
├── utils.py                # Shared helpers
├── requirements.txt
├── .env.example            # Copy to .env and fill in
├── templates/
│   ├── base.html
│   ├── auth/login.html
│   ├── admin/              # All dashboard pages
│   └── errors/             # 403, 404, 429, 500
├── static/
│   ├── css/dashboard.css
│   ├── js/dashboard.js
│   └── img/
├── uploads/faces/          # Enrolled face images
├── recordings/             # Saved video clips
├── snapshots/              # Event snapshot JPEGs
├── logs/
└── systemd/
    ├── cachesec.service
    └── install.sh
```

---

## Quick Start

### 1 — Clone / copy the project

```bash
cd /home/cache
# (project already at /home/cache/cachesec)
```

### 2 — Create a virtual environment

```bash
cd /home/cache/cachesec
python3 -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on InsightFace model download:**  
> On first run, InsightFace downloads the `buffalo_l` model weights (~300 MB)
> from a CDN. Ensure your Pi has internet access. Models are cached in
> `~/.insightface/models/`. After that, the system works fully offline.

### 4 — Configure environment

```bash
cp .env.example .env
nano .env
```

**Required changes in `.env`:**

| Key | What to set |
|-----|-------------|
| `SECRET_KEY` | Generate with: `python3 -c "import secrets; print(secrets.token_hex(32))"` |
| `DISCORD_WEBHOOK_URL` | Your Discord channel webhook URL |
| `DISCORD_MENTION_EVERYONE` | Set `true` to include `@everyone` in unknown Discord alerts; defaults to `false` |
| `SAVE_RECORDINGS_LOCALLY` | Set `false` to upload completed clips to Discord and remove local video files |
| `RECORD_AUDIO_ENABLED` | Optional microphone capture for recordings; defaults to `false` |
| `SESSION_COOKIE_SECURE` | `true` if behind Cloudflare HTTPS, `false` for LAN-only HTTP |

### 5 — Initialise and run (dev)

```bash
source venv/bin/activate
python app.py
```

Open `http://localhost:5000` in your browser.  
**Default credentials: `admin` / `changeme123`** — change immediately.

---

## Face Enrollment

1. Log in as `admin`.
2. Navigate to **Enrolled Faces → Add Person**.
3. Enter the person's name and save.
4. On the person detail page, click **Upload & Enroll** and select 3–10 clear
   face photos (good lighting, different angles).
5. CacheSec generates ONNX embeddings and stores them in the database.
6. The person is immediately active in the recognition gallery.

**Tips for good enrollment images:**
- Well-lit, front-facing shots work best.
- Include some slight angle variation (left/right/up/down ~15°).
- Avoid sunglasses or heavy obstructions.
- 5–8 images per person gives good accuracy.

---

## systemd Service

### Install (run as root)

```bash
sudo bash systemd/install.sh
```

### Manual setup

```bash
# Copy service file
sudo cp systemd/cachesec.service /etc/systemd/system/

# Edit User/Group/WorkingDirectory if needed
sudo nano /etc/systemd/system/cachesec.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable cachesec
sudo systemctl start cachesec
```

### Management commands

```bash
# Status
sudo systemctl status cachesec

# View live logs
sudo journalctl -u cachesec -f

# Restart (after config change)
sudo systemctl restart cachesec

# Stop
sudo systemctl stop cachesec

# Disable autostart
sudo systemctl disable cachesec
```

### GPIO group (for buzzer)

The service user needs to be in the `gpio` group:

```bash
sudo usermod -aG gpio cache
sudo usermod -aG audio cache   # required if recording microphone audio
# Log out and back in, or restart the service
```

---

## Cloudflare Tunnel Deployment

Cloudflare Tunnel lets you expose CacheSec's dashboard remotely without
opening any firewall ports. Traffic flows:

```
Browser → Cloudflare Edge → cloudflared (on Pi) → localhost:5000
```

### Step 1 — Install cloudflared

```bash
# Raspberry Pi OS (aarch64)
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared-linux-arm64.deb
```

### Step 2 — Authenticate with your Cloudflare account

```bash
cloudflared tunnel login
```

This opens a browser for OAuth. Authorise and a certificate is saved to
`~/.cloudflared/cert.pem`.

### Step 3 — Create a named tunnel

```bash
cloudflared tunnel create cachesec
```

Note the **Tunnel ID** shown (e.g. `abc123...`).

### Step 4 — Configure the tunnel

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: <YOUR_TUNNEL_ID>
credentials-file: /home/cache/.cloudflared/<YOUR_TUNNEL_ID>.json

ingress:
  - hostname: security.yourdomain.com
    service: http://127.0.0.1:5000
    originRequest:
      noTLSVerify: false
  - service: http_status:404
```

Replace `security.yourdomain.com` with your actual subdomain.

### Step 5 — Create DNS CNAME record

```bash
cloudflared tunnel route dns cachesec security.yourdomain.com
```

This creates a `CNAME` in your Cloudflare DNS pointing to the tunnel.

### Step 6 — Run cloudflared as a service

```bash
sudo cloudflared service install
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
```

### Step 7 — Update .env for HTTPS

```env
SESSION_COOKIE_SECURE=true
PROXY_COUNT=1
ALLOWED_HOSTS=security.yourdomain.com
```

Restart CacheSec:

```bash
sudo systemctl restart cachesec
```

### Cloudflare Access (Zero Trust) — recommended

Add an additional authentication layer in front of your dashboard so it
requires a Cloudflare identity check (Google/GitHub/email OTP) before
the Flask login page is even reached.

1. Go to **Cloudflare Zero Trust → Access → Applications**.
2. Add a **Self-hosted** application.
3. Set the domain to `security.yourdomain.com`.
4. Add a policy: allow only your email address.
5. Save. Cloudflare will now prompt for identity before forwarding to your Pi.

This provides defence-in-depth: Cloudflare Access blocks untrusted visitors
before they can attempt Flask login brute-force.

---

## Security Notes

| Topic | Implementation |
|-------|---------------|
| Password storage | bcrypt via passlib (cost factor 12) |
| Login lockout | 5 failed attempts → 5-minute lockout (DB-backed) |
| Session cookies | HttpOnly, SameSite=Lax, Secure (behind HTTPS) |
| RBAC | admin / operator / viewer enforced per route |
| File uploads | Extension whitelist, size limit, sanitised filenames |
| Path traversal | Filenames sanitised with `os.path.basename()` |
| SQL injection | Parameterised queries throughout |
| Security headers | CSP-compatible headers added on every response |
| Proxy trust | `ProxyFix` with `PROXY_COUNT=1` for Cloudflare |
| Secrets | Never hardcoded; always loaded from `.env` |
| Debug mode | Disabled in production (`FLASK_ENV=production`) |

---

## Face Recognition — Library Choice Rationale

**InsightFace with ONNX Runtime** was chosen over alternatives for these reasons:

| Library | Speed (Pi 5) | Accuracy | Install complexity |
|---------|-------------|----------|--------------------|
| **InsightFace + ONNX** | ~80–150ms/detection | Excellent (buffalo_l ~99.7% LFW) | pip install, weights auto-download |
| face_recognition (dlib) | ~400–800ms | Good | Requires dlib compilation (~30min on Pi) |
| DeepFace | ~300–600ms | Good | Heavy dependencies |
| OpenCV DNN | ~50ms | Fair | Built into OpenCV |

**Tuning options** (all via `.env` or Settings page):

- Swap to `buffalo_s` for ~2× speed at slight accuracy cost: change
  `model_name="buffalo_s"` in `recognition.py:get_recognizer()`
- Reduce `FRAME_WIDTH`/`FRAME_HEIGHT` to `320x240` for faster capture
- Increase `FRAME_SKIP` to `5–10` to further reduce CPU (sacrifices responsiveness)
- Lower `RECOGNITION_THRESHOLD` (e.g. `0.35`) to be stricter about matches

---

## Troubleshooting

**Camera not detected:**
```bash
# Check camera is visible
ls /dev/video*
# Test with OpenCV
python3 -c "import cv2; c=cv2.VideoCapture(0); print(c.isOpened()); c.release()"
```

**InsightFace model download fails:**
```bash
# Manual download
python3 -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=0)"
```

**GPIO buzzer not working:**
```bash
# Check group membership
groups cache
# Add if missing
sudo usermod -aG gpio cache
```

**Dashboard 500 errors:**
```bash
sudo journalctl -u cachesec -n 50 --no-pager
tail -f /home/cache/cachesec/logs/cachesec.log
```

**Slow recognition on Pi:**
- Increase `FRAME_SKIP` in `.env`
- Use `buffalo_s` model instead of `buffalo_l`
- Reduce detection resolution in `recognition.py` (`det_size=(160, 160)`)

---

## Privacy & Consent

This system is designed for use on **your own property** with people who have
**given explicit consent** to be enrolled. Key privacy defaults:

- All data stays on-device (no cloud processing)
- Embeddings are 512-dimensional float vectors; they cannot reconstruct a face
- Snapshots are stored locally; recordings can be stored locally or uploaded to Discord only
- Audit log tracks who enrolled or deleted face data
- Enrolled people can be deleted (removing images and embeddings) at any time

---

## License

MIT — Use responsibly and in accordance with local privacy laws.
