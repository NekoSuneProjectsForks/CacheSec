#!/usr/bin/env bash
# systemd/install.sh — Install and enable the CacheSec systemd service.
#
# Run as root (or with sudo):
#   sudo bash systemd/install.sh
#
set -euo pipefail

SERVICE_NAME="cachesec"
SERVICE_FILE="$(dirname "$0")/${SERVICE_NAME}.service"
INSTALL_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

echo "==> Installing ${SERVICE_NAME}.service ..."
cp "$SERVICE_FILE" "$INSTALL_PATH"
chmod 644 "$INSTALL_PATH"

echo "==> Reloading systemd daemon ..."
systemctl daemon-reload

echo "==> Enabling ${SERVICE_NAME} to start on boot ..."
systemctl enable "$SERVICE_NAME"

echo "==> Starting ${SERVICE_NAME} ..."
systemctl start "$SERVICE_NAME"

echo ""
echo "Done. Check status with:"
echo "  systemctl status ${SERVICE_NAME}"
echo "  journalctl -u ${SERVICE_NAME} -f"
