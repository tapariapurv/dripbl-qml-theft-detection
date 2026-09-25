#!/usr/bin/env bash
# Raspberry Pi / Linux fog server setup: venv, dependencies, systemd services.
#   cd dripbl-fog-server && sudo -E bash deploy/install.sh
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="${SUDO_USER:-$(whoami)}"

if [ ! -x "$DIR/.venv/bin/python3" ]; then
  sudo -u "$USER_NAME" python3.11 -m venv "$DIR/.venv"
fi
sudo -u "$USER_NAME" "$DIR/.venv/bin/pip" install -q -r "$DIR/requirements.txt"
# Real GPIO on a Pi (the node falls back to a mock elsewhere)
if grep -qi raspberry /proc/device-tree/model 2>/dev/null; then
  sudo -u "$USER_NAME" "$DIR/.venv/bin/pip" install -q RPi.GPIO
fi

[ -f "$DIR/deploy/dripbl.env" ] || { cp "$DIR/deploy/dripbl.env.example" "$DIR/deploy/dripbl.env"; chmod 600 "$DIR/deploy/dripbl.env"; chown "$USER_NAME" "$DIR/deploy/dripbl.env"; }

for svc in dripbl-node dripbl-api dripbl-dashboard; do
  sed -e "s|@USER@|$USER_NAME|g" -e "s|@DIR@|$DIR|g" "$DIR/deploy/$svc.service" > "/etc/systemd/system/$svc.service"
done
systemctl daemon-reload
systemctl enable --now dripbl-api dripbl-node dripbl-dashboard

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "DRIPBL is running and will start on boot."
echo "  Dashboard: http://${IP:-<pi-ip>}:8501"
echo "  API docs:  http://${IP:-<pi-ip>}:8000/docs"
echo "  Logs:      journalctl -u dripbl-node -f"
echo "Set DRIPBL_DASHBOARD_PASSWORD in deploy/dripbl.env, then: sudo systemctl restart dripbl-dashboard"
