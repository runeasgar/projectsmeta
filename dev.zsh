# For direnv, installed via Brew
eval "$(direnv hook zsh)"

# Docker Compose Up Directory
dcud() {
  if [ -d "$1" ]; then
    (cd "$1" && docker compose up -d)
  else
    echo "dcud: directory not found: $1"
  fi
}

# --- Glances web dashboard helpers ----------------------------------------
_start_glances() {
  if ! command -v glances >/dev/null 2>&1; then
    echo "glances not found (brew install glances)"
    return 1
  fi

  if pgrep -f glances >/dev/null 2>&1; then
    echo "Glances already running"
    return 0
  fi

  # Run in the background, web mode on port 61208
  nohup glances -w >/tmp/glances.log 2>&1 &
  disown
  echo "Glances started on http://localhost:61208"
}

_stop_glances() {
  if pgrep -f glances >/dev/null 2>&1; then
    pkill -f glances
    echo "Glances stopped"
  else
    echo "Glances not running"
  fi
}

_glances_status() {
  if pgrep -f glances >/dev/null 2>&1; then
    echo "Glances: running (http://localhost:61208)"
  else
    echo "Glances: stopped"
  fi
}
