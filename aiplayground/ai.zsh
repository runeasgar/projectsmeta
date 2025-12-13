# Quick overview of active development stuff
devstatusai() {
  echo; echo "== Orbstack status =="; command -v orb >/dev/null && orb status
  echo; echo "== Docker containers =="; docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
  echo; echo "== Brew services =="; brew services list
  echo; echo "== Ollama activity =="; command -v ollama >/dev/null && ollama ps
  echo; echo "== Mflux ==";
  if lsof -i :8000 >/dev/null 2>&1; then
    echo "mflux service: RUNNING (port 8000)"
  else
    echo "mflux service: STOPPED"
  fi
  echo; echo "== Glances =="; _glances_status
  echo;
}

# Spins up all common development stuff
devupai() {
  echo "Starting up dev infrastructure..."
  echo "Starting Ollama..."
  brew services start ollama
  echo "Starting Orbstack..."
  orb start
  echo "Starting Open-WebUI..."
  dcud "$HOME/projects/aiplayground/open-webui"
  echo "Starting n8n v1..."
  dcud "$HOME/projects/aiplayground/n8n-v1"
  echo "Starting glances..."
  _start_glances
  echo "Running devstatusai..."
  devstatusai
}

# Spins down all common development stuff
devdownai() {
  echo "Stopping Orbstack..."
  orb stop
  echo "Stopping Ollama..."
  brew services stop ollama
  echo "Deleting Ollama REPL history..."
  rm "$HOME/.ollama/history"
  echo "Stopping glances..."
  _stop_glances
  echo "Running devstatusai..."
  devstatusai
}

# All remaining code is Ollama convenience behaviors soruced from ChatGPT 
# --- robust one-shot helper (safe JSON), optional speech ---
_ollama_query() {
  local model="$1"; shift
  local speak=false
  if [[ "$1" == "-s" ]]; then
    speak=true
    shift
  fi
  local prompt="$*"

  # Build JSON safely with jq (handles quotes/newlines)
  local payload
  payload=$(jq -n --arg m "$model" --arg p "$prompt\nRespond only in plain text. Do not use Markdown or code formatting." '{model:$m, prompt:$p, stream:false}')

  # Call API
  local raw
  raw=$(curl -s http://localhost:11434/api/generate \
           -H "Content-Type: application/json" \
           --data-binary "$payload")

  # Prefer .response; fall back to .error
  local text
  text=$(jq -r '.response // .error // empty' <<<"$raw")

  if [[ -z "$text" ]]; then
    echo "[ollama] unexpected response:"
    echo "$raw"
    return 1
  fi

  printf "%s\n" "$text"
  $speak && say "$text"
}

# --- single dispatcher; first arg is the model name injected by alias ---
_ollama_model_cmd() {
  local model="$1"; shift
  if [[ $# -eq 0 ]]; then
    ollama run "$model"          # REPL/chat
  else
    _ollama_query "$model" "$@"  # one-shot (add -s to speak)
  fi
}

# --- declare the names you approve as commands ---
# edit this one line as you add/remove local tags:
typeset -a OLLAMA_MODELS=(llama hermes deepseek stablecode cogito gptoss qwen gemma phi mistral nomic)

# Clean up any conflicting aliases from earlier attempts
for m in "${OLLAMA_MODELS[@]}"; do
  alias "$m" &>/dev/null && unalias "$m"
done

# Create lightweight aliases that inject the model name
for m in "${OLLAMA_MODELS[@]}"; do
  alias "$m"="_ollama_model_cmd $m"
done
