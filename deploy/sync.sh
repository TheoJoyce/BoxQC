#!/usr/bin/env bash
# Sync the project code between a dev machine (Linux PC / MacBook) and the Pi with rsync.
#
#   ./deploy/sync.sh push        dev -> Pi (dev is the source of truth; deletes stale
#                                files on the Pi), then restarts the boxqc service
#   ./deploy/sync.sh pull        Pi -> dev (fetch edits made on the Pi; never deletes)
#   ./deploy/sync.sh status      show what push and pull would change, change nothing
#   ./deploy/sync.sh ssh-key     install your SSH key on the Pi (no more passwords)
#   ./deploy/sync.sh fetch-data  one-off, when upgrading from BoxQC 0.1: copy the image
#                                dataset that 0.1 kept on the Pi to this machine
#                                (nothing is deleted on either side)
#
# Options:  -y  skip the confirmation      --no-restart  don't restart after push
#           --host USER@HOST               --dir PATH_ON_PI
#           --name NAME   with USER@auto: which device, when several answer
#
# Dev side: always this project folder (found from the script's location; all local
# paths are relative to it). Pi side: the SSH login you name, and by default a folder in
# that user's home named like this one. Set both once per machine in .sync.env at the
# project root (not synced, not committed), e.g.
#   PI_HOST=pi@auto        # "auto": find the Pi on the network, as `boxqc devices` does
#   PI_NAME=piqc           # optional: which one, if several answer
#   PI_DIR=boxqc
#
# Never touched: .venv, data/, models/, work/ ... (see deploy/rsync-exclude.txt), nor
# the folders that data_dir, models_dir and train.workdir name in the configuration of
# either machine, if you renamed them.
# Works with macOS's bash 3.2 and its built-in rsync/openrsync.
set -euo pipefail

# Work from the project root so every local path below is relative ("./"): the
# script behaves the same wherever the project lives and whatever the cwd is.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EXCLUDES="deploy/rsync-exclude.txt"

PI_HOST=""                        # USER@HOST of the Pi (HOST may be "auto"): from .sync.env or --host
PI_NAME=""                        # with USER@auto: the device's name, if several answer
PI_DIR="$(basename "$PWD")"       # relative to the Pi user's home
# shellcheck disable=SC1091
[ -f .sync.env ] && . ./.sync.env

CMD="${1:-}"
[ $# -gt 0 ] && shift
ASSUME_YES=0
RESTART=1
while [ $# -gt 0 ]; do
    case "$1" in
        -y|--yes) ASSUME_YES=1 ;;
        --no-restart) RESTART=0 ;;
        --host) PI_HOST="$2"; shift ;;
        --name) PI_NAME="$2"; shift ;;
        --dir) PI_DIR="$2"; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

case "$CMD" in
    push|pull|status|ssh-key|fetch-data)
        if [ -z "$PI_HOST" ]; then
            echo "Which Pi? Pass --host USER@HOST, or set it once in .sync.env at the project root:" >&2
            echo "    PI_HOST=pi@auto       (auto: find the Pi on the network)" >&2
            exit 2
        fi ;;
esac

SSH_EXTRA=""
# USER@auto: ask the network, like `boxqc devices` (src/boxqc/discovery.py, standard
# library only, so no virtualenv is needed). The Pi must be running `boxqc device`.
case "$CMD" in push|pull|status|ssh-key|fetch-data)
if [ "${PI_HOST#*@}" = "auto" ] && [ "$PI_HOST" != "auto" ]; then
    PY=python3
    if .venv/bin/python -c 'import sys' >/dev/null 2>&1; then PY=.venv/bin/python; fi
    found="$("$PY" - "$PI_NAME" <<'EOF'
import sys
if sys.version_info < (3, 10):
    sys.exit("PI_HOST=...@auto needs Python 3.10 or newer; set the Pi's address instead.")
sys.path.insert(0, "src")
from boxqc.discovery import DiscoveryError, choose, discover
try:
    device = choose(discover(), sys.argv[1] or None)
except DiscoveryError as exc:
    sys.exit(f"{exc}\n(For this script: set PI_NAME in .sync.env or pass --name, or use --host USER@ADDRESS.)")
print(device.address, device.hostname)
EOF
)" || exit 1
    PI_HOST="${PI_HOST%@*}@${found% *}"
    # Keep the Pi's SSH host key under its name, not under an address that changes: the
    # host-key check then also stops another machine from answering in the Pi's place.
    SSH_EXTRA="-o HostKeyAlias=${found#* }.local"
    echo "==> found the Pi: ${found#* } at ${found% *}"
fi ;;
esac

# One SSH connection shared by the preview, the transfer and the restart
# (so a password is asked at most once).
# %C = short hash of the connection, keeps the socket path under the length limit.
mkdir -p "$HOME/.ssh"
SSH="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/boxqc-sync-%C -o ControlPersist=60 -o ConnectTimeout=5 $SSH_EXTRA"

# GNU rsync can itemize changes; macOS openrsync / rsync 2.x only do -v, which also
# prints a "Transfer starting" line and every directory it looks at ("sub/"). Those are
# not changes, and must not make an up-to-date tree look like it has some.
if rsync --version 2>/dev/null | grep -q "openrsync\|version 2\."; then
    LIST_FLAG="-v"
    LIST_NOISE='^Transfer starting|/$'
else
    LIST_FLAG="--itemize-changes"
    LIST_NOISE='^\.d'
fi
# Arrays keep paths with spaces intact (bash 3.2 compatible).
RSYNC=(rsync -az "--exclude-from=$EXCLUDES" -e "$SSH")
LOCAL="./"

# Value of the first "key: value" line in a YAML text, without quotes or a trailing comment.
yaml_value() {      # $1 = key, $2 = YAML text
    printf '%s\n' "$2" | sed -n -E "s/^[[:space:]]*$1:[[:space:]]*//p" | head -n 1 \
        | sed -E "s/[[:space:]]+#.*$//; s/[[:space:]]+$//; s/^[\"']//; s/[\"']$//"
}

# data_dir, models_dir and train.workdir can be renamed in the configuration. Whatever
# they are called, on this machine or on the Pi, they hold that machine's own data:
# exclude them too, or `push --delete` would erase the Pi's settings and inspection log.
exclude_configured_dirs() {      # $1 = the text of a config file (may be empty)
    local key value
    for key in data_dir models_dir workdir; do
        value="$(yaml_value "$key" "$1")"
        case "$value" in
            ""|/*|"~"*|..*) ;;      # not set, or outside the project folder: rsync never sees it
            *) value="${value#./}"; RSYNC+=("--exclude=/${value%/}/") ;;
        esac
    done
}
for cfg_file in boxqc.yaml boxqc.yml boxqc.local.yaml boxqc.local.yml; do
    [ -f "$cfg_file" ] && exclude_configured_dirs "$(cat "$cfg_file")"
done
# The Pi's own configuration; called once the connection is known to work.
exclude_pi_configured_dirs() {
    local cfg_file
    for cfg_file in boxqc.yaml boxqc.yml boxqc.local.yaml boxqc.local.yml; do
        exclude_configured_dirs "$($SSH "$PI_HOST" "cat '$PI_DIR/$cfg_file' 2>/dev/null" || true)"
    done
}

remote="$PI_HOST:$PI_DIR/"

confirm() {
    [ "$ASSUME_YES" -eq 1 ] && return 0
    printf "%s [y/N] " "$1"
    read -r answer
    case "$answer" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

# Print the rsync dry-run without its header/footer noise.
# Returns 0 = changes listed, 1 = nothing to do, 2 = rsync failed.
preview() {
    local out
    out="$("${RSYNC[@]}" "$LIST_FLAG" -n "$@")" || return 2
    printf '%s\n' "$out" \
        | grep -v -e '^sending incremental' -e '^receiving' -e '^sent ' -e '^total size' \
                  -e '^building file list' -e '^$' -e '^\./$' \
        | grep -E -v "$LIST_NOISE" \
        | grep . || return 1
}

# Run preview into $CHANGES. Returns 0 if there is something to sync, 1 if not;
# aborts the script if rsync failed. (Called outside $(...) so `exit` works.)
CHANGES=""
gather() {
    local rc=0
    CHANGES="$(preview "$@")" || rc=$?
    if [ "$rc" -eq 2 ]; then
        echo "rsync failed (see above)" >&2
        exit 1
    fi
    return "$rc"
}

check_connection() {
    if ! $SSH "$PI_HOST" true 2>/dev/null; then
        echo "Cannot reach $PI_HOST over SSH." >&2
        echo "  - Is the Pi plugged in / on the network?  ping ${PI_HOST#*@}" >&2
        echo "  - Try another address: --host USER@HOST  (or set PI_HOST in .sync.env)" >&2
        echo "    ('boxqc devices' prints the Pi's address while it runs boxqc device)" >&2
        exit 1
    fi
    exclude_pi_configured_dirs
}

do_push() {
    check_connection
    $SSH "$PI_HOST" "mkdir -p '$PI_DIR'"
    echo "==> push: $LOCAL  ->  $remote"
    if ! gather --delete "$LOCAL" "$remote"; then
        echo "Already up to date."
        return 0
    fi
    local changes="$CHANGES"
    echo "$changes"
    if echo "$changes" | grep -q "deleting"; then
        echo "(lines with 'deleting' are files on the Pi that don't exist here)"
    fi
    confirm "Push these changes to the Pi?" || { echo "Aborted."; exit 1; }
    "${RSYNC[@]}" --delete "$LOCAL" "$remote"
    echo "Pushed."

    if echo "$changes" | grep -q "pyproject.toml"; then
        echo "NOTE: pyproject.toml changed. If dependencies changed, re-run on the Pi:"
        echo "      ./deploy/pi/setup_pi.sh --no-service"
    fi
    if [ "$RESTART" -eq 1 ]; then
        echo "==> restarting boxqc on the Pi (sudo may ask for the Pi password)"
        if $SSH -t "$PI_HOST" "systemctl list-unit-files boxqc.service >/dev/null 2>&1 \
                && sudo systemctl restart boxqc && sleep 2 && systemctl is-active boxqc"; then
            echo "boxqc restarted."
        else
            echo "Restart failed or service not installed - check: journalctl -u boxqc -n 50" >&2
        fi
    fi
}

do_pull() {
    check_connection
    echo "==> pull: $remote  ->  $LOCAL   (nothing is deleted locally)"
    if ! gather "$remote" "$LOCAL"; then
        echo "Already up to date."
        return 0
    fi
    echo "$CHANGES"
    confirm "Overwrite local files with the Pi's versions?" || { echo "Aborted."; exit 1; }
    "${RSYNC[@]}" "$remote" "$LOCAL"
    echo "Pulled."
}

do_status() {
    check_connection
    echo "==> push would change (dev -> Pi):"
    if gather --delete "$LOCAL" "$remote"; then echo "$CHANGES"; else echo "  nothing"; fi
    echo
    echo "==> pull would change (Pi -> dev):"
    if gather "$remote" "$LOCAL"; then echo "$CHANGES"; else echo "  nothing"; fi
}

# BoxQC 0.1 stored the training images on the Pi. They now live with the studio, so
# bring them over once. Only the dataset is copied (not the Pi's settings, inspection
# log or models, which stay the device's own), and nothing is deleted anywhere.
do_fetch_data() {
    check_connection
    if ! $SSH "$PI_HOST" "test -d '$PI_DIR/data/annotations'"; then
        echo "No dataset on the Pi ($PI_HOST:$PI_DIR/data/annotations does not exist) - nothing to fetch."
        return 0
    fi
    # Where the studio on this machine keeps its dataset: data_dir from the local config
    # (the overlay wins), "data" unless it was changed.
    local dest="" cfg_file value
    for cfg_file in boxqc.local.yaml boxqc.local.yml boxqc.yaml boxqc.yml; do
        [ -f "$cfg_file" ] || continue
        value="$(yaml_value data_dir "$(cat "$cfg_file")")"
        if [ -n "$value" ]; then dest="$value"; break; fi
    done
    dest="${dest:-data}"
    dest="${dest%/}"
    case "$dest" in "~/"*) dest="$HOME/${dest#"~/"}" ;; esac
    echo "==> fetch-data: $PI_HOST:$PI_DIR/data/{images,annotations,datasets.json}  ->  $dest/"
    echo "    Files that exist on both sides are replaced by the Pi's version if it differs."
    confirm "Copy the dataset from the Pi to this machine?" || { echo "Aborted."; exit 1; }
    mkdir -p "$dest/images" "$dest/annotations"
    rsync -az -e "$SSH" "$PI_HOST:$PI_DIR/data/images/" "$dest/images/"
    rsync -az -e "$SSH" "$PI_HOST:$PI_DIR/data/annotations/" "$dest/annotations/"
    # datasets.json may not exist; the studio rebuilds the list from the annotations then
    if $SSH "$PI_HOST" "test -f '$PI_DIR/data/datasets.json'"; then
        rsync -az -e "$SSH" "$PI_HOST:$PI_DIR/data/datasets.json" "$dest/"
    fi
    echo "Fetched: $(ls "$dest/images" | wc -l | tr -d ' ') images, $(ls "$dest/annotations" | wc -l | tr -d ' ') annotations in $dest/."
    echo "Check them in the studio (boxqc studio -> Dataset). The copies on the Pi are no longer"
    echo "used; once you are satisfied you can remove them there:"
    echo "    ssh $PI_HOST \"rm -r '$PI_DIR/data/images' '$PI_DIR/data/annotations' '$PI_DIR/data/thumbs'\""
}

do_ssh_key() {
    if [ ! -f "$HOME/.ssh/id_ed25519.pub" ] && [ ! -f "$HOME/.ssh/id_rsa.pub" ]; then
        echo "==> no SSH key found, creating ~/.ssh/id_ed25519"
        ssh-keygen -t ed25519 -f "$HOME/.ssh/id_ed25519" -N ""
    fi
    # shellcheck disable=SC2086  # SSH_EXTRA is empty or one "-o KEY=VALUE" word
    ssh-copy-id $SSH_EXTRA "$PI_HOST"
}

case "$CMD" in
    push) do_push ;;
    pull) do_pull ;;
    status) do_status ;;
    ssh-key) do_ssh_key ;;
    fetch-data) do_fetch_data ;;
    *) sed -n '2,26p' deploy/sync.sh | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
