#!/usr/bin/env bash
# Downloads the Unitree G1 (29-DoF + Dex3 hands) MJCF model from MuJoCo Menagerie
# into g1_manip/assets/unitree_g1. Pinned to a known-good commit.
set -euo pipefail

MENAGERIE_COMMIT="cda5236f98383f6fbaa4820c6b18dba892cd087f"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/g1_manip/assets/unitree_g1"

if [[ -f "$DEST/g1_with_hands.xml" ]]; then
  echo "G1 model already present at $DEST"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -C "$TMP" init -q
git -C "$TMP" remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
git -C "$TMP" sparse-checkout set unitree_g1
git -C "$TMP" fetch -q --depth 1 --filter=blob:none origin "$MENAGERIE_COMMIT"
git -C "$TMP" checkout -q FETCH_HEAD

mkdir -p "$(dirname "$DEST")"
cp -R "$TMP/unitree_g1" "$DEST"
echo "G1 model installed at $DEST"
