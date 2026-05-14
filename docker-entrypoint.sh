#!/bin/sh
set -e

if [ "$BOOTSTRAP_MODEL" = "true" ]; then
  if [ ! -f "${PRIVATE_KEY_PATH:-./models/private.key}" ] || [ ! -f "${PUBLIC_KEY_PATH:-./models/public.key}" ]; then
    python -m ml.generate_keys
  fi

  if [ ! -f "${MODEL_PATH:-./models/model.pkl}" ] || [ ! -f "${MODEL_PATH:-./models/model.pkl}.sig" ]; then
    python -m ml.train
  fi
fi

exec "$@"
