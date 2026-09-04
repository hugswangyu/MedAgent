#!/bin/sh
set -eu

load_secret() {
  variable="$1"
  path="$2"
  if [ -f "$path" ]; then
    value="$(cat "$path")"
    export "$variable=$value"
  fi
}

load_secret JWT_SECRET_KEY /run/secrets/jwt_secret
load_secret PG_PASSWORD /run/secrets/postgres_password
load_secret MEDAGENT_INTERNAL_API_KEY /run/secrets/internal_api_key
load_secret MEDAGENT_CONTROL_PLANE_KEY /run/secrets/control_plane_key
load_secret LIVEKIT_API_KEY /run/secrets/livekit_api_key
load_secret LIVEKIT_API_SECRET /run/secrets/livekit_api_secret
load_secret LIGHTRAG_API_KEY /run/secrets/lightrag_api_key
load_secret ELASTIC_PASSWORD /run/secrets/elasticsearch_password

if [ -n "${ELASTIC_PASSWORD:-}" ] && [ -z "${ES_HOSTS:-}" ]; then
  export ES_HOSTS="http://elastic:${ELASTIC_PASSWORD}@elasticsearch:9200"
fi

exec "$@"
