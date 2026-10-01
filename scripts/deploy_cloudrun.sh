#!/usr/bin/env bash
# Deploy the PhotoFix demo to Google Cloud Run. The image is built in Google Cloud from the Dockerfile,
# so Docker isn't needed locally.
#
# One-time setup:
#   gcloud auth login
#   gcloud config set project <PROJECT_ID>        # a project with billing enabled
#
#   ./scripts/deploy_cloudrun.sh
#
# Cost controls: scales to zero when idle, at most 2 instances, us-central1 (free-tier region).
# Typical portfolio traffic stays within Cloud Run's free tier; image storage costs a few cents a month.
set -euo pipefail
cd "$(dirname "$0")/.."

SERVICE="${SERVICE:-photofix}"
REGION="${REGION:-us-central1}"

PROJECT="$(gcloud config get-value project 2>/dev/null)"
if [[ -z "$PROJECT" ]]; then
  echo "No project set. Run: gcloud config set project <PROJECT_ID>" >&2
  exit 1
fi
echo "Deploying $SERVICE to project $PROJECT ($REGION)"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --cpu 2 --memory 2Gi \
  --concurrency 2 \
  --min-instances 0 --max-instances 2 \
  --timeout 120 \
  --cpu-boost \
  --quiet

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')"
echo
echo "Live: $URL"
curl -fsS "$URL/api/health" && echo
