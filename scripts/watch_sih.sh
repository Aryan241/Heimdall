#!/usr/bin/env bash
# Poll the SIH portal for our problem statement's submitted-idea count and push a phone
# notification whenever it moves. Fifteen-minute interval: frequent enough to be useful,
# light enough to be a courteous amount of traffic for a government site.
PS_ID="SIH26175"
NTFY="https://ntfy.sh/heimdall-207ac67c9d"
STATE="$1"; INTERVAL="${2:-900}"; DEADLINE=$((SECONDS + ${3:-90000}))
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

count() {
  curl -sL --max-time 60 -A "$UA" "https://www.sih.gov.in/sih2026PS" 2>/dev/null \
    | tr -d '\n' | sed 's/<[^>]*>/ /g' \
    | grep -oE "$PS_ID *[0-9]+/500" | head -1 | grep -oE "[0-9]+/500" | cut -d/ -f1
}

prev=$(cat "$STATE" 2>/dev/null)
while [ $SECONDS -lt $DEADLINE ]; do
  n=$(count)
  if [ -z "$n" ]; then
    echo "$(date '+%H:%M') could not read the count (site down, or the page layout changed)"
  elif [ -z "$prev" ]; then
    prev=$n; echo "$prev" > "$STATE"
    echo "$(date '+%H:%M') baseline $n/500"
  elif [ "$n" != "$prev" ]; then
    d=$((n - prev))
    curl -fsS -m 15 -H "Title: SIH ideas: $n/500" -H "Tags: chart_with_upwards_trend" \
         -H "Priority: default" \
         -d "DepthWizard submissions $prev -> $n (${d:+$d} since last check). $((500-n)) slots left." \
         "$NTFY" >/dev/null 2>&1
    echo "$(date '+%H:%M') $prev -> $n  (notified)"
    prev=$n; echo "$prev" > "$STATE"
  fi
  sleep "$INTERVAL"
done
echo "watcher finished after ${3:-90000}s"
