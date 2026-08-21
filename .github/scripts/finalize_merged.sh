#!/usr/bin/env bash
# finalize_merged.sh — auto-finalize tasks when a PR is merged.
#
# Usage: finalize_merged.sh <pr_body> <tasks_file> <issues_file>
#
# Reads the PR body, extracts every closed issue number via "Closes/Fixes/Resolves #N"
# keywords (word-boundary match so end-of-line refs are caught — Bug 1 fix), then
# marks the matching task row in tasks.md as [x].
#
# Bug 1 fix: use `#N\b` (word boundary) instead of `#N[^0-9]` so that an issue
#   reference at the very end of a line (e.g. "Closes #23\n") is also detected.
#
# Bug 2 fix: [bug] tasks appear in issues.md as "Bug N" rows, not "Task N" rows.
#   The script looks up the task number that corresponds to a Bug row by reading
#   the issues file and mapping "Bug N -> task number" via the task description.

set -euo pipefail

PR_BODY="${1:-}"
TASKS_FILE="${2:-TradingBot/plans/trading-bot/tasks.md}"
ISSUES_FILE="${3:-TradingBot/plans/trading-bot/issues.md}"

if [[ -z "$PR_BODY" ]]; then
  echo "Usage: $0 <pr_body> [tasks_file] [issues_file]" >&2
  exit 1
fi

# Extract all closed issue numbers from the PR body.
# Matches: Closes #N, Fixes #N, Resolves #N (case-insensitive), word boundary on N.
mapfile -t ISSUE_NUMS < <(echo "$PR_BODY" | grep -ioP '(?:closes|fixes|resolves)\s+#\K[0-9]+\b')

if [[ ${#ISSUE_NUMS[@]} -eq 0 ]]; then
  echo "No closed issue references found in PR body." >&2
  exit 0
fi

for ISSUE_NUM in "${ISSUE_NUMS[@]}"; do
  echo "Processing closed issue #${ISSUE_NUM} ..."

  # Look up this issue in issues.md to find the task description.
  # issues.md rows look like:
  #   | Task N  | #NN | open/closed | Description |
  #   | Bug N   | #NN | open/closed | Description |
  ISSUE_ROW=$(grep -P "^\|\s*(Task|Bug)\s+\d+\s*\|\s*#${ISSUE_NUM}\b" "$ISSUES_FILE" || true)

  if [[ -z "$ISSUE_ROW" ]]; then
    echo "  Issue #${ISSUE_NUM} not found in ${ISSUES_FILE}, skipping." >&2
    continue
  fi

  # Extract the description (4th pipe-delimited column).
  DESCRIPTION=$(echo "$ISSUE_ROW" | awk -F'|' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5); print $5}')

  if [[ -z "$DESCRIPTION" ]]; then
    echo "  Could not extract description for issue #${ISSUE_NUM}, skipping." >&2
    continue
  fi

  # Find and update the matching task row in tasks.md.
  if grep -qF "$DESCRIPTION" "$TASKS_FILE"; then
    # Replace [ ] or [~] with [x] on the line containing the description.
    python3 - "$TASKS_FILE" "$DESCRIPTION" <<'PYEOF'
import sys, re
tasks_file, description = sys.argv[1], sys.argv[2]
with open(tasks_file) as f:
    lines = f.readlines()
updated = []
for line in lines:
    if description in line:
        line = re.sub(r'^(\s*-\s*)\[[ ~]\]', r'\1[x]', line)
    updated.append(line)
with open(tasks_file, 'w') as f:
    f.writelines(updated)
PYEOF
    echo "  Marked task as [x] for issue #${ISSUE_NUM}: ${DESCRIPTION}"
  else
    echo "  Task not found in ${TASKS_FILE} for issue #${ISSUE_NUM}: ${DESCRIPTION}" >&2
  fi
done
