#!/usr/bin/env bash
# Merge NousResearch/hermes-agent main into this fork and open/update a PR.
# Intended for Nag112/hermes-agent scheduled automation — not upstream.
set -euo pipefail

UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/NousResearch/hermes-agent.git}"
UPSTREAM_REF="${UPSTREAM_REF:-main}"
BASE_REF="${BASE_REF:-main}"
SYNC_BRANCH="${SYNC_BRANCH:-chore/sync-nous-main}"
REMOTE="${REMOTE:-origin}"

git config user.name "${GIT_AUTHOR_NAME:-github-actions[bot]}"
git config user.email "${GIT_AUTHOR_EMAIL:-41898282+github-actions[bot]@users.noreply.github.com}"

if ! git remote get-url upstream >/dev/null 2>&1; then
  git remote add upstream "$UPSTREAM_URL"
else
  git remote set-url upstream "$UPSTREAM_URL"
fi

git fetch --prune "$REMOTE" "$BASE_REF"
git fetch --prune upstream "$UPSTREAM_REF"

local_main="$REMOTE/$BASE_REF"
upstream_main="upstream/$UPSTREAM_REF"

if git merge-base --is-ancestor "$upstream_main" "$local_main"; then
  echo "Already up to date: $local_main contains $upstream_main"
  exit 0
fi

git checkout -B "$SYNC_BRANCH" "$local_main"

set +e
git merge -X theirs --no-edit "$upstream_main"
merge_status=$?
set -e

if [[ "$merge_status" -ne 0 ]]; then
  # Remaining conflicts: take upstream for content; honor upstream deletions.
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    # Stage 3 present → they still have the file.
    if git rev-parse --verify --quiet ":3:$path" >/dev/null; then
      git checkout --theirs -- "$path"
      git add -- "$path"
    else
      git rm -f -- "$path" 2>/dev/null || { rm -f -- "$path"; git add -u -- "$path"; }
    fi
  done < <(git diff --name-only --diff-filter=U)
  if [[ -n "$(git diff --name-only --diff-filter=U)" ]]; then
    echo "Unresolved conflicts remain:" >&2
    git diff --name-only --diff-filter=U >&2
    exit 1
  fi
  git commit --no-edit -m "Merge NousResearch/hermes-agent ${UPSTREAM_REF} into this fork."
fi

if git merge-base --is-ancestor HEAD "$local_main"; then
  echo "Nothing to push (sync branch is already in $local_main)"
  exit 0
fi

git fetch "$REMOTE" "$SYNC_BRANCH" || true
git push -u "$REMOTE" "HEAD:refs/heads/$SYNC_BRANCH" --force-with-lease

if ! command -v gh >/dev/null 2>&1; then
  echo "gh not found; branch pushed as $SYNC_BRANCH"
  exit 0
fi

existing="$(gh pr list --base "$BASE_REF" --head "$SYNC_BRANCH" --state open --json number --jq '.[0].number // empty')"
if [[ -n "$existing" ]]; then
  echo "Updated open PR #$existing"
  gh pr view "$existing" --json url --jq .url
  exit 0
fi

upstream_sha="$(git rev-parse --short "$upstream_main")"
gh pr create --base "$BASE_REF" --head "$SYNC_BRANCH" \
  --title "Merge latest NousResearch/hermes-agent ${UPSTREAM_REF}" \
  --body "$(cat <<EOF
## What does this PR do?

Automated sync of [NousResearch/hermes-agent \`${UPSTREAM_REF}\`](https://github.com/NousResearch/hermes-agent) (\`${upstream_sha}\`) into this fork.

Conflicts (if any) take upstream for core files and honor upstream deletions. Fork-only additions such as \`skills/productivity/calendar\` and \`skills/productivity/vikunja\` are kept when they do not conflict.

## Type of Change

- [x] ♻️ Refactor (no behavior change) — upstream sync

Opened by \`.github/workflows/sync-upstream.yml\` (every other day at 10:00 Asia/Kolkata, or manual dispatch).
EOF
)"
