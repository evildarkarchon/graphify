# Issue tracker: GitHub

Issues and PRDs for this repository live in GitHub Issues at `evildarkarchon/graphify`. Use the `gh` CLI for all operations and pass `--repo evildarkarchon/graphify` explicitly so commands cannot resolve to the `Graphify-Labs/graphify` upstream remote.

## Conventions

- **Create an issue**: write multiline content to a temporary file, then run `gh issue create --repo evildarkarchon/graphify --title "..." --body-file <path>`.
- **Read an issue**: run `gh issue view <number> --repo evildarkarchon/graphify --comments`, fetching labels and filtering comments as needed.
- **List issues**: run `gh issue list --repo evildarkarchon/graphify --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: run `gh issue comment <number> --repo evildarkarchon/graphify --body-file <path>`.
- **Apply or remove labels**: run `gh issue edit <number> --repo evildarkarchon/graphify --add-label "..."` or `--remove-label "..."`.
- **Close an issue**: run `gh issue close <number> --repo evildarkarchon/graphify --comment "..."`.

## Pull requests as a triage surface

**PRs as a request surface: no.**

External pull requests do not run through the issue triage labels and states. Collaborator and maintainer pull requests likewise remain outside the request queue.

GitHub shares one number space across issues and pull requests, so a bare `#42` may be either. Resolve ambiguity with `gh pr view 42 --repo evildarkarchon/graphify`, then fall back to `gh issue view 42 --repo evildarkarchon/graphify`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue in `evildarkarchon/graphify`.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --repo evildarkarchon/graphify --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes, Decisions-so-far, and Fog body. Create it with `gh issue create --repo evildarkarchon/graphify --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue through `gh api`. Where sub-issues are unavailable, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels are `wayfinder:<type>` (`research`, `prototype`, `grilling`, or `task`). Once claimed, assign the ticket to the driving developer.
- **Blocking**: use GitHub's native issue dependencies as the canonical, UI-visible representation. Add an edge with `gh api --method POST repos/evildarkarchon/graphify/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric database ID from `gh api repos/evildarkarchon/graphify/issues/<number> --jq .id`, not the issue number or `node_id`. Where dependencies are unavailable, put `Blocked by: #<number>, #<number>` at the top of the child body.
- **Frontier query**: list the map's open children, drop any with an open blocker or assignee, and take the first ticket in map order.
- **Claim**: run `gh issue edit <number> --repo evildarkarchon/graphify --add-assignee @me`; this is the session's first write.
- **Resolve**: comment with the answer, close the ticket, then append a context pointer and link to the map's Decisions-so-far.
