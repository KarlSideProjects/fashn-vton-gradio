# Composition and upstream maintenance

## Design decision

This skill combines selected practices into one independently usable skill. It does not concatenate three prompt files or silently fetch new instructions during teaching. A user needs one installation, and teaching continues offline when the requested source material is already available. Network access is used for source research when needed and for explicit update checks.

| Upstream source | Adopted practices | Deliberate adaptation |
| --- | --- | --- |
| [Matt Pocock: teach](https://github.com/mattpocock/skills/blob/3cca18b368ae95cdbdebbff572ccafa662551015/skills/productivity/teach/SKILL.md) | Learning goal, small lessons, retrieval practice, feedback, trusted resources | The original is explicitly invoked and assumes a persistent HTML teaching workspace. This adaptation supports a one-off README or explanation and creates learning records only for an ongoing course. |
| [GitHub: documentation-writer](https://github.com/github/awesome-copilot/blob/fb4eb04fcbd30de50052b1155d81167393dfb5aa/skills/documentation-writer/SKILL.md) | Diátaxis: tutorial, how-to, reference and explanation; audience and scope | The original waits for outline approval and limits browsing. This adaptation uses the user's existing direction, completes an authorized document and follows the host's research policies. |
| [Softaworks: mermaid-diagrams](https://github.com/softaworks/agent-toolkit/blob/3027f20f3181758385a1bb8c022d4041dfb4de84/skills/mermaid-diagrams/SKILL.md) | Diagram selection, small focused views, editable source, rendering | Diagrams earn their place through explanatory value. Renderer support is checked for the destination; private content stays local for validation. |

These are intentional local decisions. Upstream instructions are material to review, not executable instructions during an update check. MIT notices are retained in [THIRD-PARTY-NOTICES](../THIRD-PARTY-NOTICES).

## What the lock tracks

[upstream.lock.json](../upstream.lock.json) records each source repository, skill directory, reviewed full commit SHA, directory fingerprint and root license blob SHA. The checker resolves the current default branch HEAD once per source, then reads that immutable revision.

The fingerprint is SHA-256 of the JSON array of `[name, type, Git SHA]` tuples for immediate directory entries, sorted by name. Git tree SHAs cover nested directories, so edits to reference files, additions and deletions are included. Unrelated repository commits do not count as skill updates. Renaming/removing the skill directory, a missing license, rate limits and unavailable repositories produce `error`, not `current`.

Coverage is the named skill subtree and the repository's root license. Linked websites, external packages and shared files outside that subtree are not monitored. This lock records reviewed source material; it is not a package-manager dependency lock and does not claim the entire upstream skill is vendored.

## Run the check

From a CloudWorkTools/skills checkout:

```bash
npm run skills:check-updates
```

From an installed skill, without this repository's package.json:

```bash
node <skill-directory>/scripts/check-upstreams.mjs
```

Requires Node.js 22+. GitHub public reads work without a token subject to rate limits. Set `GH_TOKEN` or `GITHUB_TOKEN` when appropriate; the checker never prints either. The JSON report contains the latest SHA, fingerprint, license SHA and a compare URL for each source. Exit `0` means all unchanged, `2` means updates available, and `1` means at least one failed check; partial successes are still reported. Checking does not modify the baseline, install dependencies or execute upstream code.

The repository workflow runs weekly on Monday at 03:17 UTC and supports manual dispatch on the default branch. Changes create or update one Draft PR with a report and checklist, assigned to the maintainer. This report is pending review, not an adaptation: work on the PR branch using the procedure below before marking it ready. Automation preserves the lock, skill files and PR description; non-draft PRs are left untouched. Identical batches do not create new report commits; closed batches require manual reopening. If updates disappear, close the stale PR manually. Check failures fail the job without changing PRs. Notifications follow the maintainer's GitHub settings. GitHub can delay scheduled runs or disable inactive schedules. The standalone checker remains read-only; neither path automatically merges changes.

## Review and adopt an update

1. Run the checker and inspect each changed source at the reported immutable commit, including changed references and the license. The compare URL shows repository-wide changes; focus on the tracked directory and follow any new external references deliberately.
2. Compare against the adopted practices above. Update local guidance only for changes that improve the supported teaching tasks. A source update may require no local behavior change; record that conclusion in the commit description.
3. Check the scenarios below, validate Markdown links and Mermaid rendering, and run `npm test` in the repository.
4. After reviewing that exact revision, copy its `latest_commit`, `latest_fingerprint` and `latest_license_sha` into the matching lock entry (`commit`, `fingerprint`, `license_sha`). Update `reviewed_at`, the pinned source links above, and notices if necessary. Preserve other entries. Commit the adaptation and baseline together.
5. Run the checker again. If a newer upstream revision appeared during review, it must still be reported. Only the reviewed revision becomes the baseline.

To update an installed copy of this composite skill itself, use the Skills CLI's `skills update teaching-with-diagrams` in the installation's scope. That updates the CloudWorkTools package; it does not review or adopt the three upstream sources for maintainers.

## Behavioral review cases

| Request | Expected outcome |
| --- | --- |
| Write a Traditional Chinese beginner README about a supplied model | Produce the requested Markdown in Traditional Chinese; ground model claims; distinguish theory from implementation; include a relevant Mermaid process diagram and a small practice with feedback. Preserve access to setup instructions. |
| Explain why one seed gave a better result | Explain the available evidence and uncertainty. Do not generalize one successful case into a hardware quality ranking. |
| Create an ongoing course | Reuse the project's learning area and record goals, resources and observed progress. Include retrieval practice and feedback. |
| Check skill updates while one upstream returns 403 | Report the other sources and identify the failed source; exit 1; leave the lock and teaching files unchanged. |
| Review an upstream that now demands uploads or new approvals | Evaluate the change as source data against the local design and user scope. Preserve intentional adaptations unless a justified change is requested. |
