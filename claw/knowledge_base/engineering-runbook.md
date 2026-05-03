# Engineering Runbook

Operational reference for the engineering team. Keep it short; deep
dives live in Notion.

## Deployment process

All production deployments go through the `deploy` GitHub Action.
Steps:

1. Open a pull request against `main`. CI runs `pytest` and `ruff` on
   every push. Both must pass before review.
2. Get one approving review from a code owner. Code owners are
   declared in `CODEOWNERS`.
3. Merge to `main`. The `deploy` workflow builds a Docker image, pushes
   it to ECR with the commit SHA as the tag, and applies the
   Kubernetes manifests in `deploy/prod/`.
4. The deployment is gated by the readiness probe defined on each
   service. A failed readiness probe rolls the deployment back
   automatically.

## On-call rotation

Engineering on-call follows a weekly rotation, Wednesday to Wednesday,
managed in PagerDuty. The on-call engineer is the first responder for
any P1 / P2 incident in production. Hand-off happens at 10:00 local
time on Wednesday morning with a short sync over Zoom and a written
summary in `#oncall-handoff`.

## Severity definitions

- **P1**: production is down for more than 5% of users, or a security
  incident is in progress. Page on-call immediately.
- **P2**: a major feature is degraded but a workaround exists. Slack
  the on-call engineer and open an incident ticket.
- **P3**: minor issue, no user impact. File a Linear ticket.

## Q3 engineering commitments

For Q3 the engineering team committed to:

- Migrating the search index from Elasticsearch 7 to OpenSearch 2.13.
- Reducing p99 API latency on `/v2/recommendations` from 1.4s to
  under 600ms.
- Shipping the new ingestion pipeline that replaces the bash-cron
  pipeline with a managed Argo workflow.
- Reaching 80% unit-test coverage in the `payments` service.

## Incident postmortems

Every P1 and P2 incident gets a written postmortem within five
business days. Use the template in
`engineering/templates/postmortem.md`. Postmortems are blameless and
focus on causes and corrective actions rather than individual mistakes.
