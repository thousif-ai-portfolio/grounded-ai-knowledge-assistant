# Deploying and Operating an AI Service

## Containers

Docker packages an application with its dependencies into an image, so it runs
the same way on a laptop and on a server. A Dockerfile lists the steps to build
the image, and docker compose starts several containers together, such as an API
and a model server.

## Continuous integration

A CI pipeline such as GitHub Actions runs linting and tests on every push. A
quality gate is a CI step that fails the build when a metric drops below a
threshold, for example when Recall@5 falls under 0.8.

## Latency percentiles

p50 is the median latency: half of all requests are faster. p95 is the latency
that 95 percent of requests are faster than, and it reveals slow outliers that an
average hides.

## Caching

Caching the embeddings of repeated questions avoids recomputing them. Caching
full answers is riskier, because answers must change when documents change.

## Authentication

JSON Web Tokens (JWT) are signed tokens that prove who a user is. The server signs
each token with a secret key, and the token includes an expiry time, after which
the user must log in again. Passwords must be stored as slow salted hashes such
as Argon2, never as plain text.

## Observability

Logging each request's question, retrieved chunk IDs, latency and token counts
makes it possible to debug wrong answers and to spot regressions.
