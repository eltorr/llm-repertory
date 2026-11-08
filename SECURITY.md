# Security

## Reporting a vulnerability

Please report it privately: on GitHub, open the **Security** tab and choose **Report a vulnerability**. Don't open a
public issue for it. You'll get an answer within a few days.

Repertory's own code is in scope: the manager, the model server, setup, and its changes to the bundled engines. A
problem in ExLlamaV3 or glm53-flash-offload themselves is worth reporting to those projects as well.

## How Repertory protects itself

- The API and the dashboard listen on 127.0.0.1 by default. The network port for other devices is off unless you
  turn it on.
- The dashboard needs its private link on every port. Requests from other websites and DNS rebinding are refused.
- The model server answers only the manager, which sends a key with every request.
- Settings, the private link and the backend key are kept in files only your user can read.
- Downloads are pinned to one upload per model and checked against Hugging Face's hashes.
- Every push, and a weekly run, checks for: lint with security rules, shell script mistakes, secrets in the code,
  known CVEs in every pinned package, and CodeQL's data-flow analysis.

More in [the manager guide](docs/manager.md#security).
