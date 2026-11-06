# Deploying Steelbound

Steelbound is a local-first AI core with zero runtime dependencies
(Python ≥ 3.12). There are three supported ways to run it; all three
run the same packaged code and differ only in who supervises the
process. Configuration templates live in `../config/` — note that
Steelbound is configured by CLI flags (`--data-dir`, `--provider`,
`--auto-local`), not by a config file or environment variables; see
`config/settings.env.example` for the full explanation.

## 1. pip (simplest; run it yourself)

```bash
pip install .            # from a checkout, or a release archive
steelbound --data-dir /path/to/data evals   # release-gate check
steelbound --data-dir /path/to/data serve --host 127.0.0.1 --port 8765
```

The API and GUI shell are then at `http://127.0.0.1:8765/`.

## 2. Docker (Compose)

```bash
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml logs -f
docker compose -f deploy/docker-compose.yml down   # data survives (named volume)
```

`deploy/Dockerfile` installs the package into a `python:3.12-slim`
image and runs it as a non-root user with the data dir at `/data`.
`deploy/docker-compose.yml` builds from the repository root, stores
`/data` on the named volume `steelbound-data`, restarts the service
`unless-stopped`, and publishes the port as `127.0.0.1:8765:8765` —
loopback on the host side, deliberately.

## 3. systemd (bare metal / VM)

```bash
sudo ./deploy/install.sh        # venv in /opt/steelbound, data in /var/lib/steelbound
sudo systemctl enable --now steelbound
```

`install.sh` checks for root before touching `/opt`, `/var/lib`, or
`/etc`, creates the dedicated `steelbound` user and the data dir,
installs the package into a venv, and installs
`deploy/steelbound.service` when systemd is present (it skips
gracefully otherwise). For a user-level install without root, set
`PREFIX` and `DATA_DIR` under your home directory — see the header of
`install.sh`. The unit runs the server as the `steelbound` user with
`Restart=on-failure` and baseline hardening
(`NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome`,
`PrivateTmp`, with `ReadWritePaths` limited to the data dir).

## Data dir / volume notes

The data dir holds `steelbound.db` — the append-only, hash-chained
ledger plus all memory — and a `workspace/` folder for file tools and
daily episodes. It is the one thing worth backing up
(`steelbound backup` makes a verified zip snapshot). A fresh/empty
data dir simply starts a new core with a new ledger; pointing a new
container or service at an old data dir continues the same core.
Never run two servers against one data dir at the same time.

## Loopback warning

The HTTP API binds **127.0.0.1 by design** and the server refuses any
non-loopback bind host before opening a socket (`server.py`). There
is no authentication on the API — loopback *is* the access control,
backed by the same deliberation gate as the CLI. Do not widen the
Compose port mapping or otherwise expose the port publicly. If you
want remote access, putting it behind a reverse proxy, VPN, or SSH
tunnel — with that layer's own authentication — is your decision as
operator, and it changes the threat model; Steelbound makes no
guarantees for a non-loopback deployment.

## Upgrading

The procedure is the same on every path: install the new release,
then restart the process. Memory and ledger in the data dir are
carried forward untouched.

- pip: `pip install --upgrade .` (new checkout/archive), re-run your command.
- Docker: `docker compose -f deploy/docker-compose.yml up -d --build`
- systemd: `sudo ./deploy/install.sh && sudo systemctl restart steelbound`

Run `steelbound --data-dir <dir> evals` after an upgrade as a smoke
check; all five suites should pass before the core serves traffic.

## License

Steelbound Source-Available License v1.0 — see `../LICENSE`.
Not legal advice; counsel review required before public or
commercial distribution.
