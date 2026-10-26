"""HTTP balancer for OpenAI-compatible vLLM instances, with prefix affinity.

Requests that carry ``X-CaP-Affinity`` stay on one instance (rendezvous
hashing with a load slack, ``prefix_cache.choose_backend``); others go to the
instance with the fewest requests in flight. Responses name the instance in
``X-CaP-Backend``. Backends are read from a JSON list on every request, so
instances can be added or removed without restarting clients. Bodies are
buffered (no streaming).

    python -m synthesis.cap.serving.balancer PORT BACKENDS_JSON [--slack N]

Run it with an interpreter that has ``aiohttp`` (the vLLM serving venv) and
``roboverify/`` on ``PYTHONPATH``. The CaP runtime does not import this module.
"""

import argparse
import json
from pathlib import Path

from .prefix_cache import AFFINITY_HEADER, BACKEND_HEADER, DEFAULT_SLACK, choose_backend

FORWARDED_HEADERS = ("authorization", "content-type")


def read_backends(path):
    try:
        return [b for b in json.loads(Path(path).read_text()) if b]
    except (OSError, ValueError):
        return []


def application(backends_path, slack=DEFAULT_SLACK):
    import aiohttp
    from aiohttp import web

    inflight = {}

    async def forward(request):
        body = await request.read()
        headers = {
            k: v for k, v in request.headers.items() if k.lower() in FORWARDED_HEADERS
        }
        key = request.headers.get(AFFINITY_HEADER) or None
        tried = []
        for _ in range(3):
            pool = [b for b in read_backends(backends_path) if b not in tried]
            if not pool:
                break
            backend = choose_backend(pool, inflight, key, slack)
            tried.append(backend)
            inflight[backend] = inflight.get(backend, 0) + 1
            try:
                async with request.app["session"].request(
                    request.method,
                    backend + request.path_qs,
                    data=body,
                    headers=headers,
                ) as resp:
                    payload = await resp.read()
                    return web.Response(
                        body=payload,
                        status=resp.status,
                        content_type=resp.content_type,
                        headers={BACKEND_HEADER: backend},
                    )
            except aiohttp.ClientConnectorError:
                continue
            finally:
                inflight[backend] -= 1
        return web.Response(status=503, text="no backend available")

    async def status(request):
        return web.json_response(
            {
                "backends": read_backends(backends_path),
                "inflight": inflight,
                "affinity_header": AFFINITY_HEADER,
                "slack": slack,
            }
        )

    async def start(app):
        app["session"] = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=7200, sock_connect=10),
            connector=aiohttp.TCPConnector(limit=0),
        )

    async def stop(app):
        await app["session"].close()

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    app.router.add_get("/balancer/status", status)
    app.router.add_route("*", "/{tail:.*}", forward)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("port", type=int)
    parser.add_argument("backends", help="JSON list of instance base URLs")
    parser.add_argument("--slack", type=int, default=DEFAULT_SLACK)
    args = parser.parse_args()
    from aiohttp import web

    web.run_app(
        application(args.backends, args.slack),
        host="127.0.0.1",
        port=args.port,
        access_log=None,
    )


if __name__ == "__main__":
    main()
