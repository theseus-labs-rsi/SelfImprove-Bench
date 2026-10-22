"""Check container-side model routing without sending an API request."""

import json


def container_model_environment(environment):
    """Mask image-baked proxies while preserving host lowercase precedence."""
    result = dict(environment)
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        value = result.get(name, result.get(name.upper(), ""))
        result[name] = value
        result[name.upper()] = value
    return result


_CHECK_NETWORK = r"""
import json
import os
import socket
import sys
from urllib.parse import urlsplit
from urllib.request import getproxies_environment, proxy_bypass_environment

proxies = getproxies_environment()
checked = set()
for name in json.loads(sys.argv[1]):
    endpoint = os.environ.get(name) or os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
    try:
        target = urlsplit(endpoint)
        if target.scheme not in ('http', 'https') or not target.hostname:
            raise ValueError('invalid endpoint')
        target_port = target.port or (443 if target.scheme == 'https' else 80)
        bypass_host = target.netloc.rsplit('@', 1)[-1]
        proxy = None if proxy_bypass_environment(bypass_host, proxies) else (
            proxies.get(target.scheme) or proxies.get('all')
        )
        route = urlsplit(proxy if '://' in proxy else 'http://' + proxy) if proxy else target
        if not route.hostname:
            raise ValueError('invalid route')
        port = route.port or ({'https': 443, 'socks5': 1080, 'socks5h': 1080}.get(route.scheme, 80) if proxy else target_port)
    except ValueError:
        print('Invalid endpoint or proxy configuration for ' + name + '; check URL syntax.')
        sys.exit(1)
    key = (route.hostname, port)
    if key in checked:
        continue
    checked.add(key)
    try:
        with socket.create_connection(key, timeout=5):
            pass
    except OSError as error:
        kind = 'proxy' if proxy else 'endpoint'
        print('Cannot connect to ' + kind + ' ' + route.hostname + ':' + str(port)
              + ' for ' + name + ' (' + type(error).__name__ + ').')
        if proxy and route.hostname in ('localhost', '127.0.0.1', '::1'):
            print('Container loopback is not host loopback under bridge networking. '
                  'Verify the host proxy is listening, then on Linux opt in with '
                  'HYPERAGENTS_DOCKER_NETWORK=host, or use a container-reachable proxy address.')
        else:
            print('Check container DNS, routing, firewall, proxy and NO_PROXY settings.')
        sys.exit(1)
print('Container model routes are TCP-reachable (authentication and model availability not tested).')
"""


def check_model_network(container, runtime):
    """Fail before gameplay if the actual container cannot reach its model route."""
    names = sorted({
        model.get("base_url_env", "OPENAI_BASE_URL")
        for model in (runtime.task_model, runtime.meta_model)
    })
    result = container.exec_run(
        ["python", "-c", _CHECK_NETWORK, json.dumps(names)],
        workdir="/",
        environment=container_model_environment(runtime.llm_environment),
    )
    if result.exit_code != 0:
        output = result.output.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"HyperAgents network preflight failed: {output}")
