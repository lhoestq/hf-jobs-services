# Paired with use_server-services.yml: `hf jobs-services run use_server.py`
#
# Members of a services group are resolvable before they are ready, so the client retries instead of
# assuming the server is up when the Job starts.
import os
import time
import urllib.request

prefix = os.environ.get("HF_NETWORK_GROUP_PREFIX", "")
url = f"http://{prefix}server:8000/"

for _ in range(40):
    try:
        body = urllib.request.urlopen(url, timeout=3).read(80)
        print("SMOKE OK", url, "->", body[:40], flush=True)
        break
    except Exception:
        print("waiting for the server...", flush=True)
        time.sleep(3)
else:
    raise SystemExit("the server never answered")
