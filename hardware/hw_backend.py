"""Connection layer for Quandela Cloud.

Token resolution order:
  1. PCVL_CLOUD_TOKEN env var (perceval's native default)
  2. QUANDELA_TOKEN env var
  3. token previously saved on this machine via `pcvl.save_token(...)`

Never hardcode or commit a token.
"""
import os
import time

from perceval.runtime import RemoteConfig, RemoteProcessor

DEFAULT_PLATFORM = "qpu:ascella"
LOCAL_PLATFORM = "sim:slos"  # cloud-side simulator, useful to test the full remote path cheaply


def get_token() -> str:
    token = os.environ.get("PCVL_CLOUD_TOKEN") or os.environ.get("QUANDELA_TOKEN")
    if not token:
        try:
            token = RemoteConfig().get_token()
        except Exception:
            token = None
    if not token:
        raise RuntimeError(
            "No Quandela Cloud token found. Get one at https://cloud.quandela.com "
            "(account settings > tokens), then either:\n"
            "  export PCVL_CLOUD_TOKEN='...'\n"
            "or save it once with:\n"
            "  python -c \"from perceval.runtime import RemoteConfig; "
            "rc = RemoteConfig(); rc.set_token('...'); rc.save()\""
        )
    return token


def get_processor(platform: str = DEFAULT_PLATFORM, m: int = None,
                  request_timeout: int = 60, retries: int = 3) -> RemoteProcessor:
    """Return a RemoteProcessor connected to the given cloud platform.

    The default perceval RPC read timeout is 10s, which the cloud API exceeds
    under load — both during job polling AND inside the RemoteProcessor
    constructor (it fetches platform details). So the RPCHandler is built
    beforehand with a generous timeout, and construction is retried.

    m: number of modes to reserve on the chip (defaults to the circuit's size
       when the circuit is set).
    """
    from perceval.runtime.rpc_handler import RPCHandler

    url = RemoteConfig().get_url() or "https://api.cloud.quandela.com"
    handler = RPCHandler(platform, url, get_token())
    handler.request_timeout = request_timeout

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return RemoteProcessor(rpc_handler=handler, m=m)
        except Exception as e:
            last_err = e
            print(f"[hw_backend] connect attempt {attempt}/{retries} failed: "
                  f"{type(e).__name__}: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    raise last_err


def platform_info(platform: str = DEFAULT_PLATFORM) -> dict:
    """Fetch platform specs (modes, max photons, constraints) from the cloud."""
    proc = get_processor(platform)
    return {
        "name": platform,
        "specs": getattr(proc, "specs", None),
        "constraints": getattr(proc, "constraints", None),
        "performance": getattr(proc, "performance", None),
    }


if __name__ == "__main__":
    import pprint

    info = platform_info()
    pprint.pprint(info)
