"""Opt-in checkpoint policy selected by the HTTP socket peer, never proxy headers."""
from dataclasses import dataclass
import ipaddress


TAIL_ROLE_KEY = "_strata_tail_role_token"  # server-owned; stripped from incoming JSON
CAPABILITY = b"request_tail_role=1"


def address(value):
    ip = ipaddress.ip_address(value)
    return str(ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped else ip)


@dataclass(frozen=True)
class ClientCachePolicy:
    tail_user_ips: frozenset[str] = frozenset()
    user_token: int | None = None

    @classmethod
    def from_config(cls, cfg, tokenizer):
        values = cfg.get("cache_tail_user_ips", [])
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError("cache_tail_user_ips: expected a list of IP addresses")
        try:
            ips = frozenset(address(v) for v in values)
        except ValueError:
            raise ValueError("cache_tail_user_ips: use literal IP addresses, not hostnames or networks") from None
        if not ips:
            return cls()
        role = tokenizer.encode("user", parse_special=True)
        if len(role) != 1 or not isinstance(role[0], int) or isinstance(role[0], bool) or role[0] < 0:
            raise ValueError("cache_tail_user_ips: the tokenizer must encode 'user' as one token")
        return cls(ips, role[0])

    def request(self, req, peer):
        try:
            matched = bool(self.tail_user_ips) and address(peer) in self.tail_user_ips
        except ValueError:
            matched = False
        if not matched and TAIL_ROLE_KEY not in req:
            return req  # the default request is unchanged
        result = dict(req)
        result.pop(TAIL_ROLE_KEY, None)
        if matched:
            result[TAIL_ROLE_KEY] = self.user_token
        return result

    def check_engine(self, executable, engine_args):
        if not self.tail_user_ips:
            return
        if "--batch" in engine_args or "--slots" in engine_args:
            raise ValueError("cache_tail_user_ips: parallel/batched engines are not supported")
        with open(executable, "rb") as stream:
            supported = CAPABILITY in stream.read()
        if not supported:
            raise ValueError("cache_tail_user_ips: this engine lacks request_tail_role=1 support")
