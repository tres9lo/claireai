#!/usr/bin/env python3
"""Opt-in nftables guard for ClaireAI's web egress.

The browser tools already resolve once, reject non-public addresses, and pin the
approved IPs.  This script is defence in depth: it drops TCP 80/443 traffic from
one UID to non-public ranges so a missed address format cannot reach the local
network.  It is Linux/nftables only and never runs unless --apply is passed.

    python scripts/egress_firewall.py --print
    sudo python scripts/egress_firewall.py --apply --uid "$(id -u)"
    sudo python scripts/egress_firewall.py --remove

Loopback is deliberately left alone because the local model and presence
server need it.  DNS is deliberately left alone because many owners use a
LAN resolver.  The runtime's address check remains the primary control.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

TABLE = "claireai_egress"

# TCP web ports only.  Blocking all protocols would also block a LAN DNS
# resolver, and blocking loopback would break the local model and presence
# server.  The browser never uses QUIC, so UDP 443 needs no rule here.
WEB_PORTS = (80, 443)

# Non-public IPv4 ranges.  Loopback (127.0.0.0/8) is intentionally absent.
BLOCKED_IPV4 = (
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.0.2.0/24",
    "192.88.99.0/24",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "224.0.0.0/4",
    "240.0.0.0/4",
)

# Non-public IPv6 ranges.  Native loopback (::1) is intentionally absent; the
# IPv4-mapped private ranges below include mapped loopback.  The well-known NAT64
# prefix is not blocked wholesale because it can carry a public IPv4 address;
# the runtime checks the embedded address, and network-specific NAT64 prefixes
# still need an egress gateway.
BLOCKED_IPV6 = (
    "::/128",
    # IPv4-mapped forms of the private ranges above.
    "::ffff:0:0/104",
    "::ffff:a00:0/104",
    "::ffff:6440:0/106",
    "::ffff:7f00:0/104",
    "::ffff:a9fe:0/112",
    "::ffff:ac10:0/108",
    "::ffff:c0a8:0/112",
    "64:ff9b:1::/48",
    "2001:2::/48",
    "2001:10::/28",
    "2001:db8::/32",
    "3fff::/20",
    "fc00::/7",
    "fe80::/10",
    "ff00::/8",
)


def ruleset(uid: int) -> str:
    """Return an nftables ruleset for one UID."""
    if not 1 <= uid <= 0xFFFFFFFF:
        raise ValueError("uid must be a non-root user id")
    ipv4 = ", ".join(BLOCKED_IPV4)
    ipv6 = ", ".join(BLOCKED_IPV6)
    ports = ", ".join(str(port) for port in WEB_PORTS)
    return (
        f"table inet {TABLE} {{\n"
        "  chain output {\n"
        "    type filter hook output priority 0; policy accept;\n"
        f"    meta skuid {uid} ip daddr {{ {ipv4} }} "
        f"tcp dport {{ {ports} }} drop\n"
        f"    meta skuid {uid} ip6 daddr {{ {ipv6} }} "
        f"tcp dport {{ {ports} }} drop\n"
        "  }\n"
        "}\n"
    )


def _default_uid() -> int:
    sudo_uid = os.environ.get("SUDO_UID", "")
    if sudo_uid.isdigit():
        return int(sudo_uid)
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        raise RuntimeError("--uid is required on this platform")
    return int(getuid())


def _run_nft(arguments: list[str], *, input_text: str | None = None) -> int:
    if shutil.which("nft") is None:
        raise RuntimeError("nft was not found; install nftables first")
    try:
        completed = subprocess.run(
            ["nft", *arguments],
            input=input_text,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"could not run nft: {exc}") from exc
    return completed.returncode


def _require_root() -> None:
    if sys.platform != "linux":
        raise RuntimeError("this helper currently supports Linux/nftables only")
    if os.geteuid() != 0:
        raise RuntimeError("--apply and --remove require root")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--print", action="store_true", help="print the ruleset (default)")
    action.add_argument("--apply", action="store_true", help="install or replace the ruleset")
    action.add_argument("--remove", action="store_true", help="remove the ruleset")
    parser.add_argument("--uid", type=int, default=None, help="UID whose web traffic to restrict")
    args = parser.parse_args(argv)

    try:
        uid = _default_uid() if args.uid is None else args.uid
        text = ruleset(uid)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    if args.remove:
        try:
            _require_root()
            # An absent table is already the desired state.
            _run_nft(["delete", "table", "inet", TABLE])
        except RuntimeError as exc:
            parser.error(str(exc))
        return 0

    if args.apply:
        try:
            _require_root()
            # Make re-running the command idempotent.
            _run_nft(["delete", "table", "inet", TABLE])
            if _run_nft(["-f", "-"], input_text=text) != 0:
                print("nft rejected the ruleset; nothing was applied", file=sys.stderr)
                return 1
        except RuntimeError as exc:
            parser.error(str(exc))
        print(f"installed nftables table inet {TABLE} for uid {uid}")
        return 0

    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
