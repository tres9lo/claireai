# Browser egress, DNS pinning and proxies

ClaireAI's browser tools are the only code that fetches untrusted pages. They
must not become a path into the machine, the LAN or a cloud metadata service.

## What the runtime does

1. Resolve the hostname once.
2. Reject the URL if **any** answer is not globally routable. This includes
   private, loopback, link-local, CGNAT and other special-use ranges, plus
   IPv4-mapped, 6to4 and Teredo addresses whose embedded IPv4 destination is
   non-public.
3. Pin every approved numeric address to the host and port.
4. Connect only to those pinned addresses. The connection does not resolve the
   hostname again, and it fails over across the pinned addresses.
5. Repeat the check for every redirect hop.

NAT64 is handled in two places:

- `64:ff9b::/96` is checked using the embedded IPv4 address.
- `64:ff9b:1::/48` is always rejected because it is the local-use NAT64 prefix.

A network-specific NAT64 prefix can still map a globally routable IPv6 address
to a private IPv4 destination. The runtime cannot see that mapping. If your
network uses a custom NAT64 prefix, enforce the destination policy at the
egress gateway as well.

## Optional host firewall

`scripts/egress_firewall.py` is an opt-in Linux/nftables backstop. It prints
rules by default and only changes the firewall when `--apply` is passed:

```bash
python scripts/egress_firewall.py --print
sudo python scripts/egress_firewall.py --apply --uid "$(id -u)"
sudo python scripts/egress_firewall.py --remove
```

The rules drop TCP 80/443 traffic from that UID to non-public IPv4 and IPv6
ranges. They leave native loopback alone because the local model and presence
server need it, and leave DNS alone because many owners use a LAN resolver. They affect
every web client run by that UID, not just ClaireAI. The runtime check remains
the primary control; this is a second layer.

This helper is intentionally not applied by the installer. It needs root, it is
Linux-specific, and a host firewall policy is an owner decision.

## Proxies

ClaireAI ignores `HTTP_PROXY`, `HTTPS_PROXY` and similar environment variables.
An ordinary HTTP proxy resolves the target hostname itself, which puts the
public-first/private-second rebinding window back at the proxy.

If the network requires a proxy, use a controlled egress gateway that applies
the same destination policy and does not let an untrusted client choose an
arbitrary internal destination. Treat that gateway as part of the trust boundary;
do not simply enable environment proxies.

## What not to rely on

DNSSEC and DNS-over-HTTPS protect DNS authenticity or transport. They do not
prevent a domain from returning alternating public and private addresses.
Address pinning plus a destination allow-list is the control that matters.
