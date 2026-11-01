"""Executable resource-contract kernels for the new research design.

These are deterministic accounting/validation functions, NOT a packet simulator
or a replacement for ndnSIM. No empirical cache-hit multiplier is used. A caller
may merge requests only when synchrony is explicitly part of the input contract.
Amounts are network bytes, serialization lower bounds, and logical send events.
PHY airtime and pinned-cache enforcement require independent executor evidence.
"""
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from math import ceil, isfinite
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class Link:
    id: str
    source: str
    target: str
    reverse: str
    service_bps: float
    domain: str | None = None


@dataclass(frozen=True)
class CacheLease:
    tenant: str
    slice_id: str
    node: str
    name: str
    version: str
    bytes: int
    valid_from: float
    valid_until: float
    pinned: bool = False



def lease_key(lease):
    return (lease.tenant, lease.slice_id, lease.node, lease.name, lease.version)


def validate_lease(lease):
    if (not all(lease_key(lease)) or type(lease.bytes) is not int or lease.bytes <= 0
            or not isfinite(lease.valid_from) or lease.valid_from < 0
            or not isfinite(lease.valid_until) or lease.valid_until <= lease.valid_from):
        raise ValueError("invalid cache lease")


@dataclass(frozen=True)
class Reservation(MappingABC):
    """Traffic delta plus explicit storage dependencies and acquisitions.

    Existing cache dependencies are NOT charged again. New leases are proposals,
    not proof that an executor has loaded/pinned the object. CacheLedger is the
    authoritative interval/quota check at atomic commit, including concurrent
    plans created from stale snapshots.
    """
    amounts: Mapping[str, float]
    required_leases: tuple[CacheLease, ...] = ()
    new_leases: tuple[CacheLease, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "amounts", MappingProxyType(dict(self.amounts)))
        object.__setattr__(self, "required_leases", tuple(self.required_leases))
        object.__setattr__(self, "new_leases", tuple(self.new_leases))

    def __getitem__(self, key): return self.amounts[key]
    def __iter__(self): return iter(self.amounts)
    def __len__(self): return len(self.amounts)


def cache_usage(leases, at):
    """Physical payload bytes once per owned object at time t; intervals [a,b)."""
    active, usage = {}, defaultdict(int)
    for lease in leases:
        validate_lease(lease)
        if not lease.pinned or not lease.valid_from <= at < lease.valid_until:
            continue
        key = lease_key(lease)
        if key in active and active[key] != lease.bytes:
            raise ValueError("conflicting stored size for the same object/version")
        active[key] = lease.bytes
    for key, size in active.items(): usage[key[2]] += size
    return dict(usage)


class CacheLedger:
    """Reference-counted cache ownership; no cross-tenant/slice implicit sharing.

    Storage sizes are payload bytes, not allocator overhead or encoded Data size.
    Every owner retains its own interval. Release of the producer of a lease
    does not invalidate another admitted owner's dependency on that same object.
    """
    def __init__(self): self._owners = {}

    @property
    def leases(self):
        return tuple(l for group in self._owners.values() for l in group)

    def usage(self, at): return cache_usage(self.leases, at)

    def commit(self, owner, reservation, capacities):
        if not owner or owner in self._owners:
            raise ValueError("unique reservation owner required")
        _amounts(capacities)
        dependencies = tuple(reservation.required_leases)
        new = tuple(reservation.new_leases)
        for lease in dependencies + new:
            validate_lease(lease)
            if not lease.pinned: raise ValueError("cache reservations require pinned leases")
        for dep in dependencies:
            matches = [l for l in self.leases + new if l.pinned and
                       lease_key(l) == lease_key(dep) and l.bytes == dep.bytes]
            intervals = sorted((l.valid_from, l.valid_until) for l in matches)
            reach = dep.valid_from
            for start, end in intervals:
                if start > reach: break
                reach = max(reach, end)
            if reach < dep.valid_until:
                raise ValueError("cache dependency is no longer covered; re-plan")
        installed = dependencies + new
        proposed = self.leases + installed
        for at in sorted({l.valid_from for l in proposed}):
            for node, used in cache_usage(proposed, at).items():
                if node not in capacities or used > capacities[node]:
                    raise ValueError("cache capacity exceeded")
        self._owners[owner] = installed  # atomic after every check

    def release(self, owner):
        if owner not in self._owners: raise KeyError(owner)
        del self._owners[owner]

def _amounts(values):
    if any(not isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("resource quantities must be finite and nonnegative")


def object_reservation(*, links: Mapping[str, Link], paths: Mapping[str, tuple[str, ...]],
                       tenant: str, slice_id: str, name: str, version: str,
                       payload_bytes: int, start: float, end: float,
                       modality: str, synchronous=False, leases=(), pin_requests=(),
                       chunk_bytes=1000, data_header_bytes=32, interest_bytes=64):
    """Count one declared object delivery to every consumer on directed paths.

    Paths start at the producer and end at each consumer. NDN may truncate a
    path at an already-pinned same-slice cache valid for the ENTIRE interval.
    Synchronous NDN merges tree edges only; asynchronous cold requests remain
    separate. Wire fields include the supplied headers; MAC retransmission and
    simulated CPU cost need separately calibrated contracts in the full model.
    IP here is explicitly the unicast comparator, not a claim that IP cannot
    implement application caching or multicast.
    """
    if modality not in ("ip_unicast", "ndn"):
        raise ValueError("unknown object-transfer mechanism")
    if not paths or not all((tenant, slice_id, name, version)):
        raise ValueError("complete object and tenant identity required")
    if (not isfinite(start) or not isfinite(end) or start < 0 or end <= start
            or any(not isinstance(v, int) or v < 0 for v in
                   (payload_bytes, chunk_bytes, data_header_bytes, interest_bytes))
            or min(payload_bytes, chunk_bytes) <= 0):
        raise ValueError("invalid object size or contract interval")
    leases, pin_requests = tuple(leases), tuple(pin_requests)
    for lease in leases + pin_requests: validate_lease(lease)
    if any(not lease.pinned for lease in pin_requests):
        raise ValueError("new pin request must require pinning")
    for key, link in links.items():
        reverse = links.get(link.reverse)
        if (key != link.id or not isfinite(link.service_bps) or link.service_bps <= 0
                or link.source == link.target or reverse is None
                or (reverse.source, reverse.target, reverse.reverse) !=
                   (link.target, link.source, link.id)):
            raise ValueError("directed links need explicit valid reverse links")

    selected = []
    required = {}
    root = None
    for consumer, path in paths.items():
        if not consumer or not path or any(k not in links for k in path):
            raise ValueError("each consumer needs a nonempty known path")
        edges = [links[k] for k in path]
        nodes = [edges[0].source] + [e.target for e in edges]
        if (len(set(nodes)) != len(nodes)
                or any(a.target != b.source for a, b in zip(edges, edges[1:]))):
            raise ValueError("path must be continuous and simple")
        if root is None:
            root = nodes[0]
        elif root != nodes[0]:
            raise ValueError("one object contract requires one producer")
        cut = 0
        if modality == "ndn":
            for i, node in enumerate(nodes):
                eligible = any(
                    (l.tenant, l.slice_id, l.node, l.name, l.version) ==
                    (tenant, slice_id, node, name, version)
                    and l.pinned and l.bytes >= payload_bytes
                    and l.valid_from <= start and l.valid_until >= end
                    for l in leases + pin_requests)
                if eligible:
                    cut = i  # nearest eligible cache on this declared route
            candidates = [l for l in leases + pin_requests
                          if lease_key(l) == (tenant, slice_id, nodes[cut], name, version)
                          and l.pinned and l.bytes >= payload_bytes
                          and l.valid_from <= start and l.valid_until >= end]
            if candidates:
                chosen = min(candidates, key=lambda l: l.bytes)
                required[lease_key(chosen)] = CacheLease(
                    tenant, slice_id, chosen.node, name, version, chosen.bytes, start, end, True)
        selected.append(tuple(path[cut:]))

    # This kernel supports arborescences, not arbitrary reconverging DAGs. A
    # richer topology must be split into independently scheduled delivery groups.
    if modality == "ndn" and synchronous:
        union = set(k for path in selected for k in path)
        parent = {}
        adjacency = defaultdict(set)
        for key in union:
            edge = links[key]
            if edge.target in parent and parent[edge.target] != key:
                raise ValueError("a synchronous group must form a delivery forest")
            parent[edge.target] = key
            adjacency[edge.source].add(edge.target)
        visiting, visited = set(), set()

        def visit(node):
            if node in visiting:
                raise ValueError("delivery forest cannot contain a cycle")
            if node in visited:
                return
            visiting.add(node)
            for nxt in adjacency[node]:
                visit(nxt)
            visiting.remove(node)
            visited.add(node)

        for node in list(adjacency):
            visit(node)
        groups = [sorted(union)]
    else:
        groups = selected

    chunks = ceil(payload_bytes / chunk_bytes)
    amounts = defaultdict(float)
    for group in groups:
        for key in group:
            edge = links[key]
            transmissions = [(edge, payload_bytes + chunks * data_header_bytes, chunks)]
            if modality == "ndn":
                transmissions.append((links[edge.reverse], chunks * interest_bytes, chunks))
            for hop, wire_bytes, events in transmissions:
                amounts[f"bytes:{hop.id}"] += wire_bytes
                amounts[f"tx_events:{hop.source}"] += events
                if hop.domain:
                    amounts[f"serialization_lower_bound_seconds:{hop.domain}"] += wire_bytes * 8 / hop.service_bps
    # Incremental peak storage is informative only; atomic CacheLedger.commit
    # rechecks complete intervals against concurrent owners. Never sum this delta
    # as a substitute for the authoritative object-keyed ledger.
    for node in {l.node for l in pin_requests}:
        times = {t for l in leases + pin_requests if l.node == node for t in (l.valid_from,l.valid_until)}
        delta = max((cache_usage(leases + pin_requests, t).get(node, 0)
                     - cache_usage(leases, t).get(node, 0) for t in times), default=0)
        if delta: amounts[f"new_pinned_cache_bytes:{node}"] = delta
    return Reservation(amounts, tuple(required.values()), pin_requests)


@dataclass(frozen=True)
class MobilityScenario:
    id: str
    access_nodes: frozenset[str]
    links: frozenset[str]
    usage: Mapping[str, float]

    def __post_init__(self):
        # Copy caller-owned containers: installed contracts cannot silently
        # change when a candidate builder reuses its scratch dictionaries.
        object.__setattr__(self, "access_nodes", frozenset(self.access_nodes))
        object.__setattr__(self, "links", frozenset(self.links))
        object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))


@dataclass(frozen=True)
class MobilityEnvelope:
    """Preinstalled mutually exclusive attachment scenarios, for ALL modes.

    Sum all concurrent traffic/hops WITHIN a scenario first, including hops in
    the same competition domain; reserve the resource-wise maximum ACROSS
    scenarios. Sum envelopes across slices. Make-before-break dual attachment
    must appear as its own scenario; a locator update never creates capacity.
    """
    scenarios: tuple[MobilityScenario, ...]

    def __post_init__(self):
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        if not self.scenarios or len({s.id for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("unique nonempty attachment scenarios required")
        for scenario in self.scenarios:
            if not scenario.id or not scenario.access_nodes:
                raise ValueError("attachment identity required")
            _amounts(scenario.usage)

    def reservation(self):
        keys = set().union(*(s.usage for s in self.scenarios))
        return {k: max(s.usage.get(k, 0.) for s in self.scenarios) for k in sorted(keys)}

    def permits(self, access_nodes, links, usage):
        _amounts(usage)
        # Require ONE complete scenario, not arbitrary mixing of the union.
        return any(set(access_nodes) <= s.access_nodes and set(links) <= s.links
                   and all(v <= s.usage.get(k, 0.) + 1e-9 for k, v in usage.items())
                   for s in self.scenarios)

    def check_capacity(self, capacities, existing=None):
        existing = existing or {}
        _amounts(capacities)
        _amounts(existing)
        total = dict(existing)
        for key, value in self.reservation().items():
            total[key] = total.get(key, 0.) + value
        if any(k not in capacities or v > capacities[k] + 1e-9 for k, v in total.items()):
            raise ValueError("mobility envelope exceeds available resources")
        return True
