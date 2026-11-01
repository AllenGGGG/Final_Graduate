"""Build and verify native ns-3/ndnSIM protocol integration. Run in sdn.

python -B -m slice_alloc.tools.platform_verify
Third-party source/build stays in ~/.local/share/lowalt-ndnsim, outside the
manuscript. Exact commits, compatibility patches, commands and log hashes are
retained. Tests do not claim per-slice isolation or a trained allocation policy.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from ..protocol_contracts import Link, CacheLease, CacheLedger, object_reservation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "slice_alloc/results/platform"
NS3 = Path.home() / ".local/share/lowalt-ndnsim/ns-3"
PINS = {
    "ns3": "1b3bab9b6dba5f1e616c0b286eb889e9ce3d5d59",
    "ndnSIM": "90d50396654dabad54b6979f2dc8fa929ade544c",
    "NFD": "361b381b9b552ff3341aa160858bea1d470d2122",
    "ndn-cxx": "2b51e5796415b5d41fe268b35250b9453f16d3d6",
}
CASES = {
    "wired_sync_cache": dict(wireless=0, cache=1, synchronous=1, speed=0),
    "wired_sync_cold": dict(wireless=0, cache=0, synchronous=1, speed=0),
    "wired_async_cold": dict(wireless=0, cache=0, synchronous=0, speed=0),
    "wifi_sync_cache": dict(wireless=1, cache=1, synchronous=1, speed=0),
    "wifi_sync_cold": dict(wireless=1, cache=0, synchronous=1, speed=0),
    "wifi_async_cold": dict(wireless=1, cache=0, synchronous=0, speed=0),
    "wifi_moving_cache": dict(wireless=1, cache=1, synchronous=1, speed=5),
    "wifi_outage_cache": dict(wireless=1, cache=1, synchronous=1, speed=80),
    "wired_pin": dict(wireless=0, cache=1, synchronous=1, speed=0, pin=1),
    "wired_pin_control": dict(wireless=0, cache=1, synchronous=1, speed=0, pin=1,pinControl=1),
}

EQUIVALENT_CASES = {
    f"equivalent_{base}_{mode}": dict(CASES[base],taskMode="equivalent_"+mode)
    for base in ("wired_sync_cache","wired_sync_cold","wired_async_cold","wifi_sync_cache","wifi_sync_cold")
    for mode in ("ip","ndn")
}

# Frozen before confirmation: no group is selected by which protocol wins.
CONTROLLED_SEEDS = (41, 42, 43)
CONTROLLED_CASES = {
    f"l2_{profile}_{group}_{mode}": dict(wireless=1,cache=cache,synchronous=sync,speed=0,
        taskMode="equivalent_"+mode,equalRetry=1,uniformL2=uniform,peerUnicast=peer)
    for profile,cache,sync in (("async_cold",0,0),("sync_cold",0,1),("sync_cache",1,1))
    for group,uniform,peer in (("native",0,0),("common_phy_broadcast",1,0),("aligned_unicast",1,1))
    for mode in ("ip","ndn")
}
CONTROLLED_CASES['l2_peer_filter_negative'] = dict(wireless=1,cache=1,synchronous=1,speed=0,
                                                 peerUnicast=1,uniformL2=1,filterTest=1)


def frame_accounting(frames):
    """Actual PHY TX MPDUs. Business data excludes ARP and link controls."""
    from collections import Counter
    business=[r for r in frames if r['frame_type']=='data' and int(r['ethertype']) in (2048,30583)]
    def signature(r):
        return '|'.join(r[k] for k in ('mode','phy_bps','nss','gi_ns','width_mhz','preamble'))
    kinds={r['destination'] for r in business}
    return dict(
        unicast_frame_bytes=sum(int(r['bytes']) for r in business if r['destination']=='unicast'),
        broadcast_frame_bytes=sum(int(r['bytes']) for r in business if r['destination']=='broadcast'),
        multicast_frame_bytes=sum(int(r['bytes']) for r in business if r['destination']=='multicast'),
        l2_delivery_mode=next(iter(kinds)) if len(kinds)==1 else ('mixed' if kinds else 'none'),
        actual_phy_mode_histogram=dict(Counter(signature(r) for r in business)),
        actual_control_phy_mode_histogram=dict(Counter(signature(r) for r in frames if r['frame_type']=='control')),
        actual_qos_tids=sorted({r.get('qos_tid','unavailable') for r in business}),
        aggregate_delimiter_padding_bytes=sum(int(r.get('aggregate_overhead_bytes',0)) for r in frames),
        business_frames=len(business),business_mac_retries=sum(int(r['retry']) for r in business),
        business_aggregated_mpdus=sum(int(r['aggregate']) for r in business),
        arp_frame_bytes=sum(int(r['bytes']) for r in frames if int(r['ethertype'])==2054),
        control_frame_bytes=sum(int(r['bytes']) for r in frames if r['frame_type']=='control'),
        mac_management_frame_bytes=sum(int(r['bytes']) for r in frames if r['frame_type']=='management'),
        scope='transmitted business data frames for delivery mode/bytes; actual control and ARP frames reported separately; NOT a PHY rate inferred from throughput')


def export_radio_evidence(radio, frames):
    """Expose measured L2 fields and a separately labelled derived byte rate."""
    layer = frame_accounting(frames)
    duration = radio['tx_seconds_from_state']
    return dict(radio, link_layer=layer,
                l2_delivery_mode=layer['l2_delivery_mode'],
                actual_phy_mode_histogram=layer['actual_phy_mode_histogram'],
                implied_rate_mbps=radio['phy_frame_bytes'] * 8 / duration / 1e6 if duration else None,
                implied_rate_scope='derived MPDU bytes * 8 / SUM of transmitter PHY TX seconds / 1e6; '
                    'includes transmitted control/retry MPDUs, excludes idle/backoff; '
                    'NOT the channel-interval union, application goodput, measured PHY bitrate, or admission capacity')


def compare_controlled(results):
    pairs=[]
    for profile in ('async_cold','sync_cold','sync_cache'):
        for group in ('native','common_phy_broadcast','aligned_unicast'):
            for seed in CONTROLLED_SEEDS:
                records={m:next(r for r in results if r['case']==f'l2_{profile}_{group}_{m}' and r['seed']==seed) for m in ('ip','ndn')}
                traces={m:list(csv.DictReader((OUT/f'l2_{profile}_{group}_{m}_{seed}-events.csv').open())) for m in records}
                def tasks(mode):
                    kind='object_ip_interest' if mode=='ip' else 'app_interest'
                    return [(r['time'],r['node'],r['name']) for r in traces[mode] if r['event']==kind]
                radio={m:records[m]['radio_accounting'] for m in records}
                layer={m:radio[m]['link_layer'] for m in records}
                common_modes=(set(layer['ip']['actual_phy_mode_histogram'])==set(layer['ndn']['actual_phy_mode_histogram']))
                config_keys=('manager','data_mode','control_mode','nonunicast_mode','requested_max_ssrc','requested_max_slrc',
                             'BE_ampdu','BE_amsdu','BK_ampdu','BK_amsdu','VI_ampdu','VI_amsdu','VO_ampdu','VO_amsdu')
                configs={m:{tuple(row[k] for k in config_keys) for row in records[m]['radio_configuration']} for m in records}
                config_equal=configs['ip']==configs['ndn'] and len(configs['ip'])==1
                aligned=(common_modes and config_equal and layer['ip']['actual_qos_tids']==layer['ndn']['actual_qos_tids'] and layer['ip']['l2_delivery_mode']==layer['ndn']['l2_delivery_mode']=='unicast'
                         and all(records[m]['uniform_l2'] and records[m]['equal_retry'] for m in records)
                         and all(layer[m]['business_aggregated_mpdus']==layer[m]['arp_frame_bytes']==0 for m in records))
                checks=dict(same_logical_tasks=tasks('ip')==tasks('ndn') and len(tasks('ip'))==6,
                    same_deadline_and_retry_budget=all(records[m]['equal_retry'] for m in records),
                    six_objects_delivered_on_each_side=records['ip']['object_ip_received']==records['ndn']['ndn_received']==6,
                    controls_identifiable_exactly_for_aligned_group=aligned==(group=='aligned_unicast'),
                    exported_radio_fields_match_measured_evidence=all(
                        radio[m]['l2_delivery_mode']==layer[m]['l2_delivery_mode']
                        and radio[m]['actual_phy_mode_histogram']==layer[m]['actual_phy_mode_histogram']
                        and bool(radio[m]['actual_phy_mode_histogram'])
                        and radio[m]['implied_rate_mbps']==radio[m]['phy_frame_bytes']*8/radio[m]['tx_seconds_from_state']/1e6
                        for m in records))
                if group!='native':
                    checks['actual_business_phy_modes_match']=common_modes
                    checks['actual_mac_configuration_matches']=config_equal
                    checks['actual_qos_access_class_matches']=layer['ip']['actual_qos_tids']==layer['ndn']['actual_qos_tids']
                payload={m:sum(int(r['bytes']) for r in traces[m] if r['event']==('object_ip_produce' if m=='ip' else 'producer_payload')) for m in records}
                confounds=[]
                if layer['ip']['l2_delivery_mode']!=layer['ndn']['l2_delivery_mode']:confounds.append('business destination type and ACK policy differ')
                if not common_modes:confounds.append('actual PHY modes differ')
                if group=='native':confounds.append('native aggregation/neighbor preparation differ')
                pairs.append(dict(profile=profile,group=group,seed=seed,checks=checks,passed=all(checks.values()),
                    link_layer_conditions_aligned=aligned,protocol_effect_identifiable_under_controls=aligned,
                    confounding_factors=confounds,ip_radio=radio['ip'],ndn_radio=radio['ndn'],
                    producer_payload_bytes=payload,ip_completed=records['ip']['object_ip_received'],ndn_completed=records['ndn']['ndn_received'],
                    ip_application_retries=records['ip']['object_ip_retransmissions'],ndn_application_retries=records['ndn']['ndn_retransmissions'],
                    mean_delivery_delay_ms={m:sum(float(r['detail'])*1000 for r in traces[m] if r['event']==('object_ip_data' if m=='ip' else 'app_data'))/6 for m in records},
                    ndn_to_ip_phy_tx_ratio=radio['ndn']['tx_seconds_from_state']/radio['ip']['tx_seconds_from_state']))
    return dict(passed=all(r['passed'] for r in pairs),pairs=pairs,confirmation_seeds=list(CONTROLLED_SEEDS),
                scope='conditional comparison of implementations under measured controls; not universal protocol causality or allocation/RL results',
                controls='common PHY: ERP6 data/control/nonunicast, no A-MPDU/A-MSDU, permanent prepared ARP, same MAC limits and 5s application retry contract; aligned group additionally uses peer-addressed NDN transport',
                remaining=['IP app-cache and multicast enhancement comparators','independent service-envelope calibration','multi-chunk/large-object/mobility and congestion sweeps'])


def radio_accounting(rows):
    """Cross-check actual PHY clocks; sum and union have distinct semantics.

    Sum = accumulated transmitter service demand, including simultaneous sends.
    Union = channel transmit occupancy without counting overlaps twice. Neither
    is a queue-delay or schedulable-throughput guarantee. Includes all protocols,
    link-layer control frames and retries in this run, not only NDN payload.
    """
    starts, intervals = {}, []
    physical_bytes=0
    for row in rows:
        key=(row["node"],row["face"])
        if row["event"]=="phy_begin":
            if key in starts: raise ValueError("overlapping packet TX for same radio/uid")
            starts[key]=float(row["time"]);physical_bytes+=int(row["bytes"])
        elif row["event"]=="phy_end":
            intervals.append((row["node"],starts.pop(key),float(row["time"])))
    if starts: raise ValueError("uncensored PHY transmissions required")
    # PhyTxBegin/End are emitted per MPDU, but an aggregated PPDU occupies
    # the radio only once. Same-radio MPDUs with identical TX interval share it.
    ppdus=set(intervals)
    durations=sum(b-a for node,a,b in ppdus)
    state_sum=sum(float(r["detail"]) for r in rows if r["event"]=="phy_state_tx")
    union=0.;end=0.
    for a,b in sorted((a,b) for node,a,b in ppdus):
        union+=max(0.,b-max(a,end));end=max(end,b)
    count=len(intervals)
    return dict(phy_mpdu_events=count,phy_transmission_events=len(ppdus),phy_frame_bytes=physical_bytes,
                tx_seconds_from_begin_end=durations,tx_seconds_from_state=state_sum,
                channel_union_tx_seconds=union,
                independent_phy_clocks_agree=abs(durations-state_sum)<1e-7,
                observed_tx_seconds_per_frame_byte=durations/physical_bytes if physical_bytes else 0.,
                scope="all-radio PPDU TX including overhead/control/retries; MPDU intervals deduplicated within each radio; excludes idle/backoff; empirical, not admission guarantee")


def compare_equivalent(results):
    pairs=[]
    for base in ("wired_sync_cache","wired_sync_cold","wired_async_cold","wifi_sync_cache","wifi_sync_cold"):
        for seed in (11,12,13):
            records={m:next(r for r in results if r['case']==f'equivalent_{base}_{m}' and r['seed']==seed) for m in ('ip','ndn')}
            traces={m:list(csv.DictReader((OUT/f'equivalent_{base}_{m}_{seed}-events.csv').open())) for m in records}
            def tasks(mode):
                event='object_ip_interest' if mode=='ip' else 'app_interest'
                return [(r['time'],r['node'],r['name']) for r in traces[mode] if r['event']==event]
            ip_payload=sum(int(r['bytes']) for r in traces['ip'] if r['event']=='object_ip_produce')
            ndn_payload=sum(int(r['bytes']) for r in traces['ndn'] if r['event']=='producer_payload')
            identical=tasks('ip')==tasks('ndn') and len(tasks('ip'))==6
            pairs.append(dict(case=base,seed=seed,identical_tasks=identical,
                ip_completed=records['ip']['object_ip_received'],ndn_completed=records['ndn']['ndn_received'],
                ip_producer_payload_bytes=ip_payload,ndn_producer_payload_bytes=ndn_payload,
                ip_radio=records['ip'].get('radio_accounting'),ndn_radio=records['ndn'].get('radio_accounting'),
                link_layer_conditions_aligned=False if records['ip'].get('radio_accounting') else None,
                protocol_effect_identifiable_under_controls=False if records['ip'].get('radio_accounting') else None,
                confounding_factors=['native IP unicast and NDN broadcast; PHY/aggregation/control behavior differ'] if records['ip'].get('radio_accounting') else []))
    return dict(pairs=pairs,passed=all(p['identical_tasks'] and p['ip_completed']==p['ndn_completed']==6 for p in pairs),
                scope='IP unicast without application cache vs NDN; one-chunk objects, no claim against IP multicast/app-cache; wired task clocks exact',
                remaining=['multicast/application-cache IP comparators','larger objects and multi-chunk congestion sweep'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, cwd=None, log=None, env=None):
    result = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                            capture_output=True, text=True)
    if log:
        Path(log).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command}\n{result.stderr[-2000:]}")
    return result.stdout.strip()


def ensure_repository(path, url, commit):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", url, path])
        if run(["git", "rev-parse", "HEAD"], cwd=path) != commit:
            run(["git", "fetch", "--depth", "1", "origin", commit], cwd=path)
            run(["git", "checkout", "--detach", commit], cwd=path)
    actual = run(["git", "rev-parse", "HEAD"], cwd=path)
    if actual != commit:
        raise ValueError(f"unexpected platform source at {path}: {actual}; expected {commit}")


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_repository(NS3, "https://github.com/named-data-ndnSIM/ns-3-dev.git", PINS["ns3"])
    module = NS3 / "src/ndnSIM"
    ensure_repository(module, "https://github.com/named-data-ndnSIM/ndnSIM.git", PINS["ndnSIM"])
    if not (module / "NFD/daemon/fw/strategy.cpp").exists():
        run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=module)
    for name in ("NFD", "ndn-cxx"):
        if run(["git", "rev-parse", "HEAD"], cwd=module / name) != PINS[name]:
            raise ValueError("submodule pin mismatch: " + name)
    # Small, recorded compile/runtime compatibility fixes, not protocol rewards.
    scheduler = module / "ndn-cxx/ndn-cxx/util/scheduler.hpp"
    text = scheduler.read_text()
    if "#include <optional>" not in text:
        scheduler.write_text(text.replace("#include <set>", "#include <set>\n#include <optional>"))
    strategy = module / "NFD/daemon/fw/strategy.cpp"
    text = strategy.read_text()
    before = "  m_forwarder.onOutgoingData(data, egress);\n\n  if (pitEntry->getInRecords().empty())"
    if before in text:
        text = text.replace(before, "  bool sent = m_forwarder.onOutgoingData(data, egress);\n\n  if (pitEntry->getInRecords().empty())", 1)
        text = text.replace("    pitEntry->isSatisfied = true;\n  }\n}\n\nvoid\nStrategy::sendDataToAll",
                            "    pitEntry->isSatisfied = true;\n  }\n  return sent;\n}\n\nvoid\nStrategy::sendDataToAll", 1)
        strategy.write_text(text)
    if "return sent;" not in strategy.read_text():
        raise ValueError("NFD return fix was not applied")
    source = ROOT / "slice_alloc/execution/protocol_probe.cc"
    target = NS3 / "scratch/protocol-probe.cc"
    if not target.exists() or sha(target) != sha(source):
        shutil.copyfile(source, target)
    if not (NS3 / "build/c4che/_cache.py").exists():
        run(["/usr/bin/python3", "waf", "configure", "-d", "optimized", "--disable-python",
             "--disable-tests", "--disable-examples",
             "--enable-modules=ndnSIM,wifi,internet,applications,mobility,point-to-point,flow-monitor"],
            cwd=NS3, log=OUT / "configure.log")
    run(["/usr/bin/python3", "waf", "-j4", "--targets=protocol-probe"],
        cwd=NS3, log=OUT / "probe-build.log")
    return {
        "pins": PINS, "compiler": run(["g++", "--version"]).splitlines()[0],
        "ns3_release": "3.35 companion fork", "system_python": run(["/usr/bin/python3", "--version"]),
        "ndn_cxx_compatibility_patch": run(["git", "diff", "--", "ndn-cxx/util/scheduler.hpp"], cwd=module / "ndn-cxx"),
        "nfd_runtime_patch": run(["git", "diff", "--", "daemon/fw/strategy.cpp"], cwd=module / "NFD"),
        "source_sha256": sha(source), "binary_sha256": sha(NS3 / "build/scratch/protocol-probe"),
        "library_sha256": {p.name: sha(p) for p in sorted((NS3 / "build/lib").glob("*.so"))},
        "wifi_face_type": "native broadcast AD_HOC or explicit peer-filtered POINT_TO_POINT on the same shared radio; selected per recorded case",
        "identity_wire_scope": "custom 17-byte EID/RLOC mechanism header; NOT interoperable LISP",
    }


def analyze(prefix, name, config):
    summary_path = Path(str(prefix) + "-summary.json")
    rows = list(csv.DictReader(Path(str(prefix) + "-events.csv").open()))
    summary = json.loads(summary_path.read_text())
    checks = {}
    if config['wireless']:
        frames=list(csv.DictReader(Path(str(prefix)+'-frames.csv').open()))
        summary['radio_accounting']=export_radio_evidence(radio_accounting(rows), frames)
        summary['radio_configuration']=list(csv.DictReader(Path(str(prefix)+'-radio-config.csv').open()))
        checks['phy_monitor_and_packet_event_counts_match']=len(frames)==summary['radio_accounting']['phy_mpdu_events']
        checks['phy_monitor_and_packet_bytes_match']=sum(int(r['bytes']) for r in frames)==summary['radio_accounting']['phy_frame_bytes']
    if config.get("taskMode"):
        mode=config['taskMode']
        checks['six_equivalent_tasks_completed']=(summary['object_ip_received']==6 if mode=='equivalent_ip' else summary['ndn_received']==6)
        checks['completions_before_earliest_task_deadline']=all(float(r['time'])<6. for r in rows if r['event'] in ('object_ip_data','app_data'))
        checks['no_coexist_background_traffic']=summary['ip_sent']==summary['id_sent']==0
        checks['objects_are_1000_byte_payloads']=all(int(r['bytes'])==1000 for r in rows if r['event'] in ('object_ip_data','app_data'))
        if config.get('peerUnicast') and mode=='equivalent_ndn':
            obj='/tenantA/slice1/map/v1/chunk0'
            def upstream_count(a,b,name):
                return sum(r['event']=='out_interest' and r['node']=='1' and r['face']=='257'
                           and a<=float(r['time'])<b and r['name']==name for r in rows)
            checks['peer_transport_keeps_pit_aggregation']=upstream_count(.99,1.29,obj)==(1 if config['synchronous'] else 2)
            checks['peer_transport_keeps_cache_contract']=upstream_count(1.29,1.55,obj)==(0 if config['cache'] else 1)
            checks['peer_transport_revalidates_expired_cache']=upstream_count(1.99,2.2,obj)==1
            checks['peer_transport_keeps_versions_distinct']=upstream_count(2.29,2.5,obj.replace('v1','v2'))==1
            checks['peer_transport_keeps_namespaces_distinct']=upstream_count(2.59,2.8,'/tenantB/slice2/map/v1/chunk0')==1
        if config['wireless']:
            checks['actual_phy_airtime_clocks_agree']=summary['radio_accounting']['independent_phy_clocks_agree']
            checks['actual_radio_transmissions_present']=summary['radio_accounting']['phy_transmission_events']>0
        summary.update(case=name,checks=checks,passed=all(checks.values()),
                       files={p.name:sha(p) for p in sorted(prefix.parent.glob(prefix.name+'-*'))})
        return summary
    if config['wireless']:
        checks['actual_phy_airtime_clocks_agree']=summary['radio_accounting']['independent_phy_clocks_agree']
        checks['actual_radio_transmissions_present']=summary['radio_accounting']['phy_transmission_events']>0
    if config.get('filterTest'):
        checks['negative_peer_packet_rejected']=any(r['event']=='peer_filtered' for r in rows) and not any(r['event']=='in_interest' and r['name']=='/probe/filter' for r in rows)
        accepted=[(r['node'],r['face']) for r in rows if r['event']=='peer_rx']
        checks['each_native_frame_injected_once']=len(accepted)==len(set(accepted))
    checks["all_real_devices_shared_with_ipv4"] = summary["shared_devices"] == (5 if config["wireless"] else 6)
    checks["ipv4_delivered_on_shared_devices"] = summary["ip_received"] > 0
    checks["native_ndn_data_delivered"] = summary["ndn_received"] > 0
    checks["unknown_prefix_not_delivered"] = not any(r["event"] == "app_data" and r["name"].startswith("/unknown") for r in rows)
    checks["identity_uses_network_queries"] = summary["id_queries"] > 1 and summary["id_registrations"] >= 1
    checks["old_mapping_update_rejected"] = summary["id_stale_registrations"] == 1
    checks["only_one_stable_eid_used"] = len({r["name"] for r in rows if r["event"] == "id_decap"}) == 1
    checks["three_protocol_mechanisms_concurrent"] = all(summary[k] > 0 for k in ("ndn_received", "ip_received", "id_received"))
    if config["speed"]:
        checks["trajectory_executed"] = abs(summary["consumer_b_final_x"] - (30 + 9 * config["speed"])) < 1e-6
    if config["speed"] == 80:
        checks["coverage_loss_not_hidden"] = summary["ndn_received"] < 6 and summary["ip_received"] < summary["ip_sent"]
    else:
        checks["all_six_valid_ndn_tasks_delivered"] = summary["ndn_received"] == (8 if config.get("pin") else 6)
        checks["identity_locator_updated"] = summary["id_final_mapping_version"] == 2
        checks["new_locator_delivered_data"] = any(r["event"] == "id_decap" and r["node"] == "3" for r in rows)
        checks["stale_locator_data_counted"] = summary["id_stale_data"] > 0

    def upstream(a, b, object_name):
        return [r for r in rows if r["node"] == "1" and r["face"] == "257"
                and r["event"] == "out_interest" and a <= float(r["time"]) < b and r["name"] == object_name]
    obj = "/tenantA/slice1/map/v1/chunk0"
    if not config["wireless"]:
        first = upstream(.99, 1.29, obj)
        checks["pit_aggregation_matches_declared_synchrony"] = len(first) == (1 if config["synchronous"] else 2)
        checks["cache_hit_does_not_use_upstream"] = len(upstream(1.29, 1.55, obj)) == (0 if config["cache"] and not config.get("pinControl") else 1)
        checks["stale_cache_goes_upstream"] = len(upstream(1.99, 2.2, obj)) == 1
        checks["new_version_not_old_cache_hit"] = len(upstream(2.29, 2.5, obj.replace("v1", "v2"))) == 1
        checks["other_namespace_not_prior_cache_hit"] = len(upstream(2.59, 2.8, "/tenantB/slice2/map/v1/chunk0")) == 1
        if config["synchronous"]:
            # Reconstruct each directed link's NETWORK-layer bytes from native
            # NFD traces and compare with the independent contract kernel.
            roles = {(0, 257): "o_r", (1, 257): "r_o", (1, 258): "r_a",
                     (2, 257): "a_r", (1, 259): "r_b", (3, 257): "b_r"}
            actual = {}
            data_header = interest_size = None
            for r in rows:
                edge = roles.get((int(r["node"]), int(r["face"])))
                if (edge and r["name"] == obj and .99 <= float(r["time"]) < 1.15
                        and r["event"] in ("out_data", "out_interest")):
                    key = "bytes:" + edge
                    actual[key] = actual.get(key, 0.) + int(r["bytes"])
                    if r["event"] == "out_data":
                        data_header = int(r["bytes"]) - 1000
                    else:
                        interest_size = int(r["bytes"])
            links = {}
            for a, b in (("o", "r"), ("r", "a"), ("r", "b")):
                for s, t in ((a, b), (b, a)):
                    links[f"{s}_{t}"] = Link(f"{s}_{t}", s, t, f"{t}_{s}", 1e7)
            expected = object_reservation(links=links, paths={"a": ("o_r", "r_a"), "b": ("o_r", "r_b")},
                tenant="tenantA", slice_id="slice1", name="map", version="v1", payload_bytes=1000,
                start=0., end=1., modality="ndn", synchronous=True,
                data_header_bytes=data_header, interest_bytes=interest_size)
            predicted_events={k:v for k,v in expected.items() if k.startswith("tx_events:")}
            actual_events={}
            labels={0:'o',1:'r',2:'a',3:'b'}
            for row in rows:
                if ((int(row['node']),int(row['face'])) in roles and row['name']==obj
                    and .99<=float(row['time'])<1.15 and row['event'] in ('out_data','out_interest')):
                    key='tx_events:'+labels[int(row['node'])]
                    actual_events[key]=actual_events.get(key,0)+1
            expected = {k: v for k, v in expected.items() if k.startswith("bytes:")}
            checks["native_per_link_network_bytes_match_contract"] = actual == expected
            checks['native_logical_tx_events_match_contract']=actual_events==predicted_events
            summary["network_layer_ledger"] = {"actual": actual, "expected": expected,
                    'actual_logical_tx_events':actual_events,'expected_logical_tx_events':predicted_events}
        snapshots=[r for r in rows if r['event']=='cs_payload']
        checks['cache_payload_measurements_present']=len(snapshots)==7
        checks['cache_slots_enforced']=all(float(r['detail'])<= (1 if config.get('pin') else (64 if config['cache'] else 0)) for r in snapshots)
        if config.get('pin'):
            def present(at):return any(abs(float(r['time'])-at)<1e-6 and r['event']=='cs_entry' and r['name']==obj for r in rows)
            checks['pin_survives_pressure_or_control_evicts']=present(1.25)==(not config.get('pinControl',False))
            checks['pin_released_after_interval']=not present(1.75)
            checks['pinned_payload_bytes_match_declared_lease']=(any(float(r['time'])==1.25 and int(r['bytes'])==1000 for r in snapshots))
            if not config.get('pinControl'):
                lease=CacheLease('tenantA','slice1','relay','map','v1',1000,1.05,1.5,True)
                book=CacheLedger()
                from ..protocol_contracts import Reservation
                book.commit('object-owner',Reservation({},(),(lease,)),{'relay':1000})
                actual={float(r['time']):int(r['bytes']) for r in snapshots if 1.05<=float(r['time'])<1.5}
                checks['native_pin_payload_matches_object_ledger']=all(value==book.usage(at)['relay'] for at,value in actual.items()) and bool(actual)
                checks['pinned_entry_fresh_for_observed_lease_window']=all(float(r['detail'])==1 for r in rows if r['event']=='cs_entry' and r['name']==obj and 1.05<=float(r['time'])<1.5)
                summary['pinned_storage_ledger']={'actual_payload_by_time':actual,'expected_payload_bytes':1000,'lease_start':1.05,'lease_end':1.5,'scope':'single object, payload only; CS metadata/allocator memory uncalibrated'}
            summary['pin_scope']='one preloaded object, one-entry CS, payload-byte quota1000, lease[1.05,1.5); not full tenant isolation'

    summary.update(case=name, checks=checks, passed=all(checks.values()),
                   files={p.name: sha(p) for p in sorted(prefix.parent.glob(prefix.name + "-*"))})
    return summary


def main():
    platform = build()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(NS3 / "build/lib")
    results = []
    for case, config in {**CASES,**EQUIVALENT_CASES,**CONTROLLED_CASES}.items():
        for seed in (CONTROLLED_SEEDS if case in CONTROLLED_CASES else (11,12,13)):
            prefix = OUT / f"{case}_{seed}"
            command = [NS3 / "build/scratch/protocol-probe", f"--output={prefix}", f"--seed={seed}"]
            command += [f"--{key}={value}" for key, value in config.items()]
            run(command, env=env, log=Path(str(prefix) + "-process.log"))
            result = analyze(prefix, case, config)
            result["command"] = [str(c) for c in command]
            results.append(result)
            bad = [key for key, value in result["checks"].items() if not value]
            print(json.dumps(dict(case=case, seed=seed, passed=result["passed"],
                ndn=result["ndn_received"], ip=f"{result['ip_received']}/{result['ip_sent']}",
                identity=f"{result['id_received']}/{result['id_sent']}", failed=bad)), flush=True)
    equivalent=compare_equivalent(results)
    controlled=compare_controlled(results)
    if not isinstance(controlled['controls'], str) or not controlled['controls'].startswith('common PHY:'):
        raise ValueError('controlled comparison requires one complete controls description string')
    development=OUT/'development-verification.json'
    proof = dict(passed=all(r["passed"] for r in results) and equivalent["passed"] and controlled["passed"],
        scope="native protocol integration and L2-controlled task comparisons, NOT full slice/RL performance evaluation",
        packet_simulation_performed=True, training_performed=False,
        cases=len(results), assertions=sum(len(r["checks"]) for r in results)+sum(len(p["checks"]) for p in controlled["pairs"]),
        verification_driver_sha256=sha(__file__),contract_kernel_sha256=sha(ROOT/'slice_alloc/protocol_contracts.py'),
        confirmation_seeds=[11,12,13],development_seeds=[1,2,3],
        retained_development_failure=dict(file=development.name,sha256=sha(development)) if development.exists() else None,
        platform=platform, results=results,equivalent_tasks=equivalent,link_layer_comparison=controlled,
        remaining=["general per-slice FIB/PIT/CS and CPU/airtime quota enforcement (single-object native pin proof only)",
                   "L2-controlled single-chunk comparison is implemented; independent conservative service-envelope calibration remains open",
                   "cache ledger tracks payload bytes; allocator/metadata memory and general lease installation still require calibration",
                   "expression checks only stipulate desired ranks; no RL expressivity/optimality superiority is established",
                   "full initial slice mapper and mobility envelope installation",
                   "new-model RL training and independent performance evaluation"],
        caution="Logical sends, network bytes, PHY TX duration and payload storage are distinct. PHY empirical observations are not a schedulable-airtime guarantee. Current PCAP includes radiotap; actual PHY modes and clocks are recorded natively. implied_rate_mbps is a derived MPDU-byte/TX-work ratio, not a measured PHY bitrate or capacity. Sum of transmitter TX work and channel interval union are separate metrics.")
    (OUT / "verification.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n")
    if not proof["passed"]:
        raise SystemExit("Native protocol verification failed; see every retained case, not just averages")
    print(f"Verified {proof['cases']} native packet cases and {proof['assertions']} assertions")


if __name__ == "__main__":
    main()
