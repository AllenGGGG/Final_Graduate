"""Hand-derived protocol accounting and mobility admission counterexamples."""
from dataclasses import replace
import unittest
from slice_alloc.protocol_contracts import Link, CacheLease, CacheLedger, object_reservation, MobilityScenario, MobilityEnvelope
from slice_alloc.tools.design_checks import expression_diagnostics


def network():
    links = {}
    for a, b, domain in (("origin", "relay", None), ("relay", "a", "ag"), ("relay", "b", "ag")):
        for source, target in ((a, b), (b, a)):
            key = f"{source}>{target}"
            links[key] = Link(key, source, target, f"{target}>{source}", 1e6, domain)
    return links


def reserve(**kw):
    args = dict(links=network(), paths={"a": ("origin>relay", "relay>a"),
                                      "b": ("origin>relay", "relay>b")},
                tenant="tenant", slice_id="slice", name="map", version="v1",
                payload_bytes=1000, start=0., end=10., modality="ndn",
                synchronous=True, data_header_bytes=0, interest_bytes=0)
    args.update(kw)
    return object_reservation(**args)


def lease(**kw):
    return replace(CacheLease("tenant", "slice", "relay", "map", "v1", 1000, 0., 10., True), **kw)


def scenarios():
    return (
        MobilityScenario("at_a", frozenset({"ap_a"}), frozenset({"wire_a"}),
                         {"cpu:edge": 2., "radio:ag": .4, "link:a": 4.}),
        MobilityScenario("at_b", frozenset({"ap_b"}), frozenset({"wire_b"}),
                         {"cpu:edge": 2., "radio:ag": .6, "link:b": 6.}))


class ProtocolContractTests(unittest.TestCase):
    def test_sync_tree_saves_only_shared_upstream(self):
        ndn, ip = reserve(), reserve(modality="ip_unicast")
        self.assertEqual(ndn["bytes:origin>relay"], 1000)
        self.assertEqual(ip["bytes:origin>relay"], 2000)
        for edge in ("relay>a", "relay>b"):
            self.assertEqual(ndn[f"bytes:{edge}"], ip[f"bytes:{edge}"])

    def test_asynchronous_cold_requests_do_not_get_aggregation_discount(self):
        self.assertEqual(reserve(synchronous=False)["bytes:origin>relay"], 2000)

    def test_pinned_cache_truncates_only_upstream_and_keeps_consumers(self):
        result = reserve(leases=(lease(),))
        self.assertNotIn("bytes:origin>relay", result)
        self.assertEqual(result["bytes:relay>a"], 1000)
        self.assertEqual(result["bytes:relay>b"], 1000)
        self.assertNotIn("new_pinned_cache_bytes:relay", result)
        self.assertEqual(len(result.required_leases), 1)
        self.assertEqual(result.required_leases[0].bytes, 1000)

    def test_invalid_or_unowned_cache_cannot_reduce_reservation(self):
        for cache in (lease(pinned=False), lease(valid_until=9.), lease(valid_from=1.),
                      lease(tenant="other"), lease(slice_id="other"), lease(version="v2"),
                      lease(bytes=999), lease(name="different")):
            with self.subTest(cache=cache):
                self.assertEqual(reserve(leases=(cache,))["bytes:origin>relay"], 1000)

    def test_headers_interest_and_shared_radio_hops_are_counted(self):
        result = reserve(payload_bytes=1500, data_header_bytes=32, interest_bytes=64)
        self.assertEqual(result["bytes:origin>relay"], 1564)
        self.assertEqual(result["bytes:relay>origin"], 128)
        # Both last hops and BOTH directions use the same competition domain.
        self.assertAlmostEqual(result["serialization_lower_bound_seconds:ag"], 2 * (1564 + 128) * 8 / 1e6)

    def test_ip_does_not_silently_use_ndn_cache(self):
        self.assertEqual(reserve(modality="ip_unicast", leases=(lease(),))["bytes:origin>relay"], 2000)

    def test_wrong_or_discontinuous_path_is_rejected(self):
        for path in (("origin>relay", "relay>origin"), ("relay>a", "origin>relay"), ("missing",)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                reserve(paths={"a": path})

    def test_aggregation_cannot_discount_distinct_object_contracts(self):
        first = reserve(version="v1")
        second = reserve(version="v2")
        self.assertEqual(first["bytes:origin>relay"] + second["bytes:origin>relay"], 2000)

    def test_reverse_path_and_positive_link_service_are_required(self):
        links = network()
        del links["relay>origin"]
        with self.assertRaises(ValueError):
            reserve(links=links)
        links = network()
        links["relay>a"] = replace(links["relay>a"], service_bps=float("nan"))
        with self.assertRaises(ValueError):
            reserve(links=links)

    def test_reconverging_routes_cannot_be_labeled_a_single_tree(self):
        links = network()
        for a, b in (("origin", "x"), ("x", "relay")):
            for source, target in ((a, b), (b, a)):
                key = f"{source}>{target}"
                links[key] = Link(key, source, target, f"{target}>{source}", 1e6)
        with self.assertRaises(ValueError):
            reserve(links=links, paths={"a": ("origin>relay", "relay>a"),
                                       "b": ("origin>x", "x>relay", "relay>b")})


class MobilityContractTests(unittest.TestCase):
    def test_mutually_exclusive_scenarios_take_resourcewise_max(self):
        reserved = MobilityEnvelope(scenarios()).reservation()
        self.assertEqual(reserved, {"cpu:edge": 2., "radio:ag": .6, "link:a": 4., "link:b": 6.})

    def test_handover_within_envelope_is_permitted(self):
        env = MobilityEnvelope(scenarios())
        self.assertTrue(env.permits({"ap_b"}, {"wire_b"}, {"radio:ag": .5}))

    def test_outside_envelope_and_phantom_resources_are_rejected(self):
        env = MobilityEnvelope(scenarios())
        self.assertFalse(env.permits({"ap_c"}, {"wire_c"}, {}))
        self.assertFalse(env.permits({"ap_b"}, {"wire_b"}, {"radio:ag": .7}))
        self.assertFalse(env.permits({"ap_b"}, {"wire_b"}, {"cpu:unknown": 1.}))

    def test_dual_attachment_needs_explicit_concurrent_scenario(self):
        env = MobilityEnvelope(scenarios())
        self.assertFalse(env.permits({"ap_a", "ap_b"}, {"wire_a", "wire_b"}, {"radio:ag": .5}))
        both = MobilityScenario("both", frozenset({"ap_a", "ap_b"}), frozenset({"wire_a", "wire_b"}),
                                {"cpu:edge": 2., "radio:ag": 1., "link:a": 4., "link:b": 6.})
        self.assertEqual(MobilityEnvelope(scenarios() + (both,)).reservation()["radio:ag"], 1.)

    def test_existing_other_slices_still_consume_capacity(self):
        env = MobilityEnvelope(scenarios())
        caps = {"cpu:edge": 2., "radio:ag": .8, "link:a": 4., "link:b": 6.}
        self.assertTrue(env.check_capacity(caps))
        with self.assertRaises(ValueError):
            env.check_capacity(caps, {"radio:ag": .3})

    def test_bad_or_duplicate_scenarios_are_rejected(self):
        a = scenarios()[0]
        for entries in ((), (a, a), (replace(a, usage={"radio:ag": -1.}),)):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                MobilityEnvelope(entries)

    def test_installed_contract_does_not_alias_mutable_builder_inputs(self):
        usage, nodes, routes = {"radio:ag": .4}, {"ap_a"}, {"wire_a"}
        scenario = MobilityScenario("at_a", nodes, routes, usage)
        source = [scenario]
        envelope = MobilityEnvelope(source)
        usage["radio:ag"] = .9
        nodes.add("ap_c")
        routes.add("wire_c")
        source.clear()
        self.assertEqual(envelope.reservation(), {"radio:ag": .4})
        self.assertFalse(envelope.permits({"ap_c"}, {"wire_c"}, {}))


class ExpressionTests(unittest.TestCase):
    def test_linear_vector_limitation_does_not_invalidate_barrier_baseline(self):
        proof = expression_diagnostics()
        self.assertAlmostEqual(proof["linear_vector_price_best_margin"], -.1)
        self.assertEqual(proof["synthetic_barrier_choice"], "C")
        self.assertLess(proof["barrier_penalties"]["C"], proof["barrier_penalties"]["A"])

    def test_fixed_scalar_two_regimes_have_disjoint_ranking_intervals(self):
        proof = expression_diagnostics()["fixed_scalar_two_regimes"]
        self.assertTrue(proof["intersection_empty"])
        self.assertNotEqual(*proof["observable_occupancies"])
        self.assertTrue(all(proof["synthetic_barrier_threshold_checks"]))
        self.assertTrue(proof["desired_rankings_are_assumptions_not_proven_optima"])


class CacheOwnershipTests(unittest.TestCase):
    def test_existing_object_is_not_billed_again_for_each_transfer(self):
        ledger = CacheLedger()
        created = reserve(pin_requests=(lease(),))
        self.assertEqual(created["new_pinned_cache_bytes:relay"], 1000)
        self.assertEqual(len(created.new_leases), 1)
        ledger.commit("creator", created, {"relay":1000})
        for i in range(2):
            plan = reserve(leases=ledger.leases)
            self.assertNotIn("new_pinned_cache_bytes:relay", plan)
            ledger.commit(str(i), plan, {"relay":1000})
        self.assertEqual(ledger.usage(1.), {"relay":1000})
        ledger.release("creator")
        self.assertEqual(ledger.usage(1.), {"relay":1000})
        ledger.release("0"); ledger.release("1")
        self.assertEqual(ledger.usage(1.), {})

    def test_versions_tenants_and_slices_do_not_share_storage(self):
        for changed in (lease(version="v2"),lease(tenant="other"),lease(slice_id="other")):
            ledger=CacheLedger();ledger.commit("a",reserve(pin_requests=(lease(),)),{"relay":1000})
            extra=reserve(tenant=changed.tenant,slice_id=changed.slice_id,version=changed.version,
                          pin_requests=(changed,))
            with self.assertRaises(ValueError): ledger.commit("b",extra,{"relay":1000})
            self.assertEqual(ledger.usage(1.),{"relay":1000})

    def test_oversized_lease_is_charged_at_full_size_once(self):
        plan=reserve(pin_requests=(lease(bytes=2000),))
        self.assertEqual(plan["new_pinned_cache_bytes:relay"],2000)
        ledger=CacheLedger()
        with self.assertRaises(ValueError):ledger.commit("a",plan,{"relay":1000})
        ledger.commit("a",plan,{"relay":2000})
        self.assertEqual(ledger.usage(1.),{"relay":2000})

    def test_disjoint_objects_reuse_capacity_but_overlapping_versions_do_not(self):
        ledger=CacheLedger();ledger.commit("a",reserve(pin_requests=(lease(),)),{"relay":1000})
        later=lease(version="v2",valid_from=10.,valid_until=20.)
        ledger.commit("b",reserve(start=10.,end=20.,version="v2",pin_requests=(later,)),{"relay":1000})
        self.assertEqual(ledger.usage(10.),{"relay":1000})
        self.assertEqual(ledger.usage(20.),{})

    def test_stale_dependency_is_rejected_at_atomic_commit(self):
        ledger=CacheLedger();created=reserve(pin_requests=(lease(),))
        ledger.commit("a",created,{"relay":1000})
        stale=reserve(leases=ledger.leases)
        ledger.release("a")
        with self.assertRaises(ValueError):ledger.commit("b",stale,{"relay":1000})

    def test_conflicting_object_sizes_are_rejected(self):
        ledger=CacheLedger();ledger.commit("a",reserve(pin_requests=(lease(),)),{"relay":3000})
        with self.assertRaises(ValueError):
            ledger.commit("b",reserve(pin_requests=(lease(bytes=2000),)),{"relay":3000})

    def test_cache_at_producer_is_explicit_dependency(self):
        plan=reserve(leases=(lease(node="origin"),))
        self.assertEqual(plan.required_leases[0].node,"origin")
        self.assertIn("bytes:origin>relay",plan)

    def test_invalid_lease_never_silently_enters_ledger(self):
        for bad in (lease(bytes=0),lease(valid_from=-1),lease(valid_until=0),lease(tenant="")):
            with self.subTest(bad=bad),self.assertRaises(ValueError):reserve(pin_requests=(bad,))


class AirtimeTraceTests(unittest.TestCase):
    def test_mpdu_aggregation_is_counted_once_per_radio_not_once_per_packet(self):
        from slice_alloc.tools.platform_verify import radio_accounting
        def row(t,node,event,uid,detail=0):
            return dict(time=str(t),node=str(node),event=event,face=str(uid),bytes='100',detail=str(detail))
        rows=[row(1,1,'phy_begin',11),row(1,1,'phy_begin',12),
              row(1,1,'phy_state_tx',0,.1),row(1,2,'phy_begin',13),
              row(1,2,'phy_state_tx',0,.1),row(1.1,1,'phy_end',11),
              row(1.1,1,'phy_end',12),row(1.1,2,'phy_end',13)]
        actual=radio_accounting(rows)
        self.assertEqual(actual['phy_mpdu_events'],3)
        self.assertEqual(actual['phy_transmission_events'],2)
        self.assertAlmostEqual(actual['tx_seconds_from_begin_end'],.2)
        self.assertAlmostEqual(actual['channel_union_tx_seconds'],.1)
        self.assertTrue(actual['independent_phy_clocks_agree'])

    def test_incomplete_transmissions_are_not_silently_censored(self):
        from slice_alloc.tools.platform_verify import radio_accounting
        with self.assertRaises(ValueError):radio_accounting([dict(time='1',node='1',event='phy_begin',face='1',bytes='100',detail='0')])


class AdditionalCacheBoundaryTests(unittest.TestCase):
    def test_lease_extension_charges_new_interval_after_original_expires(self):
        original=lease(valid_until=5.)
        extension=lease(valid_from=4.,valid_until=10.)
        plan=reserve(start=4.,end=10.,leases=(original,),pin_requests=(extension,))
        self.assertEqual(plan['new_pinned_cache_bytes:relay'],1000)

    def test_negative_or_nonfinite_cache_capacity_is_rejected(self):
        for capacity in (-1.,float('nan')):
            with self.subTest(capacity=capacity),self.assertRaises(ValueError):
                CacheLedger().commit('a',reserve(pin_requests=(lease(),)),{'relay':capacity})


class LinkLayerEvidenceTests(unittest.TestCase):
    def frame(self, **changes):
        row=dict(frame_type='data',ethertype='30583',destination='broadcast',bytes='100',
                 mode='DsssRate1Mbps',phy_bps='1000000',nss='1',gi_ns='800',width_mhz='20',
                 preamble='1',retry='0',aggregate='0',qos_tid='0')
        row.update(changes)
        return row

    def test_ack_and_arp_do_not_turn_broadcast_business_into_mixed_mode(self):
        from slice_alloc.tools.platform_verify import frame_accounting
        evidence=frame_accounting([self.frame(),self.frame(frame_type='control',destination='unicast',ethertype='0',bytes='14'),
                                  self.frame(ethertype='2054',destination='broadcast',bytes='64')])
        self.assertEqual(evidence['l2_delivery_mode'],'broadcast')
        self.assertEqual(evidence['broadcast_frame_bytes'],100)
        self.assertEqual(evidence['control_frame_bytes'],14)
        self.assertEqual(evidence['arp_frame_bytes'],64)

    def test_actual_phy_mode_is_preserved_instead_of_inferred_from_bytes(self):
        from slice_alloc.tools.platform_verify import frame_accounting
        row=self.frame(destination='unicast',mode='ErpOfdmRate6Mbps',phy_bps='6000000')
        evidence=frame_accounting([row])
        self.assertIn('ErpOfdmRate6Mbps|6000000|1|800|20|1',evidence['actual_phy_mode_histogram'])
        self.assertNotIn('implied_rate_mbps',evidence)

    def test_multicast_and_mac_retries_are_separate_from_broadcast(self):
        from slice_alloc.tools.platform_verify import frame_accounting
        evidence=frame_accounting([self.frame(destination='multicast',retry='1'),self.frame()])
        self.assertEqual(evidence['l2_delivery_mode'],'mixed')
        self.assertEqual(evidence['multicast_frame_bytes'],100)
        self.assertEqual(evidence['business_mac_retries'],1)

    def test_exported_byte_rate_uses_tx_work_not_channel_union_or_phy_bitrate(self):
        from slice_alloc.tools.platform_verify import export_radio_evidence
        import json
        evidence=export_radio_evidence(dict(phy_frame_bytes=1000,tx_seconds_from_state=.002,
                                           channel_union_tx_seconds=.001),[self.frame()])
        restored=json.loads(json.dumps(evidence))
        self.assertEqual(restored['l2_delivery_mode'],'broadcast')
        self.assertEqual(restored['actual_phy_mode_histogram'],restored['link_layer']['actual_phy_mode_histogram'])
        self.assertEqual(restored['implied_rate_mbps'],4.)
        self.assertIn('DsssRate1Mbps|1000000|1|800|20|1',restored['actual_phy_mode_histogram'])
        self.assertIn('NOT the channel-interval union',restored['implied_rate_scope'])

    def test_no_transmissions_do_not_produce_a_phantom_rate(self):
        from slice_alloc.tools.platform_verify import export_radio_evidence
        evidence=export_radio_evidence(dict(phy_frame_bytes=0,tx_seconds_from_state=0.,
                                           channel_union_tx_seconds=0.),[])
        self.assertIsNone(evidence['implied_rate_mbps'])
        self.assertEqual(evidence['l2_delivery_mode'],'none')
        self.assertEqual(evidence['actual_phy_mode_histogram'],{})
