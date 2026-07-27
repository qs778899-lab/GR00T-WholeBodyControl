"""Executable invariants for the documentation-level translation mapping."""

from __future__ import annotations

import json
import math
from pathlib import Path
import unittest


FIXTURE = Path(__file__).parent / "fixtures/translation_v1_two_site.json"
IR_KEYS = {
    "schema_version",
    "dtype",
    "device",
    "units",
    "site_names",
    "site_link_offset",
    "link_offset_semantics",
    "force_semantics",
    "common_frame",
    "future_time_offset_s",
    "motion_id",
    "reference_frame_index",
    "reference_timestamp_s",
    "sample_timestamp_s",
    "reference_target",
    "current_site",
    "site_mask",
    "compliance",
    "force_on_robot",
    "enable",
}


def _finite_tree(value) -> bool:
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def _shape_4d_xyz(value, *, batch: int, future: int, sites: int) -> bool:
    return (
        len(value) == batch
        and all(len(item) == future for item in value)
        and all(len(frame) == sites for item in value for frame in item)
        and all(
            len(xyz) == 3
            for item in value
            for frame in item
            for xyz in frame
        )
    )


class TranslationV1GoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.ir = cls.fixture["ir"]
        cls.expected = cls.fixture["expected"]

    def test_exact_mapping_fixture_and_ir_schema(self):
        self.assertEqual(
            self.fixture["fixture_schema_version"],
            "compliance.translation.mapping-fixture.v1",
        )
        self.assertEqual(set(self.fixture), {"fixture_schema_version", "ir", "expected"})
        self.assertEqual(set(self.ir), IR_KEYS)
        self.assertEqual(self.ir["schema_version"], "compliance.translation.v1")
        self.assertEqual(self.ir["dtype"], "float64")
        self.assertEqual(self.ir["device"], "cpu")

    def test_names_frame_units_and_time_identity_are_explicit(self):
        ir = self.ir
        self.assertEqual(len(ir["site_names"]), len(set(ir["site_names"])))
        self.assertEqual(len(ir["site_link_offset"]), len(ir["site_names"]))
        self.assertTrue(all(len(offset) == 3 for offset in ir["site_link_offset"]))
        self.assertEqual(
            ir["link_offset_semantics"],
            "link_origin_to_application_point_in_link_frame",
        )
        self.assertEqual(ir["force_semantics"], "force_on_robot")
        self.assertEqual(
            ir["units"],
            {
                "site_link_offset": "m",
                "future_time_offset_s": "s",
                "reference_timestamp_s": "s",
                "sample_timestamp_s": "s",
                "reference_target": "m",
                "current_site": "m",
                "compliance": "m/N",
                "force_on_robot": "N",
            },
        )
        frame = ir["common_frame"]
        self.assertEqual(
            frame,
            {
                "kind": "anchor_local",
                "anchor_name": "root",
                "rotation_rule": "world_to_current_anchor",
                "handedness": "right",
                "up_axis": "+z",
                "forward_axis": "+x",
                "lateral_axis": "+y",
                "quaternion_order": "wxyz",
            },
        )
        offsets = ir["future_time_offset_s"]
        sample_time = ir["sample_timestamp_s"][0]
        self.assertEqual(
            ir["reference_timestamp_s"][0],
            [sample_time + value for value in offsets],
        )
        frames = ir["reference_frame_index"][0]
        self.assertEqual(frames, list(range(frames[0], frames[0] + len(offsets))))

    def test_shapes_finiteness_and_binary_contract(self):
        ir = self.ir
        batch = len(ir["motion_id"])
        future = len(ir["future_time_offset_s"])
        sites = len(ir["site_names"])
        self.assertTrue(_finite_tree(self.fixture))
        self.assertTrue(_shape_4d_xyz(ir["reference_target"], batch=batch, future=future, sites=sites))
        self.assertEqual(len(ir["current_site"]), batch)
        self.assertTrue(all(len(row) == sites for row in ir["current_site"]))
        self.assertTrue(all(len(xyz) == 3 for row in ir["current_site"] for xyz in row))
        for name in ("compliance", "force_on_robot"):
            self.assertEqual(len(ir[name]), batch)
            self.assertTrue(all(len(row) == sites for row in ir[name]))
            self.assertTrue(all(len(xyz) == 3 for row in ir[name] for xyz in row))
        self.assertTrue(all(type(value) is bool for value in ir["enable"]))
        self.assertTrue(
            all(type(value) is bool for row in ir["site_mask"] for value in row)
        )
        self.assertEqual(len(ir["reference_frame_index"]), batch)
        self.assertTrue(all(len(row) == future for row in ir["reference_frame_index"]))

    def test_chip_and_motion_routes_map_active_sites_only(self):
        ir = self.ir
        reference = ir["reference_target"][0]
        force = ir["force_on_robot"][0]
        compliance = ir["compliance"][0]
        mask = ir["site_mask"][0]
        chip = self.expected["chip_hindsight"]["policy_target"][0]
        motion_route = self.expected["motion_reference_selection"]
        compliant = motion_route["compliant_reference_target"][0]
        motion = motion_route["policy_target"][0]
        for future_index in range(len(reference)):
            for site_index, active in enumerate(mask):
                if active:
                    expected_chip = [
                        reference[future_index][site_index][axis]
                        - compliance[site_index][axis] * force[site_index][axis]
                        for axis in range(3)
                    ]
                    self.assertEqual(chip[future_index][site_index], expected_chip)
                    self.assertEqual(
                        motion[future_index][site_index],
                        compliant[future_index][site_index],
                    )
                else:
                    self.assertEqual(chip[future_index][site_index], reference[future_index][site_index])
                    self.assertEqual(motion[future_index][site_index], reference[future_index][site_index])

    def test_structural_off_target_is_release_reference(self):
        self.assertEqual(self.expected["off_policy_target"], self.ir["reference_target"])


if __name__ == "__main__":
    unittest.main()
