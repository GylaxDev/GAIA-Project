"""Citizen NPC meshes: asset integrity, rig data, slot selection and the fallback path."""
import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG.parent))
ha = importlib.import_module(f'{PKG.name}.peds.human_assets')
hf = importlib.import_module(f'{PKG.name}.peds.human_faces')

PROP_BITS = {1, 2, 4, 8, 2048, 262144, 4194304}


class CitizenAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ha.citizens_available():
            raise unittest.SkipTest('human_citizens.npz not baked')
        cls.meshes = [ha.slot_template(0, k) for k in range(ha.slot_count() - 1)]
        cls.meta = ha.citizen_metadata()

    def test_identities_match_face_groups(self):
        self.assertEqual(len(self.meshes), len(hf.CITIZEN_MALE) + len(hf.CITIZEN_FEMALE))
        self.assertEqual([r['variant'] for r in self.meta['identities']],
                         ['male'] * len(hf.CITIZEN_MALE) + ['female'] * len(hf.CITIZEN_FEMALE))

    def test_geometry_is_valid_and_light(self):
        for k, (v, i) in enumerate(self.meshes):
            body = v[v[:, 7] == 0]
            self.assertTrue(np.isfinite(v).all(), k)
            self.assertLess(int(i.max()), len(v), k)
            self.assertEqual(i.size % 3, 0)
            self.assertLess(i.size // 3, 8000, f'slot {k} is no longer a low-poly mesh')
            self.assertTrue((body[:, 6] == ha.M_CITIZEN).all(), k)

    def test_standing_on_ground_facing_plus_x(self):
        for k, (v, _) in enumerate(self.meshes):
            body = v[(v[:, 7] == 0) & (v[:, 6] == ha.M_CITIZEN)]
            p = body[:, :3]
            self.assertAlmostEqual(float(p[:, 2].min()), 0., delta=.01, msg=k)
            self.assertAlmostEqual(float(p[:, 2].max()), 1.78, delta=.01, msg=k)
            toes = p[p[:, 2] < .06, 0].mean() - p[(p[:, 2] > .3) & (p[:, 2] < .55), 0].mean()
            self.assertGreater(toes, .02, f'slot {k} does not face +x')
            # brow/nose stand well ahead of the neck; hair mass behind the head can skew centroids
            front = p[p[:, 2] > 1.60, 0].max() - p[(p[:, 2] > 1.45) & (p[:, 2] < 1.52), 0].mean()
            back = p[(p[:, 2] > 1.45) & (p[:, 2] < 1.52), 0].mean() - p[p[:, 2] > 1.60, 0].min()
            self.assertGreater(front, .05, f'slot {k} head does not face +x')
            self.assertGreater(front, back - .02, f'slot {k} head faces backwards')

    def test_arms_hang_beside_the_body(self):
        for k, (v, _) in enumerate(self.meshes):
            p = v[(v[:, 7] == 0), :3]
            hands = p[(p[:, 2] > .55) & (p[:, 2] < 1.0)]
            self.assertLess(float(np.abs(hands[:, 1]).max()), .42, f'slot {k} arms still held out')

    def test_hands_are_relaxed(self):
        """The pack's fanned fingers are gathered (tools/relax_citizen_hands.py); a fan reads as 'reaching' in the gait."""
        self.assertTrue(self.meta.get('hands_relaxed'), 'run tools/relax_citizen_hands.py (or rebake)')
        spans = []
        for k, ident in enumerate(self.meta['identities']):
            v = self.meshes[k][0][:ident['vertices']]
            dominant = v[np.arange(len(v)), np.argmax(v[:, 12:16], axis=1) + 8].astype(int)
            skin = v[:, 19].astype(int) % 2 == 0
            for joint in (4, 5):                                   # left / right forearm joint
                arm = (dominant == joint) & skin & (v[:, 2] < .96)
                tip = v[arm, 2].min()
                spans.append(float(np.ptp(v[arm & (v[:, 2] < tip + .08), 0])))
        # measured: 0.120 m on average for the original fan, 0.075 m relaxed
        self.assertLess(float(np.mean(spans)), .095)

    def test_skin_weights(self):
        for k, (v, _) in enumerate(self.meshes):
            joints, weights = v[:, 8:12], v[:, 12:16]
            self.assertTrue(((joints >= 0) & (joints <= 14)).all(), k)
            self.assertTrue((weights >= 0).all(), k)
            self.assertTrue(np.allclose(weights.sum(axis=1), 1., atol=1e-4), k)
            head = v[(v[:, 6] == ha.M_CITIZEN) & (v[:, 2] > 1.65)]
            dominant = head[np.arange(len(head)), np.argmax(head[:, 12:16], axis=1) + 8].astype(int)
            self.assertGreater((dominant == 12).mean(), .9, f'slot {k} head is not skinned to the head joint')

    def test_texture_layers(self):
        textures = ha.citizen_textures()
        self.assertEqual(textures.shape[0], 2 * len(self.meshes))
        self.assertEqual(textures.dtype, np.uint8)
        for k, (v, _) in enumerate(self.meshes):
            layers = np.unique(v[v[:, 6] == ha.M_CITIZEN][:, 19]).astype(int)
            self.assertEqual(set(layers), {2 * k, 2 * k + 1}, k)
            uv = v[v[:, 6] == ha.M_CITIZEN][:, 17:19]
            self.assertTrue(((uv > -.01) & (uv < 1.01)).mean() > .99, k)
        self.assertGreater(float(textures[..., :3].mean()), 20.)

    def test_props_are_carried_over(self):
        for k, (v, _) in enumerate(self.meshes):
            acc = set(v[v[:, 7] != 0][:, 7].astype(int).tolist())
            self.assertEqual(acc, PROP_BITS, k)
            self.assertTrue((v[v[:, 7] != 0][:, 23] == -2).all())


class SlotTests(unittest.TestCase):
    def test_slots_are_stable_and_in_range(self):
        n = ha.slot_count()
        variants = np.array([ha.MALE, ha.FEMALE, ha.PLAYER] * 100)
        seeds = np.tile(np.linspace(0, 1, 100), 3)
        a, b = ha.mesh_slots(variants, seeds), ha.mesh_slots(variants, seeds)
        self.assertTrue((a == b).all())
        self.assertTrue(((a >= 0) & (a < n)).all())
        if ha.citizens_available():
            m, f = len(hf.CITIZEN_MALE), len(hf.CITIZEN_FEMALE)
            self.assertTrue((a[variants == ha.MALE] < m).all())
            self.assertTrue(((a[variants == ha.FEMALE] >= m) & (a[variants == ha.FEMALE] < m + f)).all())
            self.assertTrue((a[variants == ha.PLAYER] == m + f).all())
            self.assertEqual(len(set(a[variants == ha.MALE])), m)      # every identity gets used
            self.assertEqual(len(set(a[variants == ha.FEMALE])), f)

    def test_fallback_without_citizen_asset(self):
        with mock.patch.object(ha, 'CITIZEN_PATH', PKG / 'assets/humans/_missing_.npz'):
            ha._citizens.cache_clear()
            try:
                self.assertFalse(ha.citizens_available())
                self.assertEqual(ha.slot_count(), 3)
                self.assertEqual(list(ha.mesh_slots([0, 1, 2], [.1, .2, .3])), [0, 1, 2])
                self.assertEqual(ha.slot_template(0, ha.FEMALE)[0].shape, ha.human_template(0, ha.FEMALE)[0].shape)
                self.assertEqual(ha.runtime_face_textures().shape, ha.render_face_textures().shape)
            finally:
                ha._citizens.cache_clear()

    def test_player_keeps_detailed_mesh(self):
        v, _ = ha.slot_template(0, ha.slot_count() - 1)
        self.assertGreater(len(v), 50000)
        self.assertFalse((v[:, 6] == ha.M_CITIZEN).any())


if __name__ == '__main__':
    unittest.main()
