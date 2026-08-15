import unittest

from rvq_asr.model import RVQTransformerCTC
from rvq_asr.train_probe import parse_active_layers


class LayerFusionTest(unittest.TestCase):
    def make_model(self, **kwargs):
        return RVQTransformerCTC(
            codebook_size=16,
            num_rvq_layers=8,
            vocabulary_size=5,
            max_rvq_layers=8,
            model_dim=8,
            num_encoder_layers=1,
            num_heads=2,
            feedforward_dim=16,
            time_reduction=1,
            **kwargs,
        )

    def test_fixed_mask_weights(self):
        model = self.make_model(active_rvq_layers=[0, 4, 5, 6, 7])
        self.assertEqual(
            model.normalized_layer_weights(),
            [0.2, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2],
        )

    def test_learned_weights_are_normalized(self):
        model = self.make_model(active_rvq_layers=[0, 1, 2, 3], layer_fusion="learned")
        weights = model.normalized_layer_weights()
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertEqual(weights[4:], [0.0, 0.0, 0.0, 0.0])

    def test_parse_active_layers(self):
        self.assertEqual(parse_active_layers("1,5,6,7,8"), [1, 5, 6, 7, 8])
        with self.assertRaises(ValueError):
            parse_active_layers("1,1")


if __name__ == "__main__":
    unittest.main()
