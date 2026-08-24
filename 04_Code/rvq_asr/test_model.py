import unittest

try:
    import torch  # noqa: F401
except ImportError:
    torch = None
else:
    from rvq_asr.model import RVQTransformerCTC
    from rvq_asr.train_probe import parse_active_layers, resolve_representation_metadata


@unittest.skipIf(torch is None, "PyTorch is not installed")
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

    def test_individual_metadata_and_inactive_tokens(self):
        metadata = resolve_representation_metadata(8, [8], "individual", None, "sum")
        self.assertEqual(metadata["condition"], "individual_q8")
        self.assertEqual(metadata["effective_fusion"], "single_active_layer")
        cumulative = resolve_representation_metadata(1, [1], None, None, "sum")
        self.assertEqual(cumulative["rvq_mode"], "cumulative")
        self.assertEqual(cumulative["condition"], "cumulative_q1")
        with self.assertRaises(ValueError):
            resolve_representation_metadata(8, [7, 8], "individual", None, "sum")
        with self.assertRaises(ValueError):
            resolve_representation_metadata(
                8, [8], "individual", "individual_q7", "sum"
            )

        model = self.make_model(active_rvq_layers=[7]).eval()
        first = torch.zeros((1, 4, 8), dtype=torch.long)
        second = first.clone()
        second[:, :, :7] = 3
        lengths = torch.tensor([4])
        with torch.no_grad():
            first_logits, _ = model(first, lengths)
            second_logits, _ = model(second, lengths)
        self.assertTrue(torch.equal(first_logits, second_logits))


if __name__ == "__main__":
    unittest.main()
