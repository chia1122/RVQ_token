import unittest

from extract_dac_tokens import codes_from_encode_output, infer_codebook_size


class _Quantizer:
    codebook_size = 1024


class _Model:
    quantizer = _Quantizer()


class DacExtractorHelpersTest(unittest.TestCase):
    def test_infer_codebook_size(self):
        self.assertEqual(infer_codebook_size(_Model()), 1024)

    def test_encode_output_codes(self):
        self.assertEqual(codes_from_encode_output(("latent", "codes", "other")), "codes")
        with self.assertRaises(ValueError):
            codes_from_encode_output("unexpected")


if __name__ == "__main__":
    unittest.main()
