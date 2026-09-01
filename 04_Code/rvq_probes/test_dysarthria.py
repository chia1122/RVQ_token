import unittest, torch
from rvq_probes.dysarthria import DysarthriaProbe
class DysarthriaTests(unittest.TestCase):
    def test_padding_is_excluded_from_mean(self):
        model=DysarthriaProbe(2); model.classifier=torch.nn.Identity(); model.norm=torch.nn.Identity()
        pooled=torch.tensor([[2.,4.]])
        self.assertTrue(torch.equal(model(pooled),pooled))
if __name__ == "__main__": unittest.main()
