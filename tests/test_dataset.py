from transformiloop.src.data.spindle_detect_data import FinetuneDataset
from transformiloop.src.utils.configs import get_default_config
import unittest

class TestFinetuneDataset(unittest.TestCase):

    def setUp(self):
        self.config = get_default_config("TEST")

    def test_dataset(self):
        pass

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()