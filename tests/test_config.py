from transformiloop.src.utils.configs import initialize_config, validate_config
import unittest

class TestFinetuneDataset(unittest.TestCase):

    def setUp(self):
        self.config = initialize_config("TEST_CONFIG")

    def test_config_None(self):
        self.assertNotEqual(self.config, None)

    def test_config_valid(self):
        self.assertEqual(validate_config(self.config), True)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()