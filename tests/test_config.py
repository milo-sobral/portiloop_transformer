from transformiloop.src.utils.configs import get_default_config

def test_config():
    name = 'Test Name'
    config = get_default_config(name)
    assert config is not None