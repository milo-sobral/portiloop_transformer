from transformiloop.src.utils.configs import initialize_config

def test_config():
    name = 'Test Name'
    config = initialize_config(name)
    assert config is not None