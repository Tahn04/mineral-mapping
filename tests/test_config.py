import os
import sys
import tempfile
import yaml
import pytest

# Add the project root to sys.path so 'src' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.config import Config

# Sample YAML content for testing
test_yaml_content = """
processes:
  test_process:
    path: "test/path"
    thresholds:
      parameters:
        param1: [0.1, 0.2, 0.3]
      masks:
        mask1: [0.5]
    pipeline:
      - task: "majority"
        iterations: 2
        size: 3
"""

def test_config_load_and_access():
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.yaml') as tmp:
        tmp.write(test_yaml_content)
        tmp_path = tmp.name
    try:
        cfg = Config(tmp_path)
        # Test top-level access
        assert 'test_process' in cfg.get_processes()
        # Test get_thresholds
        thresholds = cfg.get_thresholds('test_process', 'param1')
        assert thresholds == [0.1, 0.2, 0.3]
        # Test get_masks
        masks = cfg.get_masks('test_process')
        assert 'mask1' in masks
        # Test get_pipeline
        pipeline = cfg.get_pipeline('test_process')
        assert isinstance(pipeline, list)
        assert pipeline[0]['task'] == 'majority'
        assert pipeline[0]['iterations'] == 2
        assert pipeline[0]['size'] == 3
        # Test get_nested
        val = cfg.get_nested('processes', 'test_process', 'path')
        assert val == 'test/path'
    finally:
        os.remove(tmp_path)

def test_config_get_default():
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.yaml') as tmp:
        tmp.write(test_yaml_content)
        tmp_path = tmp.name
    try:
        cfg = Config(tmp_path)
        # Test default return
        assert cfg.get('not_a_key', default=123) == 123
        assert cfg.get_nested('processes', 'test_process', 'not_a_key', default='foo') == 'foo'
    finally:
        os.remove(tmp_path)
