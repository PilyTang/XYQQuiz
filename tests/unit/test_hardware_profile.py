from pathlib import Path
import json

import pytest

from xyq_quiz.config import AppConfig
from xyq_quiz.performance.hardware_profile import HardwareFacts, choose_resource_profile, initialize_resource_profile
from xyq_quiz.performance.settings import runtime_performance_config, save_performance_settings

GIB = 1024 ** 3


@pytest.mark.parametrize('threads,ram,gpu,low',[
    (4,8,True,True),(8,32,True,True),(12,8,True,True),
    (12,16,True,False),(16,64,True,False),(32,64,False,True),
    (None,64,True,True),(16,None,True,True),(16,64,None,True),
])
def test_conservative_first_run_threshold(threads,ram,gpu,low):
    selected,reason = choose_resource_profile(HardwareFacts(threads,ram*GIB if ram else None,gpu))
    assert selected is low
    assert ('低配模式' if low else '标准模式') in reason


def test_first_run_persists_once_and_preserves_manual_override(tmp_path):
    path = tmp_path/'config.json'
    config = AppConfig()
    calls = []
    def slow():
        calls.append(1)
        return HardwareFacts(4,8*GIB,True)
    selected = initialize_resource_profile(config,path,probe=slow)
    assert selected.performance.low_resource_mode
    assert selected.performance.resource_profile_origin == 'automatic'
    assert config.performance.resource_profile_origin == 'pending'
    assert selected.capture.preview_fps == 30  # Persist preferences, not caps.
    assert runtime_performance_config(selected).capture.preview_fps == 10
    again = initialize_resource_profile(AppConfig.load(path),path,probe=slow)
    assert len(calls) == 1 and again.performance.low_resource_mode
    saved = save_performance_settings(path,ocr_backend='auto',preview_backend='auto')
    assert saved.resource_profile_origin == 'automatic'
    assert saved.resource_profile_reason == selected.performance.resource_profile_reason
    save_performance_settings(path,ocr_backend='auto',preview_backend='auto',low_resource_mode=False)
    manual = initialize_resource_profile(AppConfig.load(path),path,probe=slow)
    assert len(calls) == 1 and not manual.performance.low_resource_mode
    assert manual.performance.resource_profile_origin == 'manual'


@pytest.mark.parametrize('choice',[False,True])
def test_legacy_explicit_choices_are_not_reclassified(tmp_path,choice):
    path = tmp_path/'config.json'
    path.write_text(json.dumps({'performance':{'low_resource_mode':choice}}))
    config = AppConfig.load(path)
    def forbidden(): raise AssertionError('must not probe')
    selected = initialize_resource_profile(config,path,probe=forbidden)
    assert selected.performance.low_resource_mode is choice
    assert selected.performance.resource_profile_origin == 'manual'


def test_new_bundled_config_is_pending_despite_placeholder_switch(tmp_path):
    doc = json.loads(Path('config.example.json').read_text(encoding='utf-8'))
    path = tmp_path/'config.json'
    path.write_text(json.dumps(doc))
    chosen = initialize_resource_profile(AppConfig.load(path),path,
        probe=lambda:HardwareFacts(16,64*GIB,True))
    assert not chosen.performance.low_resource_mode
    assert chosen.performance.resource_profile_origin == 'automatic'
    after = json.loads(path.read_text(encoding='utf-8'))
    assert {k:v for k,v in after.items() if k!='performance'} == {k:v for k,v in doc.items() if k!='performance'}


def test_probe_or_save_failure_keeps_app_running_conservatively(tmp_path,monkeypatch):
    import xyq_quiz.performance.hardware_profile as module
    def fail(*_): raise OSError('unavailable')
    monkeypatch.setattr(module,'_atomic_write_json',fail)
    selected = initialize_resource_profile(AppConfig(),tmp_path/'config.json',probe=fail)
    assert selected.performance.low_resource_mode
    assert not (tmp_path/'config.json').exists()
