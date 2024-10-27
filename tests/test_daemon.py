from jarvis.daemon import Daemon


def test_daemon_initialization():
    """Test that the daemon can be initialized using the default method"""
    daemon = Daemon.default()
    assert isinstance(daemon, Daemon)
    assert daemon.config is not None
    assert daemon.audio_transciever is not None
    assert daemon.jarvis_agent is not None


# @pytest.mark.skip(reason="This test requires audio input and should be run manually")
def test_daemon_run():
    """Test running the daemon (requires audio input)"""
    daemon = Daemon.default()
    daemon.run()
