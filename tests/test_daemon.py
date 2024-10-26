from jarvis.daemon import WakeDaemon


def test_daemon_initialization():
    """Test that the daemon can be initialized using the default method"""
    daemon = WakeDaemon.default()
    assert isinstance(daemon, WakeDaemon)
    assert daemon.config is not None
    assert daemon.audio_transciever is not None
    assert daemon.jarvis_agent is not None


# @pytest.mark.skip(reason="This test requires audio input and should be run manually")
def test_daemon_run():
    """Test running the daemon (requires audio input)"""
    daemon = WakeDaemon.default()
    daemon.run()
