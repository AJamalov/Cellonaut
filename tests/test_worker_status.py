from __future__ import annotations

from cellonaut.workers import SignalLogStream


def test_signal_log_stream_hides_imagej_headless_banner():
    messages: list[str] = []
    stream = SignalLogStream(messages.append)

    stream.write(
        "Operating in headless mode - the original ImageJ will have limited functionality.\n"
        "Fiji initialized successfully.\n"
    )

    assert messages == ["Fiji initialized successfully."]
