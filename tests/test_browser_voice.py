"""Deterministic tests for browser voice layer controls and safety guards."""


def test_browser_chat_serves_voice_controls(client) -> None:
    """Verify that /chat includes all required browser voice UI elements."""

    response = client.get("/chat")

    assert response.status_code == 200
    html = response.text

    assert 'id="mic-button"' in html
    assert 'id="mute-button"' in html
    assert 'id="stop-speaking-button"' in html
    assert 'id="voice-banner"' in html
    assert "Start Speaking" in html
    assert "Sound On" in html
    assert "Stop Speaking" in html


def test_browser_chat_voice_safety_guards(client) -> None:
    """Verify JS implementation contracts for STT, TTS, and error states."""

    response = client.get("/chat")

    assert response.status_code == 200
    html = response.text

    # Speech recognition & push-to-talk contract
    assert "SpeechRecognitionClass" in html
    assert "webkitSpeechRecognition" in html
    assert 'recognition.continuous = false' in html
    assert 'recognition.interimResults = true' in html
    assert 'micButton.textContent = "Stop Listening"' in html

    # Recognized transcript populates input without auto-sending
    assert 'inputElement.value = displayText' in html

    # Text-to-speech (TTS) contract
    assert 'SpeechSynthesisUtterance' in html
    assert 'window.speechSynthesis.speak' in html
    assert 'window.speechSynthesis.cancel()' in html

    # Audio mute and stop speaking control contract
    assert 'isMuted' in html
    assert 'muteButton.addEventListener("click"' in html
    assert 'stopSpeakingButton.addEventListener("click"' in html

    # Error banners contract (unsupported vs permission denied)
    assert "Voice input is not supported in this browser. You can continue using text chat." in html
    assert "Microphone access was denied or unavailable. Please enable microphone permissions in your browser settings to use voice input." in html
    assert "event.error === \"not-allowed\"" in html

    # Safety: New chat and send message stop active listening/speaking
    assert 'stopListeningIfActive()' in html
    assert 'stopSpeakingIfActive()' in html
