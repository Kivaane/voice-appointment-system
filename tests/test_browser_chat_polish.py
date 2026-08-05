def test_browser_chat_contains_milestone_one_request_guards(client) -> None:
    response = client.get("/chat")

    assert response.status_code == 200
    html = response.text
    assert 'id="new-chat-button"' in html
    assert "let isSending = false" in html
    assert "if (isSending)" in html
    assert "request_id: requestId" in html
    assert "sendButton.disabled = isLoading" in html
    assert "messagesElement.scrollTop" in html
    assert "responseBody.message" in html
    assert "Please check your connection and try again" in html
    assert "messagesElement.replaceChildren()" in html
    assert "microphone" not in html.lower()
