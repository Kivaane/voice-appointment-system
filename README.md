# Personal Voice Appointment System - Browser Voice Layer

## Overview
This repository contains a browser voice layer integrated into the personal appointment chatbot. It provides push-to-talk speech-to-text (STT) and text-to-speech (TTS) playback of assistant replies directly within the Web browser interface.

## Browser Voice Scope & Architecture
- **Interface**: Built directly into `app/static/chat.html`, served via FastAPI at `/chat`.
- **Speech Recognition (STT)**: Uses standard browser APIs (`window.SpeechRecognition` or `window.webkitSpeechRecognition`).
- **Text-to-Speech (TTS)**: Uses standard browser APIs (`window.speechSynthesis` and `SpeechSynthesisUtterance`).
- **Push-to-Talk**: Speech recognition starts ONLY upon explicit user click on the **Start Speaking** button.
- **Manual Review**: Recognized transcript populates the text input box. Speech is **never** sent automatically; the user must manually click **Send** or press Enter.
- **Assistant Speech Playback**: Assistant replies are automatically spoken aloud unless **Muted**.
- **Audio Controls**:
  - **Start Speaking / Stop Listening**: Push-to-talk toggle button.
  - **Sound On / Muted**: Mute toggle button.
  - **Stop Speaking**: Appears during active TTS playback to cancel speech immediately.
  - **New Chat**: Resets session, stops active listening/speaking, and generates a new thread ID.

## Safety & Graceful Fallbacks
1. **Unsupported Browser**: When neither `SpeechRecognition` nor `webkitSpeechRecognition` is available, a warning banner appears (`Voice input is not supported in this browser. You can continue using text chat.`), the microphone button is disabled, and text chat remains 100% functional.
2. **Permission Denied**: If microphone access is denied by the user or browser settings (`not-allowed` or `service-not-allowed` recognition error), a permission banner appears (`Microphone access was denied or unavailable. Please enable microphone permissions in your browser settings to use voice input.`).
3. **Banner Isolation**: Unsupported and permission-denied banners are mutually exclusive and never display simultaneously.
4. **Permission Reset**: If recognition later starts successfully after permission was previously denied, the permission warning banner is automatically cleared.
5. **Request In-Flight Safety**: Microphone controls are disabled while a chatbot backend request is currently loading (`isSending = true`).

## Known Browser Limitations
- **Browser API Support**: Web Speech API (`SpeechRecognition`) is natively supported in modern Google Chrome, Microsoft Edge, and Safari, but may be unavailable or disabled in Firefox, Brave (by default), or private browsing windows.
- **Network Dependency**: Some browsers (such as Google Chrome) require an active internet connection to process SpeechRecognition audio through browser-native speech services.
- **Language**: Speech recognition and synthesis are configured for English (`en-US`).
- **Microphone Permissions**: Microphone permissions must be granted at the browser/domain level. If denied, users must enable permission in browser site settings.
