## 1.0.0

- Initial release
- Kokoro-82M ONNX neural TTS via Wyoming protocol on port 10200
- Fix: `TtsProgram` and `TtsVoice` now include required `version` field
- Fix: `Describe` event handled in handler (Wyoming API ≥ 1.5)
- Fix: `server.run()` called with single handler argument
- 11 voices: Emma, Isabella, George, Lewis (en-GB) + Heart, Bella, Nicole, Sarah, Sky, Adam, Michael (en-US)
