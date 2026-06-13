#!/usr/bin/env python3
"""Wyoming TTS server wrapping Kokoro-82M ONNX."""
import argparse
import asyncio
import logging
from functools import partial
from pathlib import Path

import numpy as np

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.tts import Synthesize

_LOGGER = logging.getLogger(__name__)

VOICES = {
    "bf_emma":     {"display": "Emma",     "lang": "en-gb"},
    "bf_isabella": {"display": "Isabella", "lang": "en-gb"},
    "bm_george":   {"display": "George",   "lang": "en-gb"},
    "bm_lewis":    {"display": "Lewis",    "lang": "en-gb"},
    "af_heart":    {"display": "Heart",    "lang": "en-us"},
    "af_bella":    {"display": "Bella",    "lang": "en-us"},
    "af_nicole":   {"display": "Nicole",   "lang": "en-us"},
    "af_sarah":    {"display": "Sarah",    "lang": "en-us"},
    "af_sky":      {"display": "Sky",      "lang": "en-us"},
    "am_adam":     {"display": "Adam",     "lang": "en-us"},
    "am_michael":  {"display": "Michael",  "lang": "en-us"},
}

DEFAULT_VOICE = "bf_emma"
ACCENT_LANG   = {"b": "en-gb", "a": "en-us"}


class KokoroEventHandler(AsyncEventHandler):
    def __init__(self, *args, kokoro, speed: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._kokoro = kokoro
        self._speed  = speed

    async def handle_event(self, event: Event) -> bool:
        if not Synthesize.is_type(event.type):
            return True

        synth = Synthesize.from_event(event)
        text  = synth.text
        voice = DEFAULT_VOICE
        if synth.voice and synth.voice.name and synth.voice.name in VOICES:
            voice = synth.voice.name

        lang  = ACCENT_LANG.get(voice[0], "en-us")
        speed = self._speed
        _LOGGER.debug("Synthesise voice=%s text=%r", voice, text[:60])

        try:
            samples, sample_rate = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._kokoro.create(text, voice=voice, speed=speed, lang=lang),
            )
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            sr  = int(sample_rate)

            await self.write_event(AudioStart(rate=sr, width=2, channels=1).event())
            chunk = 4096
            for i in range(0, len(pcm), chunk):
                await self.write_event(
                    AudioChunk(rate=sr, width=2, channels=1, audio=pcm[i : i + chunk]).event()
                )
            await self.write_event(AudioStop().event())
        except Exception:
            _LOGGER.exception("Synthesis failed")

        return True


def _wyoming_info() -> Info:
    return Info(
        tts=[
            TtsProgram(
                name="kokoro",
                description="Kokoro-82M — natural neural TTS",
                attribution=Attribution(
                    name="hexgrad / thewh1teagle",
                    url="https://github.com/hexgrad/kokoro",
                ),
                installed=True,
                version="1.0",
                voices=[
                    TtsVoice(
                        name=vid,
                        description=v["display"],
                        installed=True,
                        version="1.0",
                        languages=[v["lang"]],
                        attribution=Attribution(
                            name="hexgrad", url="https://github.com/hexgrad/kokoro"
                        ),
                    )
                    for vid, v in VOICES.items()
                ],
            )
        ]
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Avery Voice Engine (Wyoming)")
    parser.add_argument("--uri",    default="tcp://0.0.0.0:10200")
    parser.add_argument("--model",  required=True)
    parser.add_argument("--voices", required=True)
    parser.add_argument("--speed",  type=float, default=1.0)
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _LOGGER.info("Loading Kokoro model …")
    from kokoro_onnx import Kokoro
    kokoro = Kokoro(args.model, args.voices)
    _LOGGER.info("Model ready. Listening on %s", args.uri)

    server = AsyncServer.from_uri(args.uri)
    await server.run(
        partial(KokoroEventHandler, kokoro=kokoro, speed=args.speed),
        _wyoming_info(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
