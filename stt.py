import contextlib
import ctypes
import os
from typing import Optional

import speech_recognition as sr


# Prevent PortAudio from trying to auto-start JACK on Linux.
os.environ.setdefault("JACK_NO_START_SERVER", "1")


def _silence_alsa_errors() -> contextlib.AbstractContextManager:
    """Silence noisy ALSA stderr logs during device probing/open."""

    try:
        asound = ctypes.cdll.LoadLibrary("libasound.so")
    except OSError:
        return contextlib.nullcontext()

    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
        None,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
    )

    def _py_error_handler(_filename, _line, _function, _err, _fmt):
        return None

    c_error_handler = ERROR_HANDLER_FUNC(_py_error_handler)

    @contextlib.contextmanager
    def _ctx():
        asound.snd_lib_error_set_handler(c_error_handler)
        try:
            yield
        finally:
            asound.snd_lib_error_set_handler(None)

    return _ctx()


class AlwaysOnSTT:
    def __init__(
        self,
        sample_rate: int = 16000,
        max_window_seconds: float = 8.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_window_seconds = max_window_seconds
        self.listen_timeout_seconds = 1.2
        self.phrase_time_limit_seconds = max_window_seconds
        self.mic_device_index = self._read_mic_device_index()

        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.6
        self.recognizer.non_speaking_duration = 0.35
        self.recognizer.operation_timeout = 9

        with _silence_alsa_errors():
            self.microphone = sr.Microphone(
                sample_rate=self.sample_rate,
                device_index=self.mic_device_index,
            )
            self.source = self.microphone.__enter__()
        self.recognizer.adjust_for_ambient_noise(self.source, duration=1.0)

    def _read_mic_device_index(self) -> Optional[int]:
        raw = os.getenv("STT_MIC_DEVICE_INDEX", "").strip()
        if not raw:
            return self._auto_select_mic_device_index()
        try:
            return int(raw)
        except ValueError:
            print(f"[warn] Ignoring invalid STT_MIC_DEVICE_INDEX='{raw}'")
            return self._auto_select_mic_device_index()

    def _auto_select_mic_device_index(self) -> Optional[int]:
        preferred_keywords = ("pulse", "pipewire", "analog", "mic", "input")
        try:
            with _silence_alsa_errors():
                names = sr.Microphone.list_microphone_names()
        except Exception:
            return None

        best_index = None
        for i, name in enumerate(names):
            lower_name = str(name).lower()
            if any(keyword in lower_name for keyword in preferred_keywords):
                best_index = i
                break

        if best_index is not None:
            print(f"[info] STT using mic device {best_index}: {names[best_index]}")
        return best_index

    def close(self) -> None:
        if getattr(self, "microphone", None) is not None and getattr(self, "source", None) is not None:
            with _silence_alsa_errors():
                self.microphone.__exit__(None, None, None)
            self.source = None
        self.microphone = None
        self.recognizer = None

    def listen_once(self, language: Optional[str] = None) -> str:
        lang = (language or "en-US").strip() or "en-US"
        try:
            audio = self.recognizer.listen(
                self.source,
                timeout=self.listen_timeout_seconds,
                phrase_time_limit=self.phrase_time_limit_seconds,
            )
        except sr.WaitTimeoutError:
            return ""
        except Exception:
            return ""

        try:
            text = self.recognizer.recognize_google(audio, language=lang)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:
            print(f"[warn] Google STT request failed: {exc}")
            return ""

        return str(text).strip()


if __name__ == "__main__":
    stt = AlwaysOnSTT()
    try:
        print("Mic always on. Speak now. Say 'exit' to quit.")
        while True:
            spoken_text = stt.listen_once()
            if not spoken_text:
                continue
            print(f"You said: {spoken_text}")
            if spoken_text.lower() in {"exit", "quit", "stop listening"}:
                break
    finally:
        stt.close()