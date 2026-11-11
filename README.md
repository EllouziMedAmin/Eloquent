# Eloquent

**Open-source live transcription for Windows that runs on your CPU.**
It works for meetings, classes, lectures, interviews, calls and notes to yourself. Speech recognition runs **locally**: no GPU needed, and audio never leaves your PC.
Optionally, a free **Gemini 3.1 Flash-Lite** key polishes the text in the background (fixes misheard words, punctuation, filler words) without slowing down the live transcript.

> Not affiliated with Google or NVIDIA.

## How it works

```
mic → Silero VAD → Parakeet TDT 0.6B v3 (INT8, CPU) → live text (instant)
                                         └→ every few sentences → Gemini (background) → polished paragraph
```

- The recognizer never waits for the LLM. If Gemini is slow, offline or rewrites too much, the raw text is kept.
- Parakeet v3 supports 25 European languages.
- Each recording is saved in `sessions/<date_time>/`: `raw.txt` (timestamped), `clean.txt` (polished), and `audio.wav` if you turn on audio saving.

## Choose your interface

| Mode | Start with | Good for |
|---|---|---|
| **Widget** | `Eloquent.bat` | A small floating pill on top of other windows; expand it to read the transcript |
| **Terminal** | `Eloquent-Terminal.bat` or `python -m eloquent --ui terminal` | Lightweight, keyboard-driven use |
| **Browser** | `python -m eloquent --ui web` | Big readable page, for example on a second screen |

All three use the same engine, so speed and accuracy are identical.

## Install (from source)

Requirements: Windows 10/11, Python 3.10+.

```bat
git clone https://github.com/EllouziMedAmin/Eloquent.git
cd eloquent
setup.bat
```

`setup.bat` creates a `.venv` and installs the dependencies. On first launch the speech model (~670 MB) downloads once into `models/`.

### Gemini key (optional, free)
1. Get a key at https://aistudio.google.com/apikey
2. Paste it in the widget: **⋯ → Settings**. Alternatively, create a `.env` file next to this README containing `GEMINI_API_KEY=your_key`.

Without a key, Eloquent still works and shows the raw transcript.

## Widget controls

- **▶ / ■** start/stop recording. The global hotkey `Ctrl+Alt+Space` does the same (you can change or disable it in Settings).
- **⤢** or double-click: show/hide the transcript. Panel options: show raw, Copy, Open folder.
- **⋯** menu: Settings, *Save audio recording*, Always on top, Hide to tray, Quit.
- Drag the pill anywhere; it remembers its position.

## Terminal controls

`Space` start/stop · `Q` quit · `Ctrl+C` stop and quit. Finished paragraphs scroll up (LLM fixes are underlined), and the live line stays at the bottom.

## Command line

```bat
python -m eloquent --ui terminal --save-audio      # also save audio.wav
python -m eloquent --no-clean                      # raw text only, no LLM
python -m eloquent --input lecture.wav             # transcribe a recording
python -m eloquent --threads 6 --device 2          # CPU threads, microphone index
python -m eloquent --list-devices
python -m eloquent --settings-path                 # where settings.json lives
```

Settings are stored in `%APPDATA%\Eloquent\settings.json`. Command-line options apply to that run only.

## Roadmap

- [ ] After a session: summary, bullet points, report, action items, or a custom prompt
- [ ] Local LLM option (Ollama / LM Studio / any OpenAI-compatible endpoint)
- [ ] Windows installer / standalone `.exe`

## Credits

- **ASR:** [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (CC-BY-4.0), INT8 export by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- **VAD:** [Silero VAD](https://github.com/snakers4/silero-vad)
- **Cleanup:** Google Gemini API
