<p align="center">
  <img src="assets/logo.png" alt="Eloquent" width="220">
</p>

# Eloquent

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform: Windows](https://img.shields.io/badge/Platform-Windows%2010%2F11-0078d4.svg)]()
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![ASR: Parakeet TDT 0.6B v3 INT8](https://img.shields.io/badge/ASR-Parakeet%200.6B%20v3%20INT8%20(CPU)-76b900.svg)]()
[![Arabic ASR: Cohere Q4_K_M GGUF](https://img.shields.io/badge/Arabic%20ASR-Cohere%20Q4__K__M%20(Vulkan%20%2F%20CPU)-ff6f00.svg)]()
[![Built by axbuilt.net](https://img.shields.io/badge/Built%20by-axbuilt.net-0d1b2a.svg)](https://axbuilt.net)

**Open-source live speech-to-text, voice dictation, and AI writing assistant for Windows that runs fast on an ordinary CPU.**
Built for **voice dictation into any app, AI writing at your cursor, meetings, classes, lectures, interviews, calls, and voice notes**. Speech recognition runs **100% locally on your PC** (audio never leaves your device), with **NVIDIA Parakeet TDT 0.6B v3** as the main default (**25 European languages** on CPU) plus built-in **Arabic & Dialects** support (**Cohere Arabic Q4_K_M GGUF** on Vulkan iGPU / CPU, toggleable anytime with **`Ctrl+Alt+M`**).
Optionally, a free **Gemini 3.1 Flash-Lite** API key polishes the transcript in the background, powers **Dictate** / **Ask AI at cursor**, and generates **Meeting Minutes, Lecture Notes, Summaries, Bullet Points, or Reports** at the end of a session.

<!-- Add your 20s demo GIF or video here:
![Eloquent Demo](docs/demo.gif)
-->

---

## Highlights

- **Fast on CPU, zero recognition lag (Default):** Powered by NVIDIA **Parakeet TDT 0.6B v3 (INT8)** and **Silero VAD** via `sherpa-onnx`. Supports 25 European languages automatically.
- **Local Arabic & Dialects support (`Ctrl+Alt+M`):** Switch anytime to **Cohere Transcribe Arabic (`Q4_K_M` GGUF)** via `transcribe.cpp` (accelerated on Vulkan iGPU or CPU) with automatic right-to-left (RTL) UI and dialect-aware Gemini proofreading across Modern Standard Arabic (الفصحى), Maghrebi/Darija, Egyptian, Levantine, and Gulf speech (including French/English technical terms).
- **Voice Dictation into Any App (`Ctrl+Alt+Space`):** Place your cursor in any text box (ChatGPT, Cursor/VS Code, WhatsApp, Slack, browser, email, docs), press `Ctrl+Alt+Space` to speak, and press again to clean up hesitations/self-corrections and auto-paste directly at your cursor while preserving your clipboard.
- **Ask AI / Compose at Cursor (`Ctrl+Alt+A`):** Speak an instruction (e.g. *"write an email to Sarah pushing tomorrow's meeting to Thursday"*, *"translate this to French"*, *"reply to this message"*) to generate and paste finished text straight at your cursor. Automatically reads highlighted text on screen for context and remembers recent turns in the current session for follow-ups (like *"now generate a subject"*).
- **Mic, System Audio, or Both:** Capture your microphone, Windows speaker audio (`WASAPI loopback` with zero speaker echo for YouTube, calls, and lectures), or both simultaneously (`You` vs `Remote`).
- **Multi-speaker diarization:** Identifies **Speaker 1, Speaker 2, …** on a low-priority background thread (`NVIDIA NeMo TitaNet-Small`) with zero impact on live speech latency, including mid-turn speaker handoffs using Parakeet word timestamps.
- **Background LLM proofreading:** Every few sentences, Gemini 3.1 Flash-Lite fixes misheard words, punctuation, and spoken fillers in a background thread without slowing down live captions. If offline or if the model rewrites too much, raw text is kept automatically.
- **3 Interfaces, 1 Engine:**
  - **Floating Widget (`Eloquent.bat`, default):** Always-on-top compact pill with live audio meter, live captions, expandable transcript drawer (`⤢`), and global hotkeys.
  - **Terminal (`Eloquent-Terminal.bat`):** Clean Rich TUI where polished paragraphs scroll up and live speech stays at the bottom.
  - **Browser (`--ui web`):** Local page streamed via Server-Sent Events (SSE).
- **One-Click Post-Session Outputs (`Generate ▾`):** Turn any recorded session into:
  - **Concise Summary** (`summary.md`)
  - **Key Bullet Points** (`bullet_points.md`)
  - **Meeting Minutes & Action Items** (`meeting_minutes.md`)
  - **Lecture / Study Notes** (`lecture_notes.md`)
  - **Structured Report** (`report.md`)
  - **Clean Article / Memo** (`memo.md`)
  - **Custom Prompt…** (`custom_output.md`)

---

## Quick Start (Windows)

Requirements: **Windows 10/11** and **Python 3.10+** ([download](https://www.python.org/downloads/), tick *"Add python.exe to PATH"*). No GPU needed.

```bat
git clone https://github.com/EllouziMedAmin/Eloquent.git
cd Eloquent
setup.bat
```

That's the whole install. When `setup.bat` finishes you can:
- Press **`Ctrl+Shift+E`** anywhere in Windows to start Eloquent, or
- double-click the **Eloquent** icon on your Desktop / Start Menu.

What `setup.bat` does:
1. Creates `.venv` and installs the dependencies (`requirements.txt`).
2. Creates the **Eloquent** Desktop and Start Menu shortcuts.
3. Starts a tiny background launcher (a few MB, no window) that listens for `Ctrl+Shift+E`, and adds it to Windows startup so the hotkey keeps working after a reboot.
4. Copies `.env.example` to `.env` (optional Gemini key, see below).

On the **first launch** Eloquent downloads the default Parakeet speech model (~670 MB, once) into `models/` (and when switching to Arabic for the first time, downloads the Cohere Arabic GGUF model and Vulkan/CPU runtime into `models/` if not already present).

**Change the launch hotkey:** edit `launch_hotkey` in `%APPDATA%\Eloquent\settings.json` (e.g. `"ctrl+shift+q"`), then sign out and in, or run `.venv\Scripts\python.exe -m eloquent --create-shortcut` again. If another app already uses the key, the reason is written to `%APPDATA%\Eloquent\hotkey.log`.

**Uninstall:** delete the Eloquent shortcuts (Desktop, Start Menu, and `Start Menu\Programs\Startup\Eloquent Hotkey.lnk`), end `pythonw.exe` in Task Manager if it's running, then delete the repo folder.

Or install as a standard Python package:
```bat
pip install -e .
eloquent
```

### Optional: Free Gemini API Key (for live polishing & summaries)
1. Grab a free key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Paste it in the widget (**`⋯` → `Settings`**), or copy `.env.example` to `.env` and set `GEMINI_API_KEY=your_key`.
3. Without a key, Eloquent runs 100% offline with raw Parakeet or Cohere Arabic transcription.

---

## Controls & Shortcuts

### Widget Mode (`Eloquent.bat` or `Ctrl+Shift+E`)
- **`Ctrl+Shift+E`** — Launch Eloquent from anywhere in Windows (handled by the tiny background launcher `setup.bat` installs); if it's already open, it comes to the front.
- **`▶` / `■`** (or **`Ctrl+Shift+Space`**, customizable in Settings) — Start / Stop recording notes or meetings.
- **`Ctrl+Alt+Space`** (customizable in Settings) — **Dictate into any app** (ChatGPT, Reddit, browser, editor): press once with your cursor in any text box, speak, then press again to clean up and paste right at your cursor (click **`■`** instead to cancel without pasting).
- **`Ctrl+Alt+A`** (customizable in Settings) — **Ask AI / Compose at cursor**: press once with your cursor in any text box (optionally highlighting text on screen first), speak an instruction like *"write an email to Sarah pushing tomorrow's meeting to Thursday"*, then press again to generate and paste the reply at your cursor. Ask AI remembers recent turns in the current run so you can move to the Subject box next and say *"now generate a subject"*.
- **`Ctrl+Alt+M`** (customizable in Settings) — **Switch speech model**: toggle in-session between **Parakeet v3 (25 European languages, Default)** and **Arabic & Dialects (Cohere)** without overwriting your startup default, automatically switching live captions, transcripts, and reports between LTR and RTL.
- **`⤢`** (or double-click the pill) — Expand / collapse the full transcript & report drawer.
- **`Generate ▾`** — Generate Summary, Bullet Points, Meeting Minutes, Lecture Notes, Report, Memo, or Custom Prompt.
- **`⋯`** — Settings, *Switch to Arabic / Parakeet*, *Audio source* (`Microphone`, `System audio`, `Mic + System`), *Save audio recording* toggle (`audio.wav`), *Always on top*, *Create shortcuts + launch hotkey*, *Minimize*.
- **`✕`** — Quit Eloquent cleanly (finishes any sentence in progress first).

### Terminal Mode (`Eloquent-Terminal.bat`)
- **`Space`** — Start / Stop recording
- **`G`** — Generate formatted output (`[1]–[6]`)
- **`Q`** or **`Ctrl+C`** — Stop and quit

### CLI Options
```bat
python -m eloquent --ui widget                     # floating widget (default: Parakeet 25 European languages)
python -m eloquent --speech-lang arabic            # start in Arabic mode (Cohere Q4_K_M GGUF, RTL UI)
python -m eloquent --ui terminal --save-audio      # terminal UI + save audio.wav
python -m eloquent --ui web                        # browser UI at http://127.0.0.1:8765
python -m eloquent --input lecture.wav --format lecture   # transcribe WAV & generate lecture_notes.md
python -m eloquent --no-clean                      # raw ASR only, no LLM
python -m eloquent --no-diarize                    # disable multi-speaker identification
python -m eloquent --list-devices                  # list microphone indices
```

All sessions are saved to `sessions/<YYYY-MM-DD_HH-MM-SS>/` with `raw.txt`, `clean.txt`, optional `audio.wav`, and any generated `.md` files.

---

## Roadmap

- [x] Shared low-latency engine + Widget, Terminal, and SSE Web UIs
- [x] Post-session generation (Summary, Bullets, Meeting Minutes, Lecture Notes, Report, Memo, Custom Prompt)
- [x] Multi-speaker diarization (`Speaker 1`, `Speaker 2`, …)
- [x] System-wide Voice Dictation (`Ctrl+Alt+Space`) and Ask AI / Compose at Cursor (`Ctrl+Alt+A`) with screen selection context
- [x] Windows WASAPI system audio loopback (`Microphone`, `System audio`, `Mic + System`)
- [x] Local Arabic & dialects speech engine (`Cohere Transcribe Arabic Q4_K_M GGUF` on Vulkan / CPU) + RTL UI
- [ ] Local LLM backend (Ollama / LM Studio / `llama.cpp` OpenAI-compatible endpoints)
- [ ] Standalone Windows installer (`.exe`)

---

## License & Credits

- **Built by:** [axbuilt.net](https://axbuilt.net) · [GitHub](https://github.com/EllouziMedAmin/Eloquent)
- **Code:** [MIT License](LICENSE)
- **Default ASR Model (25 European languages):** [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (CC-BY-4.0), INT8 ONNX export via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- **Arabic ASR Model:** [Cohere Transcribe Arabic (Q4_K_M GGUF)](https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf) via [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (CPU + Vulkan GPU)
- **VAD:** [Silero VAD](https://github.com/snakers4/silero-vad)

> Independent open-source project; not affiliated with Google or NVIDIA.
