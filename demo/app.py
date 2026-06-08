import gradio as gr
import numpy as np
import os
import plotly.graph_objects as go
import soundfile as sf
import torch
from librosa import load as libr_load

from pipeline import DixtralDemoProcessor, speakers_by_arrival

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
processor = DixtralDemoProcessor(device=device)

# Palette shared by the diarization plot and the speaker selector. Each color is
# paired with the closest circle emoji so a speaker's selector swatch matches its
# lane color in the plot. Both wrap with modulo, so any number of speakers works.
SPEAKER_PALETTE = [
    ("🔵", "#4C72B0"),
    ("🟠", "#DD8452"),
    ("🟢", "#55A868"),
    ("🔴", "#C44E52"),
    ("🟣", "#8172B2"),
    ("🟤", "#937860"),
    ("🟡", "#CCB974"),
    ("⚪", "#8C8C8C"),
]
SPEAKER_SWATCHES = [e for e, _ in SPEAKER_PALETTE]
SPEAKER_COLORS = [c for _, c in SPEAKER_PALETTE]

# Vertical placement of the waveform band drawn beneath the speaker lanes.
WAVE_YC = -1.3
WAVE_H = 0.5


def _speaker_choices(speakers):
    """Radio (label, value) pairs whose swatch matches the speaker's plot color."""
    return [
        (f"{SPEAKER_SWATCHES[i % len(SPEAKER_SWATCHES)]} {spk}", spk)
        for i, spk in enumerate(speakers)
    ]


def validate_audio_file_length(filepath):
    if not filepath:
        return  # Skip validation if empty

    duration = sf.info(filepath).duration
    if duration > 600:
        raise gr.Error("Audio is too long! Max limit is 600 seconds.")
    if duration < 1:
        raise gr.Error("Audio is too short! Min limit is 1 second.")


def _load_audio(audio_path):
    """Load mono 16k audio for the waveform overlay; None on failure/empty."""
    if not audio_path:
        return None
    try:
        audio, _ = libr_load(audio_path, sr=16_000, mono=True)
        return audio
    except Exception:
        return None


def _waveform_polygon(audio, duration, n_points=3000):
    """A mirrored amplitude-envelope polygon centered in the waveform band."""
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    if n == 0:
        return None
    if n > n_points:
        step = n // n_points
        trimmed = audio[: step * n_points].reshape(n_points, step)
        env = np.abs(trimmed).max(axis=1)
        times = np.linspace(0, duration, n_points)
    else:
        env = np.abs(audio)
        times = np.linspace(0, duration, n)
    peak = float(env.max())
    if peak > 0:
        env = env / peak
    upper = WAVE_YC + env * WAVE_H
    lower = WAVE_YC - env * WAVE_H
    xs = np.concatenate([times, times[::-1]])
    ys = np.concatenate([upper, lower[::-1]])
    return go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(width=0),
        fill="toself",
        fillcolor="rgba(255,255,255,0.28)",
        hoverinfo="skip",
        showlegend=False,
    )


def segments_to_figure(segments, audio=None, highlight=None):
    if not segments and audio is None:
        return go.Figure()

    speakers = speakers_by_arrival(segments)
    color_map = {spk: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, spk in enumerate(speakers)}

    seg_dur = max((s["end"] for s in segments), default=0.0)
    audio_dur = (len(audio) / 16_000) if audio is not None else 0.0
    duration = max(seg_dur, audio_dur, 0.01)

    n_spk = len(speakers)
    # Earliest-arriving speaker (index 0) on the top lane, matching the selector,
    # which lists speakers top-to-bottom in arrival order.
    lane = {spk: n_spk - 1 - i for i, spk in enumerate(speakers)}

    fig = go.Figure()

    # One horizontal-bar trace per speaker holds ALL that speaker's segments.
    # Using base/width bars (one trace per speaker, <=8 traces total) instead of
    # one fig.add_shape rectangle per segment: shapes are layout objects that
    # Plotly serializes and renders one-by-one, which is O(thousands) and was the
    # bottleneck on long files. Bars are a single vectorized trace each.
    by_spk = {spk: {"base": [], "width": [], "y": []} for spk in speakers}
    for seg in segments:
        spk = seg["speaker"]
        d = by_spk[spk]
        d["base"].append(seg["start"])
        d["width"].append(seg["end"] - seg["start"])
        d["y"].append(lane[spk])

    for spk in speakers:
        d = by_spk[spk]
        fig.add_trace(go.Bar(
            x=d["width"],
            base=d["base"],
            y=d["y"],
            orientation="h",
            width=0.8,
            marker=dict(color=color_map[spk], line_width=0),
            opacity=0.85,
            name=spk,
            showlegend=True,
            hovertemplate="%{base:.1f}–%{x:.1f}s<extra>" + spk + "</extra>",
        ))

    # Waveform envelope beneath the lanes, so segment edges can be eyeballed.
    if audio is not None:
        wf = _waveform_polygon(audio, duration)
        if wf is not None:
            fig.add_trace(wf)

    y_bottom = WAVE_YC - WAVE_H - 0.2
    y_top = n_spk - 0.4

    # Highlight the time span currently being replayed, full plot height.
    # Kept as a single shape (there is only ever one), which is cheap.
    if highlight is not None:
        hs, he = highlight
        if he > hs:
            fig.add_shape(
                type="rect",
                x0=hs, x1=he,
                y0=y_bottom, y1=y_top,
                fillcolor="rgba(255,221,87,0.18)",
                line=dict(color="rgba(255,221,87,0.95)", width=2),
                layer="above",
            )

    # Lanes run high-y (top) -> low-y (bottom), so tick labels are reversed to
    # keep each label next to its lane.
    tickvals = list(range(n_spk)) + [WAVE_YC]
    ticktext = list(reversed(speakers)) + ["audio"]

    fig.update_layout(
        height=max(190, 120 + 70 * n_spk),
        margin=dict(l=0, r=0, t=10, b=10),
        barmode="overlay",
        bargap=0,
        xaxis=dict(
            title="Time (s)",
            range=[0, duration * 1.01],
            tickformat=".1f",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=[y_bottom, y_top],
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(30,30,40,0.6)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


# ----------------------------------------------------------------------
# Segment <-> table conversion
# ----------------------------------------------------------------------

def _segments_to_df(segments):
    """List of {start,end,speaker} -> list-of-lists rows for the (type='array') table.

    Plain lists render reliably and avoid the pandas dtype layer that otherwise
    coerces the speaker column to numbers.
    """
    return [[s["start"], s["end"], str(s["speaker"])] for s in segments]


def _clean_speaker(spk):
    """Coerce a cell into a clean speaker label; '' if it is empty/NaN."""
    if spk is None:
        return ""
    if isinstance(spk, float):
        if np.isnan(spk):
            return ""
        # A stray numeric label like 2.0 -> "2"
        return str(int(spk)) if spk == int(spk) else str(spk)
    spk = str(spk).strip()
    return "" if spk.lower() == "nan" else spk


def _df_to_segments(df):
    """Parse an edited table back into validated, sorted segment dicts."""
    if df is None:
        return []
    rows = df.values.tolist() if hasattr(df, "values") else list(df)
    segs = []
    for row in rows:
        if row is None or len(row) < 3:
            continue
        start, end, spk = row[0], row[1], row[2]
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if np.isnan(start) or np.isnan(end):
            continue
        spk = _clean_speaker(spk)
        if not spk or end <= start:
            continue
        segs.append({"start": round(start, 1), "end": round(end, 1), "speaker": spk})
    segs.sort(key=lambda x: x["start"])
    return segs


def _rebuild_diar_state(segments, audio_path, current_speaker=None):
    """Rebuild plot, table, and speaker selector from segments."""
    audio = _load_audio(audio_path)
    fig = segments_to_figure(segments, audio)
    speakers = speakers_by_arrival(segments)
    selected = current_speaker if current_speaker in speakers else (speakers[0] if speakers else None)
    return (
        segments,
        fig,
        _segments_to_df(segments),
        gr.update(choices=_speaker_choices(speakers), value=selected),
        selected,
    )


def run_diarization(audio_path):
    if audio_path is None:
        return [], go.Figure(), [], gr.update(choices=[], value=None), None
    segments = processor.run_diarization(audio_path)
    return _rebuild_diar_state(segments, audio_path)


def apply_corrections(df_data, audio_path, current_speaker):
    """Re-derive state/plot/speaker list from the user-edited segment table."""
    segments = _df_to_segments(df_data)
    return _rebuild_diar_state(segments, audio_path, current_speaker)


# ----------------------------------------------------------------------
# RTTM import / export
# ----------------------------------------------------------------------

def _segments_to_rttm(segments, file_id="audio"):
    lines = []
    for s in sorted(segments, key=lambda x: x["start"]):
        dur = round(s["end"] - s["start"], 1)
        lines.append(
            f"SPEAKER {file_id} 1 {s['start']:.3f} {dur:.3f} "
            f"<NA> <NA> {s['speaker']} <NA> <NA>"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def export_rttm(segments):
    """Write current segments to an .rttm file and hand it to the download button."""
    out = "diarization.rttm"
    with open(out, "w") as f:
        f.write(_segments_to_rttm(segments or []))
    return out


def _parse_rttm(path):
    segs = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                continue
            try:
                start = float(parts[3])
                dur = float(parts[4])
            except ValueError:
                continue
            spk = _clean_speaker(parts[7])
            if not spk or dur <= 0:
                continue
            segs.append({"start": round(start, 1), "end": round(start + dur, 1), "speaker": spk})
    segs.sort(key=lambda x: x["start"])
    return segs


def import_rttm(file, audio_path, current_speaker):
    if file is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    path = file.name if hasattr(file, "name") else file
    segments = _parse_rttm(path)
    return _rebuild_diar_state(segments, audio_path, current_speaker)


# ----------------------------------------------------------------------
# Audio replay
# ----------------------------------------------------------------------

def _slice_audio(audio_path, start, end, sr=16_000):
    """Return (sr, samples) for [start, end] seconds, or None if invalid."""
    audio = _load_audio(audio_path)
    if audio is None:
        return None
    try:
        s = max(0, int(round(float(start) * sr)))
        e = min(len(audio), int(round(float(end) * sr)))
    except (TypeError, ValueError):
        return None
    if e <= s:
        return None
    return (sr, audio[s:e])


def play_segment_row(audio_path, segments, evt: gr.SelectData):
    """Clicking a table row selects it, fills From/To, previews it, and marks it on the plot."""
    if not segments or evt is None or evt.index is None:
        return None, gr.skip(), gr.skip(), gr.skip(), None
    idx = evt.index
    row = idx[0] if isinstance(idx, (list, tuple)) else idx
    if row is None or row < 0 or row >= len(segments):
        return None, gr.skip(), gr.skip(), gr.skip(), None
    seg = segments[row]
    clip = _slice_audio(audio_path, seg["start"], seg["end"])
    fig = segments_to_figure(segments, _load_audio(audio_path), highlight=(seg["start"], seg["end"]))
    return clip, seg["start"], seg["end"], fig, row


def play_range(audio_path, segments, start, end):
    """Re-slice the preview clip and highlight the range whenever From/To changes."""
    if audio_path is None:
        return None, gr.skip()
    clip = _slice_audio(audio_path, start, end)
    try:
        highlight = (float(start), float(end))
    except (TypeError, ValueError):
        highlight = None
    fig = segments_to_figure(segments or [], _load_audio(audio_path), highlight=highlight)
    return clip, fig


def update_segment_range(audio_path, segments, row_idx, start, end, current_speaker):
    """Write the From/To values back into the segment currently selected in the table."""
    segments = [dict(s) for s in (segments or [])]
    audio = _load_audio(audio_path)

    def _refresh(highlight=None, sel=row_idx, clip=gr.skip()):
        fig = segments_to_figure(segments, audio, highlight=highlight)
        speakers = speakers_by_arrival(segments)
        selected = current_speaker if current_speaker in speakers else (speakers[0] if speakers else None)
        return (segments, fig, _segments_to_df(segments),
                gr.update(choices=_speaker_choices(speakers), value=selected), selected, sel, clip)

    if row_idx is None or row_idx < 0 or row_idx >= len(segments):
        return _refresh()  # nothing selected — leave segments untouched
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        return _refresh()
    if np.isnan(start) or np.isnan(end) or end <= start:
        return _refresh()

    edited = segments[row_idx]
    edited["start"], edited["end"] = round(start, 1), round(end, 1)
    segments.sort(key=lambda x: x["start"])
    new_idx = segments.index(edited)
    clip = _slice_audio(audio_path, edited["start"], edited["end"])
    return _refresh(highlight=(edited["start"], edited["end"]), sel=new_idx, clip=clip)


def run_model(audio_path, segments, speaker_choice, query):
    if not segments:
        return "Please run diarization first."
    if audio_path is None:
        return "No audio provided."
    if speaker_choice is None:
        return "Please select a speaker."

    result = processor.process(
        audio_path=audio_path,
        segments=segments,
        target_speakers=[speaker_choice],
        query=query.strip() if query else "Transcribe",
    )
    torch.cuda.empty_cache()
    return result


def transcribe_all(audio_path, segments):
    if not segments:
        return "Please run diarization first."
    if audio_path is None:
        return "No audio provided."

    all_speakers = speakers_by_arrival(segments)

    result = processor.process(
        audio_path=audio_path,
        segments=segments,
        target_speakers=all_speakers,
        query="Transcribe",
    )
    torch.cuda.empty_cache()
    return result


with gr.Blocks(theme=gr.themes.Ocean(), title="Dixtral Demo") as demo:
    gr.Markdown(
        "# Dixtral Demo\n"
        "**Dixtral** is a *target-speaker* speech model (Voxtral-Mini-3B + DiCoW encoder): "
        "given a diarization, it focuses on one chosen speaker in the mixture. On top of "
        "transcription it keeps the underlying LLM's abilities, so you can also ask it "
        "questions about what was said.\n\n"
        "**What you can do**\n"
        "- 🎯 **Target-speaker ASR** — transcribe just the selected speaker (the default "
        "`Transcribe` query), even with overlapping speech.\n"
        "- ❓ **Question answering** — e.g. *\"What did this speaker agree to?\"*, "
        "*\"What is the gender of the speaker?\"*, *\"Which topic is the speaker discussing?\"*\n"
        "- 🧠 **Retained LLM capabilities** — summarization, general-knowledge QA, and "
        "reasoning still work, e.g. *\"Summarize this speaker's points in 3 bullets\"* or "
        "*\"Explain any technical terms mentioned.\"*\n\n"
        "**Flow:** upload audio → diarize → (optionally correct the diarization) → pick a "
        "target speaker → type `Transcribe` for ASR or any free-form question → **Run** "
        "(or **Transcribe All** for the whole conversation).\n\n"
        "> ⚠️ **Note:** this model was trained specifically for **English meeting recordings "
        "up to a few minutes long**. Transcription/TS-ASR is most reliable in that setting; "
        "free-form QA and summarization are limited and inherit the base model's "
        "limitations. Results on other languages, domains, or longer audio may be unreliable."
    )

    segments_state = gr.State([])
    speaker_state = gr.State(None)
    selected_row_state = gr.State(None)

    audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
    diarize_btn = gr.Button("Run Diarization", variant="primary")
    diar_plot = gr.Plot(show_label=False)

    with gr.Accordion("Found issues with diarization? Correct manually below.", open=False) as correction_accordion:
        gr.Markdown(
            "The waveform behind the lanes shows where speech actually is. "
            "**Replay** a segment (click a table row) or any range to verify, edit "
            "start/end (seconds) or speaker labels in the table, add or delete rows, "
            "then **Apply corrections**. You can also import/export the diarization as RTTM."
        )

        gr.Markdown(
            "**Replay & edit a range** — click a row in the table below to select it "
            "(highlighted on the plot above). Adjust **From/To** to preview the new range "
            "in the player, then press **Update selected segment** to write those bounds "
            "back into that segment — no need to edit the table directly."
        )
        with gr.Row():
            replay_start = gr.Number(value=0.0, label="From (s)", precision=1)
            replay_end = gr.Number(value=0.0, label="To (s)", precision=1)
            update_seg_btn = gr.Button("Update selected segment", variant="secondary")
        replay_audio = gr.Audio(label="Selection", type="numpy", interactive=False)

        seg_table = gr.Dataframe(
            headers=["start", "end", "speaker"],
            datatype=["number", "number", "str"],
            type="array",
            col_count=(3, "fixed"),
            row_count=(0, "dynamic"),
            interactive=True,
            label="Diarization segments",
        )
        apply_btn = gr.Button("Apply corrections", variant="secondary")

        with gr.Row():
            export_btn = gr.DownloadButton("⬇ Export RTTM", variant="secondary")
            import_file = gr.File(label="Import RTTM", file_types=[".rttm"], file_count="single")

    speaker_radio = gr.Radio(choices=[], value=None, label="Target speaker")
    query_box = gr.Textbox(value="Transcribe", label="Query ('Transcribe' for ASR, or any free-form question)")
    with gr.Row():
        run_btn = gr.Button("Run", variant="primary")
        transcribe_all_btn = gr.Button("Transcribe All", variant="secondary")

    output_box = gr.Markdown(
        label="Output",
        container=True,
        line_breaks=True,  # keep single newlines (speaker / text separation)
        min_height=200,
    )

    diarize_btn.click(
        fn=run_diarization,
        validator=validate_audio_file_length,
        inputs=[audio_input],
        outputs=[segments_state, diar_plot, seg_table, speaker_radio, speaker_state],
    )

    apply_btn.click(
        fn=apply_corrections,
        inputs=[seg_table, audio_input, speaker_state],
        outputs=[segments_state, diar_plot, seg_table, speaker_radio, speaker_state],
    )

    export_btn.click(
        fn=export_rttm,
        inputs=[segments_state],
        outputs=[export_btn],
    )

    import_file.upload(
        fn=import_rttm,
        inputs=[import_file, audio_input, speaker_state],
        outputs=[segments_state, diar_plot, seg_table, speaker_radio, speaker_state],
    )

    # The accordion lazy-mounts its children, so table updates pushed while it is
    # collapsed are lost. Repopulate the table from the (always-current) state
    # whenever the user expands the section.
    correction_accordion.expand(
        fn=lambda segs: _segments_to_df(segs or []),
        inputs=[segments_state],
        outputs=[seg_table],
    )

    seg_table.select(
        fn=play_segment_row,
        inputs=[audio_input, segments_state],
        outputs=[replay_audio, replay_start, replay_end, diar_plot, selected_row_state],
    )

    # Editing From/To re-slices the preview clip (the gr.Audio play control then
    # plays the new range) and highlights it on the plot.
    for _box in (replay_start, replay_end):
        _box.change(
            fn=play_range,
            inputs=[audio_input, segments_state, replay_start, replay_end],
            outputs=[replay_audio, diar_plot],
        )

    update_seg_btn.click(
        fn=update_segment_range,
        inputs=[audio_input, segments_state, selected_row_state, replay_start, replay_end, speaker_state],
        outputs=[segments_state, diar_plot, seg_table, speaker_radio, speaker_state,
                 selected_row_state, replay_audio],
    )

    speaker_radio.change(
        fn=lambda s: s,
        inputs=[speaker_radio],
        outputs=[speaker_state],
    )

    run_btn.click(
        fn=run_model,
        inputs=[audio_input, segments_state, speaker_state, query_box],
        outputs=[output_box],
    )

    transcribe_all_btn.click(
        fn=transcribe_all,
        inputs=[audio_input, segments_state],
        outputs=[output_box],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2, max_size=20)
    demo.launch(server_name="127.0.0.1", server_port=7860, root_path="/dixtral")
