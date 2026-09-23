"""Word-level speaker diarization helpers.

Ported from Buzz (https://github.com/chidiwilliams/buzz), which uses
MahmoudAshraf97/whisper-diarization for speaker identification:

1. The ASR model emits word-level timestamps.
2. Each word is anchored to a pyannote speaker turn by its timestamp.
3. Mid-sentence speaker flips are repaired by majority vote over the sentence
   span (realignment).
4. Words are regrouped into speaker-labeled sentences.

The nltk sentence-boundary check used by whisper-diarization is replaced here
with VoiceBird's existing sentence-punctuation logic (no extra dependency).
All timestamps passed in / returned are in seconds; the intermediate
word-mapping steps work in milliseconds like the original.
"""

_SENTENCE_ENDS = ".!?…"


def get_word_ts_anchor(s: float, e: float, option: str = "start") -> float:
    """Pick the anchor point (start / end / mid) used to place a word on the
    speaker-turn timeline."""
    if option == "end":
        return e
    elif option == "mid":
        return (s + e) / 2
    return s


def get_words_speaker_mapping(
    word_ts: list, speaker_ts: list, word_anchor_option: str = "start"
) -> list:
    """Assign each word to the speaker whose turn covers its timestamp.

    ``word_ts``: [{"start", "end", "text"}] in seconds.
    ``speaker_ts``: [(start, end, speaker)] in seconds.
    Returns [{"word", "start_time", "end_time", "speaker"}] with ms times.
    """
    if not speaker_ts:
        return []
    s, e, sp = speaker_ts[0]
    turn_idx = 0
    mapping = []
    for w in word_ts:
        ws = int(w["start"] * 1000)
        we = int(w["end"] * 1000)
        anchor = get_word_ts_anchor(ws, we, word_anchor_option)
        # Walk forward through turns until the word falls inside one.
        while anchor > float(e):
            turn_idx += 1
            turn_idx = min(turn_idx, len(speaker_ts) - 1)
            s, e, sp = speaker_ts[turn_idx]
            if turn_idx == len(speaker_ts) - 1:
                # Last turn: extend it to the word's end so trailing words
                # stay assigned instead of being dropped.
                e = get_word_ts_anchor(ws, we, option="end")
        mapping.append(
            {"word": w["text"], "start_time": ws, "end_time": we, "speaker": sp}
        )
    return mapping


def _is_sentence_end(word: str) -> bool:
    return bool(word) and word[-1] in _SENTENCE_ENDS


def _first_word_idx_of_sentence(
    word_idx: int, word_list: list, speaker_list: list, max_words: int
) -> int:
    """Left edge of the sentence containing ``word_idx`` (or -1 if it can't be
    determined, e.g. the sentence runs into the previous one's end)."""
    left_idx = word_idx
    while (
        left_idx > 0
        and word_idx - left_idx < max_words
        and speaker_list[left_idx - 1] == speaker_list[left_idx]
        and not _is_sentence_end(word_list[left_idx - 1])
    ):
        left_idx -= 1
    return left_idx if left_idx == 0 or _is_sentence_end(word_list[left_idx - 1]) else -1


def _last_word_idx_of_sentence(word_idx: int, word_list: list, max_words: int) -> int:
    """Right edge of the sentence containing ``word_idx`` (or -1)."""
    right_idx = word_idx
    while (
        right_idx < len(word_list) - 1
        and right_idx - word_idx < max_words
        and not _is_sentence_end(word_list[right_idx])
    ):
        right_idx += 1
    return (
        right_idx
        if right_idx == len(word_list) - 1 or _is_sentence_end(word_list[right_idx])
        else -1
    )


def realign_speaker_mapping(
    word_speaker_mapping: list, max_words_in_sentence: int = 50
) -> list:
    """Repair mid-sentence speaker flips.

    When a speaker change falls inside a sentence, majority-vote the speaker
    over the whole sentence span so the output reads cleanly.
    """
    wsp_len = len(word_speaker_mapping)
    words = [d["word"] for d in word_speaker_mapping]
    speakers = [d["speaker"] for d in word_speaker_mapping]

    k = 0
    while k < wsp_len:
        if (
            k < wsp_len - 1
            and speakers[k] != speakers[k + 1]
            and not _is_sentence_end(words[k])
        ):
            left_idx = _first_word_idx_of_sentence(
                k, words, speakers, max_words_in_sentence
            )
            right_idx = (
                _last_word_idx_of_sentence(
                    k, words, max_words_in_sentence - k + left_idx - 1
                )
                if left_idx > -1
                else -1
            )
            if min(left_idx, right_idx) == -1:
                k += 1
                continue

            spk_labels = speakers[left_idx : right_idx + 1]
            mod_speaker = max(set(spk_labels), key=spk_labels.count)
            if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                k += 1
                continue

            speakers[left_idx : right_idx + 1] = [mod_speaker] * (
                right_idx - left_idx + 1
            )
            k = right_idx
        k += 1

    realigned = []
    for d, spk in zip(word_speaker_mapping, speakers):
        d = dict(d)
        d["speaker"] = spk
        realigned.append(d)
    return realigned


def sentences_speaker_mapping(word_speaker_mapping: list) -> list:
    """Group words into speaker-labeled sentences.

    A new sentence starts on a speaker change or when the accumulated text ends
    with sentence punctuation. Returns [{speaker, start_time, end_time, text}]
    with ms times; ``speaker`` is the raw pyannote label (e.g. SPEAKER_00).
    """
    if not word_speaker_mapping:
        return []
    first = word_speaker_mapping[0]
    snts = []
    snt = {
        "speaker": first["speaker"],
        "start_time": first["start_time"],
        "end_time": first["end_time"],
        "text": "",
    }
    prev_spk = first["speaker"]
    for w in word_speaker_mapping:
        wrd, spk = w["word"], w["speaker"]
        s, e = w["start_time"], w["end_time"]
        # Check whether adding this word completes a sentence (ends with
        # punctuation) or the speaker changes — evaluated on the text BEFORE
        # the word is appended, like the original.
        acc = (snt["text"] + " " + wrd).strip()
        if spk != prev_spk or _is_sentence_end(acc):
            snts.append(snt)
            snt = {
                "speaker": spk,
                "start_time": s,
                "end_time": e,
                "text": "",
            }
        else:
            snt["end_time"] = e
        snt["text"] = (snt["text"] + " " + wrd).strip()
        prev_spk = spk
    snts.append(snt)
    return snts


def diarize_words_to_segments(words: list, speaker_turns: list) -> list:
    """Run the full word-level pipeline (Buzz / whisper-diarization).

    ``words``: [{"start", "end", "text"}] in seconds from the ASR model.
    ``speaker_turns``: [(start, end, speaker)] in seconds from pyannote.
    Returns VoiceBird-style segments
    [{"start", "end", "text", "speaker"}] (seconds), or [] if it can't.
    """
    if not words or not speaker_turns:
        return []
    # The mapping helpers work in milliseconds (like the original
    # whisper-diarization code), so convert the pyannote turns here.
    turns_ms = [(t0 * 1000, t1 * 1000, s) for t0, t1, s in speaker_turns]
    wsm = get_words_speaker_mapping(words, turns_ms)
    wsm = realign_speaker_mapping(wsm)
    ssm = sentences_speaker_mapping(wsm)
    return [
        {
            "start": round(s["start_time"] / 1000.0, 3),
            "end": round(s["end_time"] / 1000.0, 3),
            "text": s["text"],
            "speaker": s["speaker"],
        }
        for s in ssm
        if s["text"].strip()
    ]
