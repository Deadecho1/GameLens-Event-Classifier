from typing import List

from gamelens.prediction.model import Interval

LABEL_NONE = "none"


def segment_events(
    labels: List[str],
    *,
    max_none_gap: int = 4,  # allow up to N consecutive 'none' inside an event
    max_other_streak: int = 4,  # tolerate brief mislabels inside an event
    min_event_len: int = 3,  # discard very tiny events
    min_hits: int = 2,  # require at least this many occurrences of the label
    merge_none_gap: int = 8,  # merge same-label intervals if separated by <= this many frames
) -> List[Interval]:
    """
    Label-agnostic interval extraction from a label sequence.

    - Start when encounter non-none label.
    - Grow while allowing small 'none' gaps and brief noise.
    - Keep only intervals meeting global min_event_len and min_hits.
    - Merge same-label intervals separated by small none gaps.
    """
    n = len(labels)
    intervals: List[Interval] = []

    i = 0
    while i < n:
        label = labels[i]
        if label == LABEL_NONE:
            i += 1
            continue

        start = i
        end = i
        hits = 0

        none_streak = 0
        other_streak = 0

        j = i
        while j < n:
            v = labels[j]

            if v == label:
                hits += 1
                end = j
                none_streak = 0
                other_streak = 0
                j += 1
                continue

            if v == LABEL_NONE:
                none_streak += 1
                other_streak = 0
                if none_streak <= max_none_gap:
                    j += 1
                    continue
                break

            # other label (noise)
            other_streak += 1
            none_streak = 0
            if other_streak <= max_other_streak:
                j += 1
                continue
            break

        candidate = Interval(label=label, start=start, end=end)
        if candidate.length >= min_event_len and hits >= min_hits:
            intervals.append(candidate)
            i = end + 1  # consume it
        else:
            # not strong enough -> advance by 1 so we don't skip short events
            i = start + 1

    # Merge adjacent same-label intervals separated by short none gaps
    merged: List[Interval] = []
    for iv in intervals:
        if not merged:
            merged.append(iv)
            continue
        prev = merged[-1]
        if iv.label == prev.label:
            gap = iv.start - prev.end - 1
            if gap <= merge_none_gap:
                prev.end = max(prev.end, iv.end)
                continue
        merged.append(iv)

    return merged
