from collections import defaultdict

def truncate_at_repeating_ngram(text, ngram_length=10, min_n=1, max_n=None,
                                min_word_threshold=30, unigram_min_repeat=10,
                                repeat_threshold=10):
    """
    Simplified: Checks ONLY the end of the text for repeating loops.
    """
    words = text.split()

    # 1. Safety check: Don't process very short texts
    if len(words) < min_word_threshold:
        return text

    # If max_n is not set, we assume the loops are short (e.g., up to 6 words)
    # or use the provided ngram_length if it's smaller.
    if max_n is None:
        max_n = min(ngram_length, 6)

        # 2. Iterate through possible loop sizes (e.g., 1-word loop, 2-word loop...)
    # We check small loops first as they are most common.
    for n in range(min_n, max_n + 1):

        # Get the "candidate pattern" from the very end of the text
        pattern = words[-n:]

        # If pattern is too short to be unique (optional safety), skip
        if not pattern: continue

        # Count how many times this specific pattern repeats backwards
        count = 0
        idx = len(words)

        while idx >= n:
            # Check previous chunk
            chunk = words[idx - n: idx]

            # Compare (case-insensitive)
            if [w.lower() for w in chunk] == [w.lower() for w in pattern]:
                count += 1
                idx -= n
            else:
                break

        # 3. Check Thresholds
        # Use unigram_min_repeat if loop size is 1, otherwise repeat_threshold
        limit = unigram_min_repeat if n == 1 else repeat_threshold

        if count >= limit:
            # Found a loop at the end! Truncate.
            # We keep the text up to the start of the repetitions
            # (Or optionally keep 1 instance: words[:idx + n])
            return " ".join(words[:idx])

    return text



def find_first_repeating_ngram(text, target_length=10, min_n=1, max_n=None, min_word_threshold=20, unigram_min_repeat=5,
                               ngram_min_repeat=3):
    """
    Find the first repeating n-gram in the text.

    Args:
        text: Input text to analyze
        target_length: Preferred n-gram length to look for
        min_n: Minimum n-gram size to check (default: 1, includes unigrams)
        max_n: Maximum n-gram size to check
        min_word_threshold: Minimum number of words required to process
        unigram_min_repeat: Minimum consecutive repeats for unigrams
        ngram_min_repeat: Minimum total occurrences for n-grams

    Returns:
        Dictionary with details about the first repeating n-gram found, or None
    """
    if max_n is None:
        max_n = target_length

    words = text.split()

    # Heuristic: Don't process if text is too short
    if len(words) < min_word_threshold:
        return None

    # Special handling for unigrams (single words) - look for consecutive repeats
    if min_n == 1:
        for i in range(len(words) - unigram_min_repeat + 1):
            current_word = words[i].lower()
            consecutive_count = 1

            for j in range(i + 1, len(words)):
                if words[j].lower() == current_word:
                    consecutive_count += 1
                else:
                    break

            if consecutive_count >= unigram_min_repeat:
                return {
                    'ngram': words[i],
                    'length': 1,
                    'first_position': i,
                    'repeat_position': i + 1,
                    'words_before_repeat': i + 1,
                    'consecutive_repeats': consecutive_count,
                    'type': 'unigram'
                }

    # Check for n-grams with sufficient total occurrences
    ngram_positions = {}
    lengths_to_check = [target_length] + [n for n in range(2, max_n + 1) if n != target_length]

    for n in lengths_to_check:
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i + n])
            if ngram not in ngram_positions:
                ngram_positions[ngram] = []
            ngram_positions[ngram].append(i)

    # Find the earliest n-gram that repeats enough times
    earliest_ngram = None
    earliest_position = float('inf')

    for ngram, positions in ngram_positions.items():
        if len(positions) >= ngram_min_repeat:
            first_occurrence_end = positions[0] + len(ngram.split())
            if first_occurrence_end < earliest_position:
                earliest_position = first_occurrence_end
                earliest_ngram = {
                    'ngram': ngram,
                    'length': len(ngram.split()),
                    'first_position': positions[0],
                    'repeat_position': positions[1],
                    'words_before_repeat': first_occurrence_end,
                    'total_occurrences': len(positions),
                    'type': 'ngram'
                }

    return earliest_ngram
