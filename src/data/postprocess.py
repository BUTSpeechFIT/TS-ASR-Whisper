from collections import defaultdict
import string


def truncate_at_repeating_ngram(text, ngram_length=10, min_n=1, max_n=None,
                                min_word_threshold=30, unigram_min_repeat=10,
                                repeat_threshold=10):
    """
    Truncates text if repeating loops are found at the end,
    ignoring case and punctuation during comparison.
    """
    words = text.split()

    if len(words) < min_word_threshold:
        return text

    if max_n is None:
        max_n = min(ngram_length, 6)

    # Helper to clean words for comparison: lowercase and strip punctuation
    # We create a translation table once for efficiency
    table = str.maketrans('', '', string.punctuation)

    def clean(word_list):
        return [w.lower().translate(table) for w in word_list]

    for n in range(min_n, max_n + 1):
        pattern_raw = words[-n:]
        if not pattern_raw: continue

        # Clean the pattern we are looking for
        pattern_clean = clean(pattern_raw)

        count = 0
        idx = len(words)

        while idx >= n:
            chunk_raw = words[idx - n: idx]

            # Compare cleaned versions
            if clean(chunk_raw) == pattern_clean:
                count += 1
                idx -= n
            else:
                break

        limit = unigram_min_repeat if n == 1 else repeat_threshold

        if count >= limit:
            # Truncate and return the original text up to the loop start
            return " ".join(words[:idx]) + " _HALUCINATION_"

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
