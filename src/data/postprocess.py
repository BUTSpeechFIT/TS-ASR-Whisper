from collections import defaultdict


def count_ngrams(text, min_n=2, max_n=5):
    words = text.split()
    counts = defaultdict(int)
    for n in range(min_n, max_n + 1):
        for i in range(len(words) - n + 1):
            ngram_words = words[i:i + n]
            # Skip n-grams where all words are identical (case-insensitive)
            if all(word.lower() == ngram_words[0].lower() for word in ngram_words):
                continue
            ngram = ' '.join(ngram_words)
            counts[ngram] += 1
    return counts


def truncate_at_repeating_ngram(text, ngram_length=10, min_n=1, max_n=None, min_word_threshold=30,
                                unigram_min_repeat=10, repeat_threshold=10):
    """
    Truncate text at the first occurrence of a repeating n-gram that occurs more than repeat_threshold times.

    Args:
        text: Input text to process
        ngram_length: Target n-gram length to check for (default: 10 words)
        min_n: Minimum n-gram size to check (default: 1)
        max_n: Maximum n-gram size to check (default: ngram_length)
        min_word_threshold: Minimum number of words required to process (default: 30)
        unigram_min_repeat: Minimum consecutive repeats for unigrams (default: 3)
        repeat_threshold: Minimum total occurrences of n-gram to consider it repeating (default: 2)

    Returns:
        Truncated text up to the first repeating n-gram above threshold, or original text if not found
    """
    if max_n is None:
        max_n = ngram_length

    words = text.split()
    if len(words) < min_word_threshold:
        return text

    earliest_truncation_idx = len(words)  # Default: no truncation

    # Handle unigrams with consecutive repetition
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
                earliest_truncation_idx = min(earliest_truncation_idx, i + 1)
                break  # Prioritize consecutive unigrams

    # Count all n-grams first
    all_ngram_counts = count_ngrams(text, min_n=max(2, min_n), max_n=max_n)

    # Find earliest occurrence of any repeated n-gram (above threshold)
    lengths_to_check = [ngram_length] + [n for n in range(min_n, max_n + 1)
                                         if n != ngram_length and n > 1]

    for n in lengths_to_check:
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i + n])
            if all_ngram_counts[ngram] > repeat_threshold:
                earliest_truncation_idx = min(earliest_truncation_idx, i + n)

    # Return truncated text if needed
    if earliest_truncation_idx < len(words):
        return ' '.join(words[:earliest_truncation_idx])
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
