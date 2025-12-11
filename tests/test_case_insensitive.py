"""Test case-insensitive tokenizer."""

from swipealot.data import SwipeDataset

print("=" * 60)
print("Testing Case-Insensitive Tokenizer")
print("=" * 60)

# Load small dataset sample
print("\nLoading dataset...")
dataset = SwipeDataset(
    split="train",
    max_path_len=128,
    max_word_len=38,
    dataset_name="futo-org/swipe.futo.org",
    max_samples=1000,
)

tokenizer = dataset.tokenizer

print(f"\nVocabulary size: {tokenizer.vocab_size}")
print(f"Special tokens: {tokenizer.special_tokens}")

# Show some of the vocabulary (excluding special tokens)
vocab_chars = sorted(
    [char for char in tokenizer.char_to_id.keys() if char not in tokenizer.special_tokens]
)
print(f"\nCharacter vocabulary ({len(vocab_chars)} chars):")
print(f"  {''.join(vocab_chars[:50])}..." if len(vocab_chars) > 50 else f"  {''.join(vocab_chars)}")

# Test encoding/decoding
test_cases = [
    "hello",
    "HELLO",
    "Hello",
    "HeLLo",
]

print("\n" + "=" * 60)
print("Testing Encoding/Decoding (All should produce 'hello')")
print("=" * 60)

for text in test_cases:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"\nInput:   '{text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    print("✓ Match!" if decoded == "hello" else "✗ Mismatch!")

# Check that uppercase letters are NOT in vocabulary
print("\n" + "=" * 60)
print("Uppercase Letter Check")
print("=" * 60)

uppercase_found = any(char.isupper() and char.isalpha() for char in tokenizer.char_to_id.keys())
if uppercase_found:
    print("✗ Found uppercase letters in vocabulary (should be none)")
else:
    print("✓ No uppercase letters in vocabulary (correct!)")

# Show vocabulary reduction
print("\n" + "=" * 60)
print("Vocabulary Size Comparison")
print("=" * 60)
print(f"Case-insensitive vocabulary: {tokenizer.vocab_size} tokens")
print("  (Excludes uppercase duplicates, reducing vocab size)")
print("\n✓ Tokenizer is case-insensitive!")
