"""Synthetic policy examples, kept separate from real corpus statistics."""

CASES = [
    ("required_noise", "I don’t like the-best-food 😊!", ["i", "do", "not", "like", "the", "best", "food", "😊"]),
    ("irregular_negatives", "won't can't shan't cannot", ["will", "not", "can", "not", "shall", "not", "can", "not"]),
    ("regular_negatives", "doesn't wasn't shouldn't", ["does", "not", "was", "not", "should", "not"]),
    ("auxiliaries", "I'm sure they're ready; we've eaten; you'll like it.", ["i", "am", "sure", "they", "are", "ready", "we", "have", "eaten", "you", "will", "like", "it"]),
    ("ambiguous_forms", "Alice's pie; she's ready; I'd go; ain't bad.", ["alice's", "pie", "she's", "ready", "i'd", "go", "ain't", "bad"]),
    ("multiple_suffixes", "I wouldn't've left.", ["i", "would", "not", "have", "left"]),
    ("curly_apostrophes", "DON’T we’re I’m", ["do", "not", "we", "are", "i", "am"]),
    ("unicode_dashes", "the-best-food fast—friendly top–rated", ["the", "best", "food", "fast", "friendly", "top", "rated"]),
    ("zwj_skin_flag_keycap", "👩🏽‍🍳 🇸🇬 1️⃣", ["👩🏽‍🍳", "🇸🇬", "1️⃣"]),
    ("emoji_sequences", "👨‍👩‍👧‍👦 👍🏽 🏳️‍🌈 ❤️", ["👨‍👩‍👧‍👦", "👍🏽", "🏳️‍🌈", "❤️"]),
    ("ascii_emoticons", "Good:-) bad :( Love <3", ["good", ":-)", "bad", ":(", "love", "<3"]),
    ("decimal_numbers", "Paid $1,200.50 for 2 meals.", ["paid", "1,200.50", "for", "2", "meals"]),
    ("alphanumeric", "B52 24hr 3rd", ["b52", "24hr", "3rd"]),
    ("nfc_letters", "cafe\u0301 CAFÉ naïve", ["café", "café", "naïve"]),
    ("punctuation_only", "!!! -- ...", []),
    ("empty", " \n\t", []),
    ("yall_lets", "Y'all, let's eat.", ["you", "all", "let", "us", "eat"]),
]
