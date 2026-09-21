from cellonaut.colors import normalize_rgb_hex


def test_normalize_rgb_hex_canonicalizes_valid_colors():
    assert normalize_rgb_hex("  #a1b2c3 ") == "#A1B2C3"


def test_normalize_rgb_hex_uses_requested_fallback_for_invalid_colors():
    assert normalize_rgb_hex("not-a-color") == ""
    assert normalize_rgb_hex("#12345G", fallback="#FFFFFF") == "#FFFFFF"
