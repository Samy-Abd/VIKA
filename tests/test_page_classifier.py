from page_classifier import classify_page_from_features


def test_high_text_density_without_images_returns_text():
    text = "word " * 200
    assert classify_page_from_features(text, page_area=1000, image_count=0) == "text"


def test_high_text_density_with_images_returns_illustrative():
    text = "word " * 200
    assert classify_page_from_features(text, page_area=1000, image_count=1) == "illustrative"


def test_low_text_density_with_images_returns_scanned():
    assert classify_page_from_features("", page_area=1000, image_count=1) == "scanned"


def test_medium_text_density_with_images_returns_mixed():
    text = "word " * 2
    assert classify_page_from_features(text, page_area=5000, image_count=1) == "mixed"
