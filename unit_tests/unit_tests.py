import pytest
from src.helpers import get_target_text

class TestHelpers:
    def test_get_target_text_v1(self):
        text = "a b c d \nHEADER1:\n target text\nHEADER2:\nabcdefg\nHEADER3:\n12345"
        headers = ['H1', 'H2', 'HEADER1', 'HEADER2', 'HEADER3']
        target_headers = ['HEADER1']

        res = get_target_text(text, headers, target_headers)
        
        assert res == "target text"

    def test_get_target_text_v2(self):
        text = "a b c d \nHEADER1:\n target text\nHEADER2:\nabcdefg\nHEADER3:\n12345"
        headers = ['H1', 'H2', 'HEADER1', 'HEADER2', 'HEADER3']
        target_headers = ['NOTEXISTENT']

        res = get_target_text(text, headers, target_headers)
        
        assert res == "NOT FOUND"

    def test_get_target_text_v3(self):
        text = "a b c d \nHEADER1:\n target text\nHEADER2:\nabcdefg\nHEADER3:\n12345"
        headers = ['H1', 'H2', 'HEADER1', 'HEADER2', 'HEADER3']
        target_headers = ['HEADER1', 'HEADER2']

        res = get_target_text(text, headers, target_headers)
        
        assert res == "MULTIPLE FOUND"