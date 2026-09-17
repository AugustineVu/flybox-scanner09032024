import unittest
from unittest.mock import mock_open, patch

from utils.app_settings import AppSettings, diff_dicts, merge_json


class TestMergeJson(unittest.TestCase):
    def test_merge_json(self):
        # overrides should take precedence over default
        default = {
            "default": "value",
        }
        overrides = {"default": "overridden"}
        expected = {"default": "overridden"}
        self.assertEqual(merge_json(default, overrides), expected)
        # independent keys should be merged
        default = {
            "default": "value",
            "independent": "value",
        }
        overrides = {"default": "overridden"}
        expected = {"default": "overridden", "independent": "value"}
        self.assertEqual(merge_json(default, overrides), expected)
        # nested keys should be handled
        default = {
            "default": "value",
            "nested": {"default": "value", "independent": "value"},
        }
        overrides = {"default": "overridden", "nested": {"default": "overridden"}}
        expected = {
            "default": "overridden",
            "nested": {"default": "overridden", "independent": "value"},
        }
        self.assertEqual(merge_json(default, overrides), expected)


class TestDiffDicts(unittest.TestCase):
    def test_diff_dicts(self):
        d1 = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
        d2 = {"a": 1, "b": 2, "c": {"d": 3, "e": 5}}
        expected = {"c": {"e": 4}}
        self.assertEqual(diff_dicts(d1, d2), expected)


class TestAppSettings(unittest.TestCase):
    def setUp(self):
        mock_json = """{
            "key": "value"
        }"""
        self.mock_open = mock_open(read_data=mock_json)
        self.patcher = patch("builtins.open", self.mock_open)
        self.patcher.start()
        # without this the mock stays installed for the rest of the process, and
        # every later test quietly reads and writes through it instead of the disk
        self.addCleanup(self.patcher.stop)

        self.app_settings = AppSettings()

    def test_init(self):
        self.assertIsNotNone(self.app_settings.settings)
        self.mock_open.assert_any_call("default_settings.json", "r")
        self.mock_open.assert_any_call("settings.json", "r")
        self.assertEqual(self.mock_open.call_count, 2)

    def test_get(self):
        self.assertEqual(self.app_settings.get("key"), "value")
        self.assertIsNone(self.app_settings.get("nonexistent"))
        # get key from environment variable
        with patch.dict("os.environ", {"SOURCE": "test"}):
            self.assertEqual(self.app_settings.get("video.source"), "test")

    def test_set(self):
        self.app_settings.set("key", "new_value")
        self.assertEqual(self.app_settings.get("key"), "new_value")

    def test_save(self):
        self.app_settings.save(show_message=False)
        self.mock_open.assert_any_call("settings.json", "w")
        self.assertEqual(self.mock_open.call_count, 4)


if __name__ == "__main__":
    unittest.main()
