"""Tests for the typed-client generator (growthbook.codegen)."""

import importlib.util
import json
import os
import sys

import pytest

from growthbook import GrowthBook
from growthbook.codegen import decrypt_payload, extract_feature_types, generate, python_type_for

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "codegen")


def _load_fixture():
    with open(os.path.join(FIXTURE_DIR, "sample_features.json"), encoding="utf-8") as f:
        return json.load(f)


class TestPythonTypeFor:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, "bool"),
            (False, "bool"),
            # GrowthBook numbers are JS numbers: an int default doesn't pin
            # the feature to int (rules may serve decimals), so both map to
            # the same Union.
            (1, "Union[int, float]"),
            (2.5, "Union[int, float]"),
            ("x", "str"),
            ([1, 2], "List[Any]"),
            ({"a": 1}, "Dict[str, Any]"),
            (None, "Any"),
        ],
    )
    def test_mapping(self, value, expected):
        assert python_type_for(value) == expected


class TestExtractFeatureTypes:
    def test_endpoint_payload(self):
        payload = {"features": {"b": {"defaultValue": 1}, "a": {"defaultValue": "x"}}}
        # sorted for deterministic output
        assert list(extract_feature_types(payload)) == ["a", "b"]

    def test_bare_map(self):
        assert extract_feature_types({"f": {"defaultValue": True}}) == {"f": "bool"}

    def test_missing_default_value(self):
        assert extract_feature_types({"f": {}}) == {"f": "Any"}

    def test_non_object_input(self):
        with pytest.raises(ValueError):
            extract_feature_types({"features": []})

    def test_bare_map_with_feature_named_features(self):
        # A valid feature literally named "features" must not be mistaken for
        # the endpoint wrapper: "other" is not an endpoint payload key.
        payload = {"features": {"defaultValue": True}, "other": {"defaultValue": 1}}
        assert extract_feature_types(payload) == {
            "features": "bool",
            "other": "Union[int, float]",
        }

    def test_single_bare_feature_named_features(self):
        # Endpoint wrappers map feature keys to definition objects; a bare
        # definition (non-dict values) is classified as a bare map.
        assert extract_feature_types({"features": {"defaultValue": True}}) == {
            "features": "bool"
        }

    def test_explicit_payload_format_overrides_detection(self):
        wrapper = {"features": {"a": {"defaultValue": "x"}}}
        assert extract_feature_types(wrapper, "payload") == {"a": "str"}
        # Forced bare-map reading treats the whole object as {key: definition}.
        assert extract_feature_types(wrapper, "map") == {"features": "Any"}

    def test_unknown_payload_format_rejected(self):
        with pytest.raises(ValueError):
            extract_feature_types({"f": {"defaultValue": 1}}, "bogus")

    def test_encrypted_payload_rejected(self):
        # Regression: this used to be read as a bare map, fabricating fake
        # Any-typed feature keys (encryptedFeatures, status, dateUpdated).
        payload = {"status": 200, "encryptedFeatures": "aXY=.Y3Q=", "dateUpdated": "2026-01-01"}
        with pytest.raises(ValueError, match="decrypt"):
            extract_feature_types(payload)
        with pytest.raises(ValueError, match="decrypt"):
            extract_feature_types(payload, "payload")


def _encrypt_features(features: dict, key_b64: str) -> str:
    """Test-only inverse of growthbook.decrypt (AES-128-CBC, iv.ct base64)."""
    import os
    from base64 import b64decode, b64encode

    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    iv = os.urandom(16)
    padder = padding.PKCS7(128).padder()
    data = padder.update(json.dumps(features).encode()) + padder.finalize()
    encryptor = Cipher(algorithms.AES128(b64decode(key_b64)), modes.CBC(iv)).encryptor()
    ct = encryptor.update(data) + encryptor.finalize()
    return b64encode(iv).decode() + "." + b64encode(ct).decode()


class TestDecryptPayload:
    KEY = "Zvwv/+uhpFDznZ6SX28Yjg=="  # any 16 bytes, base64

    def test_round_trip(self):
        features = {"dark_mode": {"defaultValue": True}}
        payload = {"status": 200, "encryptedFeatures": _encrypt_features(features, self.KEY)}
        decrypted = decrypt_payload(payload, self.KEY)
        assert decrypted["features"] == features
        assert "encryptedFeatures" not in decrypted
        assert extract_feature_types(decrypted) == {"dark_mode": "bool"}

    def test_key_without_encrypted_features_rejected(self):
        with pytest.raises(ValueError):
            decrypt_payload({"features": {}}, self.KEY)


class TestGenerate:
    def test_matches_golden_file(self):
        with open(os.path.join(FIXTURE_DIR, "expected_output.py"), encoding="utf-8") as f:
            expected = f.read()
        assert generate(_load_fixture()) == expected

    def test_empty_features_rejected(self):
        with pytest.raises(ValueError):
            generate({"features": {}})

    def test_single_feature_uses_plain_methods_not_overloads(self):
        """PEP 484 requires >=2 @overload declarations; a one-feature payload
        must emit plain typed methods instead (regression: the overload form
        made the generated module fail both checkers)."""
        src = generate({"features": {"only_one": {"defaultValue": True}}})
        assert "@overload" not in src
        assert "overload" not in src.split("\n")[5]  # not imported either
        assert "def get_feature_value(self, key: Literal['only_one'], fallback: bool) -> bool:" in src

    def test_unused_typing_names_not_imported(self):
        src = generate({"features": {"a": {"defaultValue": 1}, "b": {"defaultValue": 2}}})
        import_line = next(line for line in src.split("\n") if line.startswith("from typing"))
        assert "Dict" not in import_line and "List" not in import_line

    def test_generated_module_is_runtime_noop(self, tmp_path):
        """The generated subclasses must behave exactly like the base clients."""
        path = tmp_path / "growthbook_features.py"
        path.write_text(generate(_load_fixture()), encoding="utf-8")

        spec = importlib.util.spec_from_file_location("growthbook_features", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["growthbook_features"] = spec.loader.exec_module(module) or module
        try:
            tgb = module.TypedGrowthBook(
                features={"banner_text": {"defaultValue": "hello"}}
            )
            assert isinstance(tgb, GrowthBook)
            assert tgb.get_feature_value("banner_text", "fallback") == "hello"
            assert tgb.get_feature_value("unknown_key", "fallback") == "fallback"
            # No typed methods may exist at runtime — everything is inherited.
            assert "get_feature_value" not in module.TypedGrowthBook.__dict__
            assert "is_on" not in module.TypedGrowthBook.__dict__
        finally:
            del sys.modules["growthbook_features"]
