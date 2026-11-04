"""Unit tests for model loading and tokenizer configuration."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.model import (
    HAS_TORCH,
    load_tokenizer,
    merge_adapter_to_base,
    resolve_device_map,
    resolve_torch_dtype,
)


class TestModelLoading(unittest.TestCase):
    """Test suite for model loading utilities and device/dtype resolution."""

    def test_resolve_torch_dtype_strings(self) -> None:
        """Verify dtype string resolution."""
        if not HAS_TORCH:
            self.assertEqual(resolve_torch_dtype("bfloat16"), "bfloat16")
            self.assertEqual(resolve_torch_dtype("float16"), "float16")
            self.assertEqual(resolve_torch_dtype("float32"), "float32")
            self.assertEqual(resolve_torch_dtype("auto"), "auto")
        else:
            import torch
            self.assertEqual(resolve_torch_dtype("float32"), torch.float32)
            self.assertEqual(resolve_torch_dtype("float16"), torch.float16)
            self.assertEqual(resolve_torch_dtype("auto"), "auto")

    def test_resolve_device_map(self) -> None:
        """Verify device map resolution logic."""
        self.assertEqual(resolve_device_map("cpu"), "cpu")
        self.assertEqual(resolve_device_map("cuda"), "cuda")
        custom_map = {"model.embed_tokens": 0, "model.layers": 0}
        self.assertEqual(resolve_device_map(custom_map), custom_map)

    @patch("src.model.AutoTokenizer.from_pretrained")
    def test_load_tokenizer_left_padding(self, mock_from_pretrained: MagicMock) -> None:
        """Verify that load_tokenizer forces left-padding for causal LM generation."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.padding_side = "right"
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|eos|>"
        mock_tokenizer.eos_token_id = 99
        mock_from_pretrained.return_value = mock_tokenizer

        tok = load_tokenizer("mock-model-path", padding_side="left")

        # Verify left padding
        self.assertEqual(tok.padding_side, "left")
        # Verify pad token set to eos token when pad token is missing
        self.assertEqual(tok.pad_token, "<|eos|>")
        self.assertEqual(tok.pad_token_id, 99)

    def test_merge_adapter_to_base_non_peft(self) -> None:
        """Verify that merge_adapter_to_base returns normal models untouched."""
        mock_model = MagicMock()
        result = merge_adapter_to_base(mock_model)
        self.assertIs(result, mock_model)


if __name__ == "__main__":
    unittest.main()
