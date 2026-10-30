"""Unit tests for dataset loader and raw GSM8K persistence."""

import unittest

from src.data.loader import generate_problem_id, load_raw_gsm8k


class TestLoader(unittest.TestCase):
    def test_generate_problem_id_determinism(self):
        q1 = "Natalia sold clips to 48 of her friends in April..."
        q2 = "Natalia sold clips to 48 of her friends in April..."
        q3 = "Different question altogether"

        id1 = generate_problem_id(q1)
        id2 = generate_problem_id(q2)
        id3 = generate_problem_id(q3)

        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)
        self.assertTrue(id1.startswith("gsm8k_"))
        self.assertEqual(len(id1), 22)  # 'gsm8k_' + 16 hex chars

    def test_raw_dataset_shapes_and_columns(self):
        train_df = load_raw_gsm8k("train")
        test_df = load_raw_gsm8k("test")

        self.assertEqual(len(train_df), 7473)
        self.assertEqual(len(test_df), 1319)

        expected_columns = {"problem_id", "question", "answer", "split"}
        self.assertEqual(set(train_df.columns), expected_columns)
        self.assertEqual(set(test_df.columns), expected_columns)

        # Check uniqueness of problem IDs
        self.assertEqual(len(train_df["problem_id"].unique()), 7473)
        self.assertEqual(len(test_df["problem_id"].unique()), 1319)

        # Check answers contain the standard answer delimiter
        sample_answer = train_df.iloc[0]["answer"]
        self.assertIn("####", sample_answer)


if __name__ == "__main__":
    unittest.main()
