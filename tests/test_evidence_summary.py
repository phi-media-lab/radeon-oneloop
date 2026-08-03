import unittest

from reports.summarize_training_evidence import parse_gpu_samples, parse_progress


class EvidenceSummaryTests(unittest.TestCase):
    def test_training_steps_are_recovered_from_progress_order(self) -> None:
        log = "\n".join(
            [
                "INFO ot_train.py:519 step:50 smpl:800 loss:1.25 grdn:2.5 lr:1e-5 updt_s:0.51 data_s:0.01",
                "INFO ot_train.py:519 step:100 smpl:2K loss:0.75 grdn:1.5 lr:1e-5 updt_s:0.49 data_s:0.02",
            ]
        )
        points = parse_progress(log)
        self.assertEqual([point["step"] for point in points], [50, 100])
        self.assertEqual(points[-1]["loss"], 0.75)

    def test_gpu_csv_columns_are_summarized(self) -> None:
        value = "\n".join(
            [
                "timestamp_utc\tdevice_sample",
                "2026-08-03T00:00:00Z\tcard0,80,12,0,N/A,0;;",
                "2026-08-03T00:00:01Z\tcard0,100,17,0,N/A,0;;",
            ]
        )
        summary = parse_gpu_samples(value)
        self.assertEqual(summary["gpu_utilization_peak_percent"], 100)
        self.assertEqual(summary["vram_allocated_peak_percent"], 17)
        self.assertEqual(summary["gpu_utilization_mean_percent"], 90)


if __name__ == "__main__":
    unittest.main()
