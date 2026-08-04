import json
import importlib.util
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

from sim.genesis_so101.live_frame_presenter import LiveFrameHttpPresenter


class LiveFrameHttpPresenterTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is not installed")
    def test_loopback_presenter_serves_page_health_and_latest_jpeg(self):
        presenter = LiveFrameHttpPresenter("127.0.0.1", 0, title="Test Twin")
        presenter.start()
        try:
            with urlopen(presenter.url, timeout=2) as response:
                self.assertIn(b"Test Twin", response.read())
            with self.assertRaises(HTTPError) as missing:
                urlopen(presenter.url + "frame.jpg", timeout=2)
            self.assertEqual(missing.exception.code, 503)

            frame = np.zeros((4, 8, 3), dtype=np.uint8)
            frame[:, :4, 0] = 255
            presenter.publish(frame)
            with urlopen(presenter.url + "frame.jpg", timeout=2) as response:
                payload = response.read()
            self.assertTrue(payload.startswith(b"\xff\xd8"))
            with urlopen(presenter.url + "health.json", timeout=2) as response:
                health = json.loads(response.read())
            self.assertEqual(health["frames_published"], 1)
            self.assertEqual(health["image_size_wh"], [8, 4])
            self.assertFalse(health["physical_output"])
            metrics = presenter.metrics()
            self.assertEqual(metrics["bind_scope"], "loopback_only")
            self.assertGreaterEqual(metrics["requests"]["frame"], 2)
        finally:
            presenter.close()

    def test_non_loopback_bind_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            LiveFrameHttpPresenter("0.0.0.0", 0)


if __name__ == "__main__":
    unittest.main()
