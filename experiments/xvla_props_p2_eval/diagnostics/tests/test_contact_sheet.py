import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
av = None
try:
    import av
    from PIL import Image
    from experiments.xvla_props_p2_eval.diagnostics import contact_sheet
except ImportError as exc:      # the whole point is to be hard about this: no silent skip
    raise AssertionError(f'contact sheet dependencies are missing: {exc}')


def write_video(path, n, w=64, h=32):
    """Frame k is a flat colour whose red channel encodes k (in steps of 20), so a decoded frame names its own index."""
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=30)
        stream.width, stream.height, stream.pix_fmt = w, h, 'yuv420p'
        stream.options = {'qscale': '1'}
        for k in range(n):
            img = np.zeros((h, w, 3), np.uint8)
            img[..., 0] = 20*k
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class ContactSheetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        write_video(self.tmp/'cameras.mp4', 10)

    def test_a_requested_tick_returns_that_frame(self):
        got = contact_sheet.frames_at(self.tmp/'cameras.mp4', [0, 3, 9])
        self.assertEqual(sorted(got), [0, 3, 9])
        for k in (0, 3, 9):
            self.assertAlmostEqual(float(got[k][..., 0].mean()), 20*k, delta=8, msg=f'tick {k} returned the wrong frame')

    def test_the_final_state_frame_is_reachable_and_one_past_it_is_not(self):
        contact_sheet.frames_at(self.tmp/'cameras.mp4', [9])            # 10 frames = ticks 0..8 plus the final state
        with self.assertRaises(ValueError):
            contact_sheet.frames_at(self.tmp/'cameras.mp4', [10])

    def test_the_sheet_stacks_rows_in_tick_order_whatever_the_request_order(self):
        out = self.tmp/'s.png'
        used = contact_sheet.sheet(self.tmp, [7, 2, 5], out, scale=1)
        self.assertEqual(used, [2, 5, 7])
        img = np.asarray(Image.open(out))
        self.assertEqual(img.shape, (3*32, 64, 3))
        for row, k in enumerate((2, 5, 7)):
            self.assertAlmostEqual(float(img[row*32:(row+1)*32, :, 0].mean()), 20*k, delta=8)

    def test_scale_downsamples(self):
        out = self.tmp/'s.png'
        contact_sheet.sheet(self.tmp, [1, 4], out, scale=2)
        self.assertEqual(np.asarray(Image.open(out)).shape, (2*16, 32, 3))


if __name__ == '__main__':
    unittest.main()
