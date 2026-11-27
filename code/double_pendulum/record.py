r"""Frame-exact recorder through FrameServer.

Why not fast_record_scene: its per-frame draw loop does not re-sync mobject
geometry, so a mobject-only scene (this one) freezes after the first frames
(verified: 10 unique decoded frames out of 1800). FrameServer's render_at does
evaluate -> sync(scene) -> tick for EVERY frame, which is exactly the path the
QA stills (save_frame) use, so the video matches the verified stills.

Loop: grab_pixels(t) -> raw BGRA/BGR pipe -> libx264 (same encoder settings
as RTM's own fast path: crf 18, preset fast, full range).
"""
import os
import subprocess
import sys
import time

import real_time_manim  # noqa: F401  (must be first)
from real_time_manim.frames import FrameServer

from scene import DoublePendulum

HERE = os.path.dirname(os.path.abspath(__file__))


def render(out_path=None, fps=60):
    out = out_path or os.path.expanduser("~/gallary/video/double_pendulum.mp4")
    t0 = time.time()
    srv = FrameServer(DoublePendulum)
    times = srv.frame_times(fps)
    w, h = srv.frame_size
    fmt = srv.readback_format
    print(f"frames={len(times)} size={w}x{h} fmt={fmt}", flush=True)
    pix_fmt = "bgra" if fmt == "bgra" else "bgr24"
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pixel_format", pix_fmt,
           "-video_size", f"{w}x{h}", "-framerate", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-vf", "scale=out_range=pc", "-color_range", "pc",
           "-crf", "18", "-preset", "fast", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i, t in enumerate(times):
            proc.stdin.write(srv.grab_pixels(t))
            if i % 300 == 0:
                print(f"  {i}/{len(times)}  ({time.time()-t0:.0f}s)", flush=True)
    finally:
        proc.stdin.close()
        rc = proc.wait()
        srv.close()
    print(f"ffmpeg rc={rc} -> {out} ({os.path.getsize(out)} bytes, "
          f"{time.time()-t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else None)
