r"""Frame-exact recorder through FrameServer (port of double_pendulum/record.py).

FrameServer's render_at does evaluate -> sync -> tick per frame, the exact
path QA stills (save_frame) use, so the video matches verified stills.
Mesh content renders in this path on current installs.
"""
import os
import subprocess
import time

import real_time_manim  # noqa: F401  (must be first)
from real_time_manim.frames import FrameServer


def render(SceneClass, out_path=None, fps=60, duration=30.0):
    out = out_path or os.path.expanduser("~/gallary/video/boy_surface.mp4")
    t0 = time.time()
    srv = FrameServer(SceneClass)
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
