# Kiểm tra bố cục: phát hiện nội dung tràn ra ngoài vùng an toàn của nền HUST.
# Chạy: /usr/bin/python3 build/check_layout.py  (sau khi render PNG vào build/render)
from PIL import Image, ImageChops
import glob
import os

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "..", "images")
KINDS = ["intro", "title", "content-full", "content-hust-full", "section", "end"]
BGS = {k: Image.open(os.path.join(IMG, "hust-blue-169-%s.png" % k)).convert("RGB") for k in KINDS}
problems = 0
for fn in sorted(glob.glob(os.path.join(BASE, "render", "p*.png"))):
    a = Image.open(fn).convert("RGB")
    W, H = a.size
    best = None
    for k, bg in BGS.items():
        d = ImageChops.difference(a, bg.resize(a.size)).convert("L").point(lambda v: 255 if v > 40 else 0)
        frac = d.histogram()[255] / float(W * H)
        if best is None or frac < best[0]:
            best = (frac, k, d)
    frac, kind, d = best
    px = d.load()

    def count(x0, y0, x1, y1, allow=()):
        n = 0
        for y in range(int(y0 / 9.0 * H), min(H, int(y1 / 9.0 * H))):
            for x in range(int(x0 / 16.0 * W), min(W, int(x1 / 16.0 * W))):
                if px[x, y]:
                    xc, yc = x / float(W) * 16, y / float(H) * 9
                    if any(ax0 <= xc <= ax1 and ay0 <= yc <= ay1 for (ax0, ay0, ax1, ay1) in allow):
                        continue
                    n += 1
        return n

    msgs = []
    if kind in ("content-full", "content-hust-full"):
        nb = count(0, 8.05, 16, 9.0, allow=[(7.9, 8.42, 14.9, 8.95), (14.95, 8.40, 15.75, 8.98)])
        nr = count(15.35, 1.1, 16, 8.0)
        # phạm vi thực của nội dung (bỏ thanh tiêu đề, lộ trình, số trang)
        m = d.copy()
        for (x0, y0, x1, y1) in [(0, 0, 16, 1.0), (7.9, 8.42, 14.9, 8.95), (14.95, 8.40, 15.75, 8.98)]:
            m.paste(0, (int(x0 / 16.0 * W), int(y0 / 9.0 * H), int(x1 / 16.0 * W), int(y1 / 9.0 * H)))
        bb = m.getbbox()
        ext = " [bottom=%.2f cm, right=%.2f cm]" % (bb[3] / float(H) * 9, bb[2] / float(W) * 16) if bb else ""
        if nb > 20:
            msgs.append("content below 8.05 cm (%d px)%s" % (nb, ext))
        if nr > 20:
            msgs.append("content right of 15.35 cm (%d px)%s" % (nr, ext))
    if kind == "title":
        bb = d.getbbox()
        if bb:
            x1, y0, y1 = bb[2] / float(W) * 16, bb[1] / float(H) * 9, bb[3] / float(H) * 9
            if x1 > 13.45 or y0 < 2.05 or y1 > 7.65:
                msgs.append("title block outside safe box: x1=%.2f y0=%.2f y1=%.2f" % (x1, y0, y1))
    if msgs:
        problems += 1
        print(os.path.basename(fn), kind, "; ".join(msgs))
print("pages with problems:", problems)
