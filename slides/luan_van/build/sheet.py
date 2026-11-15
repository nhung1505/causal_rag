# Ghép 4 trang thành một ảnh 2x2 để xem nhanh.
# Dùng: /usr/bin/python3 build/sheet.py out.png 13 14 15 16
import os
import sys
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
out, pages = sys.argv[1], sys.argv[2:]
tw, th = 640, 360
sheet = Image.new("RGB", (tw * 2, th * ((len(pages) + 1) // 2)), "white")
for i, p in enumerate(pages):
    im = Image.open(os.path.join(BASE, "render", "p%02d.png" % int(p))).convert("RGB").resize((tw, th), Image.LANCZOS)
    sheet.paste(im, ((i % 2) * tw, (i // 2) * th))
sheet.save(os.path.join(BASE, out))
