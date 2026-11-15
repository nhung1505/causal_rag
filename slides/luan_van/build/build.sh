#!/bin/zsh
# Biên dịch, kiểm tra log, render PNG và kiểm tra bố cục.
# Chạy từ thư mục slides/luan_van:  zsh build/build.sh
cd "${0:A:h}/.." || exit 1
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build/latexmk.out 2>&1
rc=$?
echo "latexmk exit=$rc"
if [ $rc -ne 0 ]; then
  grep -nE "^! " -A4 main.log | head -30
  exit $rc
fi
echo "--- log problems (errors, overfull vbox, overfull hbox > 3pt, undefined, missing char):"
grep -nE "^! |Overfull \\\\vbox|Undefined control sequence|Missing character|undefined" main.log | head -20
grep -nE "Overfull \\\\hbox \(([3-9]|[1-9][0-9]+)\." main.log | head -20
mkdir -p build/render && rm -f build/render/*.png(N)
gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r203.2 -o build/render/p%02d.png main.pdf
echo "pages: $(ls build/render/*.png | wc -l | tr -d ' ')"
/usr/bin/python3 build/check_layout.py
