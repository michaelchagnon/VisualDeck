"""Make website/static/og.png (1200x630) from docs/images/main-window.png and the site's fonts (TTF sources)."""
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
F = sys.argv[1]; ROOT = sys.argv[2]; OUT = sys.argv[3]
W, H = 1200, 630
im = Image.new("RGB", (W, H), "#ffffff")
d = ImageDraw.Draw(im)
# stage band on the right with the screenshot
shot = Image.open(f"{ROOT}/docs/images/main-window.png").convert("RGB").crop((300, 0, 1314, 883))
band_x = 560
d.rectangle([band_x, 0, W, H], fill="#101112")
sh_h = 520; sh_w = round(shot.width * sh_h / shot.height)
shot = shot.resize((sh_w, sh_h), Image.LANCZOS)
im.paste(shot, (W - sh_w - 40, (H - sh_h) // 2))
# text
head = ImageFont.truetype(f"{F}/archivo-limited.ttf", 76)
try:
    head.set_variation_by_axes([760, 87.5])
except Exception as e:
    print("axes", e, head.get_variation_axes())
sub = ImageFont.truetype(f"{F}/archivo-limited.ttf", 34)
sub.set_variation_by_axes([650, 87.5])
mono = ImageFont.truetype(f"{F}/ibmplexmono__IBMPlexMono-Regular.ttf", 21)
icon = Image.open(f"{ROOT}/src/visualdeck.ico").convert("RGBA").resize((64, 64), Image.LANCZOS)
x = 56
im.paste(icon, (x, 70), icon)
d.text((x, 160), "VisualDeck", font=head, fill="#121314")
y = 262
for line in ("Video cues for live", "shows, on Windows."):
    d.text((x, y), line, font=sub, fill="#3b3f43"); y += 44
d.text((x, 520), "Free and open source", font=mono, fill="#5f6368")
d.text((x, 550), "visualdeck.org", font=mono, fill="#121314")
im.save(OUT, optimize=True)
print(im.size)
