# Fonts

Both are under the SIL Open Font License 1.1 (the license texts are next to them).

- `archivo.woff2`: Archivo (headings at width 87.5, text at width 100), limited to width 87.5–100 and
  weight 400–800 and subset to Latin, from google/fonts `ofl/archivo/Archivo[wdth,wght].ttf`.
- `visualdeck-mono-400.woff2`, `visualdeck-mono-500.woff2`: IBM Plex Mono Regular and Medium, subset to
  Latin. Because the subset is a Modified Version and "Plex" is a Reserved Font Name, the subset's
  family name is changed to "VisualDeck Mono" (OFL condition 3); the copyright notice is unchanged.

To remake them: download the sources, limit Archivo with `fontTools.varLib.instancer`
(`{"wdth": (87.5, 100), "wght": (400, 800)}`), subset each with `pyftsubset --flavor=woff2` to
U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+20AC, U+2122,
U+2190-2199, U+2212, U+2215, U+2713, U+2715, U+25B6, U+25C0, and rename the Plex subset's name records.
