/*{
  "DESCRIPTION": "Large square pixels; Size is their width as a share of the picture's width.",
  "CREDIT": "VisualDeck",
  "ISFVSN": "2",
  "CATEGORIES": ["Stylize"],
  "INPUTS": [
    {"NAME": "inputImage", "TYPE": "image"},
    {"NAME": "size", "LABEL": "Size", "TYPE": "float", "DEFAULT": 0.05, "MIN": 0.002, "MAX": 0.25}
  ]
}*/
// SPDX-License-Identifier: MPL-2.0
void main() {
    vec2 cell = vec2(size, size * RENDERSIZE.x / RENDERSIZE.y);
    vec2 c = (floor(isf_FragNormCoord / cell) + 0.5) * cell;
    gl_FragColor = IMG_NORM_PIXEL(inputImage, clamp(c, 0.0, 1.0));
}
