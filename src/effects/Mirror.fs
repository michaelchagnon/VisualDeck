/*{
  "DESCRIPTION": "Mirrors one half of the picture onto the other.",
  "CREDIT": "VisualDeck",
  "ISFVSN": "2",
  "CATEGORIES": ["Geometry"],
  "INPUTS": [
    {"NAME": "inputImage", "TYPE": "image"},
    {"NAME": "direction", "LABEL": "Keep", "TYPE": "long", "DEFAULT": 0, "VALUES": [0, 1, 2, 3],
     "LABELS": ["Left half", "Right half", "Top half", "Bottom half"]}
  ]
}*/
// SPDX-License-Identifier: MPL-2.0
void main() {
    vec2 c = isf_FragNormCoord;
    if (direction == 0) c.x = c.x > 0.5 ? 1.0 - c.x : c.x;
    else if (direction == 1) c.x = c.x < 0.5 ? 1.0 - c.x : c.x;
    else if (direction == 2) c.y = c.y < 0.5 ? 1.0 - c.y : c.y;
    else c.y = c.y > 0.5 ? 1.0 - c.y : c.y;
    gl_FragColor = IMG_NORM_PIXEL(inputImage, c);
}
