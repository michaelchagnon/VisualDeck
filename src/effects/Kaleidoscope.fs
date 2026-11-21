/*{
  "DESCRIPTION": "Repeats a wedge of the picture around its center; Turn rotates it.",
  "CREDIT": "VisualDeck",
  "ISFVSN": "2",
  "CATEGORIES": ["Geometry"],
  "INPUTS": [
    {"NAME": "inputImage", "TYPE": "image"},
    {"NAME": "sides", "LABEL": "Sides", "TYPE": "float", "DEFAULT": 6.0, "MIN": 2.0, "MAX": 16.0},
    {"NAME": "turn", "LABEL": "Turn", "TYPE": "float", "DEFAULT": 0.0, "MIN": 0.0, "MAX": 6.2832},
    {"NAME": "center", "LABEL": "Center", "TYPE": "point2D", "DEFAULT": [0.5, 0.5], "MIN": [0.0, 0.0], "MAX": [1.0, 1.0]}
  ]
}*/
// SPDX-License-Identifier: MPL-2.0
void main() {
    float aspect = RENDERSIZE.x / RENDERSIZE.y;
    vec2 p = isf_FragNormCoord - center;
    p.x *= aspect;
    float r = length(p);
    float a = atan(p.y, p.x) + turn;
    float wedge = 6.2831853 / floor(sides);
    a = mod(a, wedge);
    a = abs(a - wedge * 0.5);
    vec2 q = r * vec2(cos(a), sin(a));
    q.x /= aspect;
    // Past the picture's edge the picture is mirrored, not its last row smeared out
    // (the far corners of a wide picture reach outside it).
    vec2 s = center + q;
    s = 1.0 - abs(1.0 - mod(s, 2.0));
    gl_FragColor = IMG_NORM_PIXEL(inputImage, s);
}
