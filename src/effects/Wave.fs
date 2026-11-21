/*{
  "DESCRIPTION": "A moving ripple across the picture.",
  "CREDIT": "VisualDeck",
  "ISFVSN": "2",
  "CATEGORIES": ["Distortion"],
  "INPUTS": [
    {"NAME": "inputImage", "TYPE": "image"},
    {"NAME": "strength", "LABEL": "Strength", "TYPE": "float", "DEFAULT": 0.01, "MIN": 0.0, "MAX": 0.08},
    {"NAME": "waves", "LABEL": "Waves", "TYPE": "float", "DEFAULT": 6.0, "MIN": 1.0, "MAX": 40.0},
    {"NAME": "speed", "LABEL": "Speed", "TYPE": "float", "DEFAULT": 1.0, "MIN": 0.0, "MAX": 5.0},
    {"NAME": "vertical", "LABEL": "Up and down", "TYPE": "bool", "DEFAULT": false}
  ]
}*/
// SPDX-License-Identifier: MPL-2.0
void main() {
    vec2 c = isf_FragNormCoord;
    float t = TIME * speed * 6.2831853;
    if (vertical) c.y += strength * sin(c.x * waves * 6.2831853 + t);
    else c.x += strength * sin(c.y * waves * 6.2831853 + t);
    gl_FragColor = IMG_NORM_PIXEL(inputImage, clamp(c, 0.0, 1.0));
}
