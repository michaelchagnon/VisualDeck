/*{
  "DESCRIPTION": "Shifts the red and blue channels apart.",
  "CREDIT": "VisualDeck",
  "ISFVSN": "2",
  "CATEGORIES": ["Color", "Glitch"],
  "INPUTS": [
    {"NAME": "inputImage", "TYPE": "image"},
    {"NAME": "shift", "LABEL": "Distance", "TYPE": "float", "DEFAULT": 0.01, "MIN": 0.0, "MAX": 0.1},
    {"NAME": "angle", "LABEL": "Angle", "TYPE": "float", "DEFAULT": 0.0, "MIN": 0.0, "MAX": 6.2832}
  ]
}*/
// SPDX-License-Identifier: MPL-2.0
void main() {
    vec2 d = shift * vec2(cos(angle), sin(angle));
    vec4 c = IMG_THIS_NORM_PIXEL(inputImage);
    float r = IMG_NORM_PIXEL(inputImage, clamp(isf_FragNormCoord + d, 0.0, 1.0)).r;
    float b = IMG_NORM_PIXEL(inputImage, clamp(isf_FragNormCoord - d, 0.0, 1.0)).b;
    gl_FragColor = vec4(r, c.g, b, c.a);
}
