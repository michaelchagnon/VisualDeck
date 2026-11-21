/*{
  "DESCRIPTION": "Inverts the colors; Amount blends from the picture to its negative.",
  "CREDIT": "VisualDeck",
  "ISFVSN": "2",
  "CATEGORIES": ["Color"],
  "INPUTS": [
    {"NAME": "inputImage", "TYPE": "image"},
    {"NAME": "amount", "LABEL": "Amount", "TYPE": "float", "DEFAULT": 1.0, "MIN": 0.0, "MAX": 1.0}
  ]
}*/
// SPDX-License-Identifier: MPL-2.0
void main() {
    vec4 c = IMG_THIS_NORM_PIXEL(inputImage);
    gl_FragColor = vec4(mix(c.rgb, 1.0 - c.rgb, amount), c.a);
}
