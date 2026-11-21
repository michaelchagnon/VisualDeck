/*{
  "DESCRIPTION": "Colors the picture with one color, keeping its light and dark.",
  "CREDIT": "VisualDeck",
  "ISFVSN": "2",
  "CATEGORIES": ["Color"],
  "INPUTS": [
    {"NAME": "inputImage", "TYPE": "image"},
    {"NAME": "tint", "LABEL": "Color", "TYPE": "color", "DEFAULT": [1.0, 0.55, 0.2, 1.0]},
    {"NAME": "amount", "LABEL": "Amount", "TYPE": "float", "DEFAULT": 1.0, "MIN": 0.0, "MAX": 1.0}
  ]
}*/
// SPDX-License-Identifier: MPL-2.0
void main() {
    vec4 c = IMG_THIS_NORM_PIXEL(inputImage);
    float y = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
    gl_FragColor = vec4(mix(c.rgb, y * tint.rgb, amount), c.a);
}
