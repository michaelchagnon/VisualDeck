/*{
  "DESCRIPTION": "Leaves fading trails behind what moves: each frame keeps part of the frame before.",
  "CREDIT": "VisualDeck",
  "ISFVSN": "2",
  "CATEGORIES": ["Feedback"],
  "INPUTS": [
    {"NAME": "inputImage", "TYPE": "image"},
    {"NAME": "keep", "LABEL": "Keep", "TYPE": "float", "DEFAULT": 0.85, "MIN": 0.0, "MAX": 0.98}
  ],
  "PASSES": [{"TARGET": "trail", "PERSISTENT": true}, {}]
}*/
// SPDX-License-Identifier: MPL-2.0
void main() {
    vec4 now = IMG_THIS_NORM_PIXEL(inputImage);
    if (PASSINDEX == 0)
        gl_FragColor = max(now, IMG_THIS_NORM_PIXEL(trail) * keep);
    else
        gl_FragColor = IMG_THIS_NORM_PIXEL(trail);
}
