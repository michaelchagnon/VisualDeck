/*{
  "DESCRIPTION": "Softens the picture: a Gaussian blur, across and then down (two passes).",
  "CREDIT": "VisualDeck",
  "ISFVSN": "2",
  "CATEGORIES": ["Blur"],
  "INPUTS": [
    {"NAME": "inputImage", "TYPE": "image"},
    {"NAME": "radius", "LABEL": "Radius (px)", "TYPE": "float", "DEFAULT": 8.0, "MIN": 0.0, "MAX": 50.0}
  ],
  "PASSES": [{"TARGET": "across"}, {}]
}*/
// SPDX-License-Identifier: MPL-2.0
void main() {
    vec2 step = (PASSINDEX == 0 ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) / RENDERSIZE;
    float r = max(radius, 0.0);
    float sigma = max(r / 2.0, 0.001);
    vec4 sum = vec4(0.0);
    float total = 0.0;
    for (int i = -24; i <= 24; i++) {
        float x = float(i) * r / 24.0;           // 49 taps spread over -radius..radius
        float w = exp(-0.5 * x * x / (sigma * sigma));
        vec2 c = isf_FragNormCoord + step * x;
        sum += w * (PASSINDEX == 0 ? IMG_NORM_PIXEL(inputImage, c) : IMG_NORM_PIXEL(across, c));
        total += w;
    }
    gl_FragColor = sum / total;
}
