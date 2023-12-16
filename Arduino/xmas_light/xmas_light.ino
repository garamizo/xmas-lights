#include <FastLED.h>
#include "coordinates.h"

#define LED_PIN     13  // LED strip control pin
#define NUM_LEDS    300  // 9 strips, 50 bulbs each
#define BRIGHTNESS  128  // range: 0-255
CRGB leds[NUM_LEDS];

const float height = 2.40;  // tree height
const float diameter = 1.51;  // tree base diameter
uint8_t D[NUM_LEDS];

long int count = 0;

float radial[NUM_LEDS], angular[NUM_LEDS], I[NUM_LEDS], J[NUM_LEDS], splash[NUM_LEDS], ray[NUM_LEDS];  // radial, angular


float mapfloat(float x, float in_min, float in_max, float out_min, float out_max)
{
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

float interp1d(float* x, float* y, float xx, int size) {
  int i;
  for (i = 0; i < size - 1 && xx > x[i]; i++);
  return (mapfloat(xx, x[i], x[i + 1], y[i], y[i + 1]));

}

void setup() {
  delay( 3000 ); // power-up safety delay
  FastLED.addLeds<WS2811, LED_PIN, RGB>(leds, NUM_LEDS).setCorrection( TypicalLEDStrip );
  FastLED.setBrightness(  BRIGHTNESS );

  Serial.begin(9600);

  // precompute cylindrical and projective coordinates
  const float ztop = 1.0,
              qwide = 120 * PI / 180.0,
              ray_center = 0.8;
  for (int k = 0; k < NUM_LEDS; k++) {
    radial[k] = sqrt(cartesian[k][0] * cartesian[k][0] + cartesian[k][1] * cartesian[k][1]);
    angular[k] = atan2(cartesian[k][1], cartesian[k][0]);
    I[k] = 1 - cartesian[k][2] / ztop;
    J[k] = angular[k] / qwide + 0.5;
    float zmiddle = cartesian[k][2] - ray_center;
    splash[k] = sqrt(cartesian[k][1] * cartesian[k][1] + zmiddle * zmiddle);
    ray[k] = atan2(cartesian[k][1], zmiddle);
  }

  //    for (int i = 0; i < 50; i++)
  //      Serial.println(interp1d(colormap_x, colormap_hsv[0], i * 255.0/50, 50));
}


void loop() {
  const float Kt = PI;  // for meaning of these constants, see measure_bulb.m
  const float Kq = 3;
  const float Kz = 4 * PI / height;
  const float Krr = -2 * PI / 50e-2;
  const float Kqq = 360 / 90.0;



  // change pattern =================================
  static int pattern_mode = 0;
  static long t1 = millis();
  const long dt1 = 15000;  // pattern period
  if (millis() - t1 > dt1) {
    t1 = millis();
    pattern_mode = (++count) % 4;
  }

  // change colormap ===============================
  static int colormap_mode = 0;
  static long t2 = millis();
  const long dt2 = 150000;  // pattern period
  if (millis() - t2 > dt2) {
    t2 = millis();
    colormap_mode = (colormap_mode + 1) % 2;
  }

  // update bulb color =============================
  static long t0 = millis();
  const long dt0 = 50;
  if (millis() - t0 > dt0) {
    t0 = millis();

    for (int i = 0; i < NUM_LEDS; i++) {
      float time = millis() / 1000.0;
      float B = 0.0;

      if (pattern_mode == 0)
        B = sin(Kt * time + Kq * angular[i]); // radial
      else if (pattern_mode == 1)
        B = sin(Kt * time + Kz * cartesian[i][2]); // falling rings
      else if (pattern_mode == 2)
        B = sin(Kt * time + Krr * splash[i]); // circles
      else if (pattern_mode == 3)
        B = sin(Kt * time + Kqq * ray[i]); // rays
      B = constrain(B, 0.0, 1.0);
//      B = constrain(mapfloat(B, -0.5, 1, 0, 1), 0.0, 1.0);
      uint8_t val = (1 - B) * 255.0;

      uint8_t hue;
      if (colormap_mode == 0) {
        const float Krc = 255 / (2 * PI), Ktc = 255 / 10.0;
        hue = Krc * angular[i] + Ktc * time;
      } else {
        hue = 50;  // golden
      }

      leds[i] = CHSV(hue, 255, val);

      //      leds[i] = CRGB(interp1d(colormap_x, colormap_hsv[0], hue, 50) * B, \
      //                     interp1d(colormap_x, colormap_hsv[1], hue, 50) * B, \
      //                     interp1d(colormap_x, colormap_hsv[2], hue, 50) * B);  // too slow


    }
    FastLED.show();
  }
  FastLED.delay(1);
}
