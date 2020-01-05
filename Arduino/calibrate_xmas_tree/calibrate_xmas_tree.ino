#include <FastLED.h>

#define NUM_LEDS 50
#define DATA_PIN 25
#define BRIGHTNESS  8  // range: 0-255

// Define the array of leds
CRGB leds[NUM_LEDS];

void setup() {
  FastLED.addLeds<WS2811, DATA_PIN, RGB>(leds, NUM_LEDS);
  FastLED.setBrightness(BRIGHTNESS);
}

void loop() {
  // turn on all lights
  for (int i = 0; i < NUM_LEDS; i++)
    leds[i] = CRGB::White;
  FastLED.show();
  delay(200);

  // turn on all lights
  for (int i = 0; i < NUM_LEDS; i++)
    leds[i] = CRGB::Black;
  FastLED.show();
  delay(200);
  
  unsigned long t0 = micros();
  for (int i = 0; i < NUM_LEDS; i++) {

    leds[i] = CRGB::White;
    FastLED.show();
    t0 += 100000;
    while (micros() < t0);
    
    leds[i] = CRGB::Black;
    FastLED.show();
    t0 += 100000;
    while (micros() < t0);
  }
  
  FastLED.show();
  delay(1000);
}
