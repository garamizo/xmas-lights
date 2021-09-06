#include <FastLED.h>

#define NUM_LED_PER_STRIP 100
#define NUM_STRIPS 1
#define NUM_LEDS (NUM_LED_PER_STRIP * NUM_STRIPS)
#define DATA_PIN 23
#define BRIGHTNESS  85  // range: 0-255

// Define the array of leds
CRGB leds[NUM_LEDS];

void setup() {
  FastLED.addLeds<NUM_STRIPS, WS2811, DATA_PIN, RGB>(leds, NUM_LED_PER_STRIP);
  FastLED.setBrightness(BRIGHTNESS);

  // turn on all lights
  for (int i = 0; i < NUM_LEDS; i++)
    leds[i] = CRGB::Yellow;
  FastLED.show();
  delay(200);
}

void loop() {

  static int dt_ms = 2000.0 / BRIGHTNESS;
  
  // turn on all lights
  for (int i = 0; i < BRIGHTNESS; i++) {
    FastLED.setBrightness(i);
    FastLED.show();
    delay(dt_ms);
  }

  // turn on all lights
  for (int i = BRIGHTNESS; i > 0; i--) {
    FastLED.setBrightness(i);
    FastLED.show();
    delay(dt_ms);
  }
  

}
