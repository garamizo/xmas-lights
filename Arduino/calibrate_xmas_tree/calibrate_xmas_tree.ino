#include <FastLED.h>

#define NUM_LED_PER_STRIP 50
#define NUM_STRIPS 9
#define NUM_LEDS (NUM_LED_PER_STRIP * NUM_STRIPS)
#define DATA_PIN 5
#define BRIGHTNESS  85  // range: 0-255

// Define the array of leds
CRGB leds[NUM_LEDS];

void setup() {
  FastLED.addLeds<WS2811, DATA_PIN, RGB>(leds, NUM_LEDS).setCorrection( TypicalLEDStrip );
//  FastLED.addLeds<NUM_STRIPS, WS2811, DATA_PIN, RGB>(leds, NUM_LED_PER_STRIP);
  FastLED.setBrightness(BRIGHTNESS);
}

void loop() {
  unsigned long t0 = micros();
  
  // turn on all lights
  for (int i = 0; i < NUM_LEDS; i++)
    leds[i] = CRGB::Green;
  FastLED.show();
//  delay(200);
  t0 += 100000;
  while (micros() < t0);

  // turn on all lights
  for (int i = 0; i < NUM_LEDS; i++)
    leds[i] = CRGB::Black;
  FastLED.show();
//  delay(200);
  t0 += 100000;
  while (micros() < t0);
  
  for (int i = 0; i < NUM_LEDS; i++) {

    leds[i] = CRGB::Green;
    FastLED.show();
    t0 += 100000;
    while (micros() < t0);
    
    leds[i] = CRGB::Black;
    FastLED.show();
    t0 += 100000;
    while (micros() < t0);
  }

}
