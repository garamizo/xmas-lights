#include <FastLED.h>

#define NUM_LED_PER_STRIP 50

//#define NUM_STRIPS 9  // Knops
//#define DATA_PIN 5   // Knops
//#define BRIGHTNESS  85  // range: 0-255

#define NUM_STRIPS 6  // Ribeiros
#define DATA_PIN 13  // Ribeiros
#define BRIGHTNESS  255  // range: 0-255

#define ENCODER_BASE 3
#define ENCODER_DELAY_MS 500

const int NUM_LEDS = NUM_LED_PER_STRIP * NUM_STRIPS;
const int ENCODER_NUM_DIGITS = ceil(log(NUM_LEDS) / log(ENCODER_BASE));

// Define the array of leds
CRGB leds[NUM_LEDS];

void setup() {
  FastLED.addLeds<WS2811, DATA_PIN, RGB>(leds, NUM_LEDS).setCorrection( TypicalLEDStrip );
  FastLED.setBrightness(BRIGHTNESS);
}

void loop()
{
  for (int i = 0; i < NUM_LEDS; i++)
    leds[i] = CHSV(0, 0, 0);
  FastLED.show(); 
  delay(ENCODER_DELAY_MS);

  for (int digit = 1; digit <= ENCODER_NUM_DIGITS; digit++) {
    for (int i = 0; i < NUM_LEDS; i++)
    {
      int val = i;
      for (int d = 1; d < digit; d++)
        val = (val / ENCODER_BASE);
      val = val % ENCODER_BASE;
        
      leds[i] = CHSV( round(255 * (val + 0.0) / ENCODER_BASE), 255, BRIGHTNESS);
    }
    FastLED.show(); 
    delay(ENCODER_DELAY_MS);
  }
}
