#include <FastLED.h>
#include "coordinates.h"  // cartesian
#include "credentials.h"  // WIFI_SSID, WIFI_PASS, APP_KEY, APP_SECRET, LIGHT_ID

#define LED_PIN     13  // LED strip control pin
#define NUM_LEDS    300  // 9 strips, 50 bulbs each
#define BRIGHTNESS  128  // range: 0-255

#ifdef ENABLE_DEBUG
  #define DEBUG_ESP_PORT Serial
  #define NODEBUG_WEBSOCKETS
  #define NDEBUG
#endif 

#include <Arduino.h>
#if defined(ESP8266)
  #include <ESP8266WiFi.h>
#elif defined(ESP32) || defined(ARDUINO_ARCH_RP2040)
  #include <WiFi.h>
#endif

#include "SinricPro.h"
#include "SinricProLight.h"

#define BAUD_RATE         115200                // Change baudrate to your need

CRGB leds[NUM_LEDS];

float mapfloat(float x, float in_min, float in_max, float out_min, float out_max)
{
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}


// we use a struct to store all states and values for our light
struct {
  bool powerState = false;
  int brightness = 0;
} device_state; 

bool onPowerState(const String &deviceId, bool &state) {
  Serial.printf("Device %s power turned %s \r\n", deviceId.c_str(), state?"on":"off");
  device_state.powerState = state;
  if (state) {
    FastLED.setBrightness(map(device_state.brightness, 0, 100, 0, 255));
  } else {
    FastLED.setBrightness(0);
  }
  FastLED.show();
  
  return true; // request handled properly
}

bool onBrightness(const String &deviceId, int &brightness) {
  device_state.brightness = brightness;
  if (device_state.brightness > 0)
    device_state.powerState = true;
  else
    device_state.powerState = false;
  Serial.printf("Device %s brightness level changed to %d\r\n", deviceId.c_str(), brightness);
  FastLED.setBrightness(map(device_state.brightness, 0, 100, 0, 255));
  FastLED.show();
  return true;
}

bool onAdjustBrightness(const String &deviceId, int brightnessDelta) {
  device_state.brightness += brightnessDelta;
  Serial.printf("Device %s brightness level changed about %i to %d\r\n", deviceId.c_str(), brightnessDelta, device_state.brightness);
  brightnessDelta = device_state.brightness;
  if (device_state.brightness > 0)
    device_state.powerState = true;
  else
    device_state.powerState = false;
  FastLED.setBrightness(map(device_state.brightness, 0, 100, 0, 255));
  FastLED.show();
  return true;
}

void setupWiFi() {
  Serial.printf("\r\n[Wifi]: Connecting");

  #if defined(ESP8266)
    WiFi.setSleepMode(WIFI_NONE_SLEEP); 
    WiFi.setAutoReconnect(true);
  #elif defined(ESP32)
    WiFi.setSleep(false); 
    WiFi.setAutoReconnect(true);
  #endif

  WiFi.begin(WIFI_SSID, WIFI_PASS); 

  while (WiFi.status() != WL_CONNECTED) {
    Serial.printf(".");
    delay(250);
  }
  IPAddress localIP = WiFi.localIP();
  Serial.printf("connected!\r\n[WiFi]: IP-Address is %d.%d.%d.%d\r\n", localIP[0], localIP[1], localIP[2], localIP[3]);
}

void setupSinricPro() {
  // get a new Light device from SinricPro
  SinricProLight &myLight = SinricPro[LIGHT_ID];

  // set callback function to device
  myLight.onPowerState(onPowerState);
  myLight.onBrightness(onBrightness);
  myLight.onAdjustBrightness(onAdjustBrightness);

  // setup SinricPro
  SinricPro.onConnected([](){ Serial.printf("Connected to SinricPro\r\n"); }); 
  SinricPro.onDisconnected([](){ Serial.printf("Disconnected from SinricPro\r\n"); });
  SinricPro.restoreDeviceStates(true); // Uncomment to restore the last known state from the server.
  SinricPro.begin(APP_KEY, APP_SECRET);
}

void setupLed()
{
  FastLED.addLeds<WS2811, LED_PIN, RGB>(leds, NUM_LEDS).setCorrection( TypicalLEDStrip );
  FastLED.setBrightness(  BRIGHTNESS );
}


void LedHandle()
{
  uint32_t t = millis();
  
 static ExplosionPattern explosion(t, true);
 if (!explosion.update(t))
   explosion.reset(t);

 static ExplosionPattern explosion2(t, true);
 if (!explosion2.update(t))
   explosion2.reset(t);

  static LinearPattern pattern(t, 128);
//  static AngularPattern pattern(t, 128);
  if (!pattern.update(t))
  {
    pattern.reset(t, 128);
//    Serial.println(pattern.msg);
  }

  for (int i = 0; i < NUM_LEDS; i++)
  {
    uint8_t hue = 0, sat = 255, val = 0;
    
    pattern.get_hsv(i, hue, sat, val);
    explosion.get_hsv(i, hue, sat, val);
//    explosion2.get_hsv(i, hue, sat, val);

    leds[i] = CHSV(hue, sat, val);
  }
  FastLED.show();

}


// main setup function
void setup() {
  Serial.begin(BAUD_RATE); Serial.printf("\r\n\r\n");
  setupWiFi();
  setupSinricPro();
  setupLed();
}

void loop() {
  SinricPro.handle();

  if (device_state.powerState == true)
    LedHandle();
  delay(1);
}
