#define NUM_LEDS    400  // 8 strips, 50 bulbs each

const float height = 105;  // tree height
const float diameter = 72;  // tree base diameter


const int8_t cartesian[NUM_LEDS][3] = {
    {1, 8, 48},
    {22, -1, 4},
    {26, -5, 9},
    {22, -10, 18},
    {23, -15, 14},
    {26, -6, 9},
    {24, -10, 2},
    {19, -19, 0},
    {24, -11, 4},
    {13, -16, 9},
    {14, -17, 10},
    {5, -12, 9},
    {19, -23, 33},
    {12, -23, 9},
    {9, -23, 9},
    {10, -22, 5},
    {7, -31, 5},
    {8, -24, 7},
    {3, -27, 7},
    {0, -35, 8},
    {-1, -29, 8},
    {9, -30, 10},
    {5, -22, 13},
    {-3, -23, 15},
    {5, -14, 13},
    {-5, -15, 16},
    {0, -21, 12},
    {1, -26, 12},
    {-5, -20, 7},
    {-5, -21, 6},
    {-2, -12, 17},
    {-4, -15, 12},
    {-12, -22, 9},
    {-14, -12, 13},
    {-19, -16, 16},
    {-29, -18, 8},
    {-24, -16, 11},
    {-20, -16, 12},
    {-20, -14, 11},
    {-11, -8, 15},
    {-25, -3, 14},
    {-25, -3, 14},
    {-30, -8, 8},
    {-25, -7, 8},
    {-28, 4, 11},
    {-20, 0, 16},
    {-17, 0, 9},
    {-14, 8, 17},
    {-5, -2, 14},
    {-3, 6, 11},
    {-12, 21, 5},
    {-12, 18, 13},
    {-15, 19, 24},
    {-19, 25, 16},
    {-24, 25, 10},
    {-17, 11, 16},
    {-25, 22, 6},
    {-25, 22, 8},
    {-18, 15, 7},
    {-23, 15, 8},
    {-7, 11, 8},
    {-8, 14, 11},
    {-9, 16, 15},
    {3, 11, 13},
    {-11, 20, 12},
    {-4, 20, 11},
    {-8, 34, 14},
    {-13, 34, 3},
    {-2, 33, 0},
    {-6, 36, 5},
    {-1, 29, 6},
    {-5, 31, 2},
    {7, 31, 8},
    {2, 28, 5},
    {6, 23, 4},
    {-2, 25, 5},
    {0, 18, 5},
    {2, 11, 4},
    {-8, 21, 8},
    {-17, 27, 8},
    {8, 6, 7},
    {16, 23, 0},
    {23, 30, 5},
    {22, 22, 7},
    {16, 19, 7},
    {19, 22, 14},
    {12, 14, 16},
    {22, 18, 16},
    {26, 22, 7},
    {26, 17, 2},
    {29, 14, 3},
    {21, 18, 12},
    {22, 6, 8},
    {16, 7, 8},
    {22, 6, 3},
    {30, -5, 6},
    {31, -1, 3},
    {19, -3, 7},
    {27, -5, 0},
    {24, -1, 10},
    {24, -17, 10},
    {17, -12, 14},
    {16, -7, 16},
    {12, -13, 21},
    {8, -12, 20},
    {9, -23, 17},
    {19, -28, 18},
    {16, -25, 24},
    {19, -23, 21},
    {21, -32, 20},
    {25, -21, 29},
    {18, -20, 36},
    {14, -28, 27},
    {8, -28, 34},
    {3, -26, 28},
    {-2, -26, 25},
    {-3, -25, 24},
    {-10, -36, 21},
    {-9, -30, 21},
    {-5, -24, 24},
    {-8, -21, 36},
    {-10, -24, 32},
    {-15, -28, 20},
    {-14, -32, 18},
    {-4, -33, 21},
    {-11, -25, 16},
    {-17, -20, 20},
    {-10, -29, 25},
    {-7, -29, 16},
    {-16, -33, 24},
    {-22, -27, 21},
    {-16, -19, 17},
    {-19, -11, 20},
    {-17, -9, 29},
    {-10, -21, 19},
    {-20, -24, 30},
    {-24, -17, 22},
    {-25, -8, 27},
    {-20, -17, 30},
    {-28, -13, 22},
    {-26, -6, 19},
    {-19, -5, 20},
    {-29, 3, 9},
    {-32, 5, 18},
    {-26, -1, 27},
    {-34, 2, 21},
    {-29, -5, 21},
    {-33, 6, 20},
    {-28, -5, 14},
    {-31, 5, 7},
    {-24, 13, 16},
    {-13, 12, 21},
    {-16, 9, 8},
    {-13, 10, 32},
    {-15, 5, 32},
    {-26, 4, 33},
    {-28, 8, 25},
    {-29, 3, 29},
    {-22, 17, 20},
    {-21, 23, 24},
    {-16, 18, 25},
    {-11, 17, 25},
    {-6, 15, 25},
    {0, 19, 24},
    {1, 25, 22},
    {-8, 26, 24},
    {-12, 26, 24},
    {-8, 20, 31},
    {-3, 23, 30},
    {3, 23, 26},
    {1, 21, 19},
    {9, 18, 16},
    {7, 25, 12},
    {-4, 28, 11},
    {-5, 32, 19},
    {5, 26, 20},
    {6, 34, 22},
    {12, 30, 21},
    {12, 27, 16},
    {14, 24, 20},
    {10, 22, 25},
    {12, 13, 33},
    {7, 9, 33},
    {18, 5, 31},
    {17, 11, 26},
    {16, 17, 21},
    {16, 20, 28},
    {16, 14, 36},
    {19, 15, 27},
    {22, 17, 26},
    {23, 17, 19},
    {20, 21, 23},
    {23, 18, 21},
    {25, 14, 29},
    {19, -18, 35},
    {19, -1, 21},
    {28, 1, 16},
    {25, 10, 12},
    {28, 16, 14},
    {24, 7, 22},
    {21, -3, 34},
    {17, -6, 34},
    {14, -9, 33},
    {17, -18, 35},
    {19, -18, 28},
    {18, -11, 21},
    {25, -5, 22},
    {21, -9, 25},
    {18, -13, 28},
    {14, -16, 31},
    {11, -20, 34},
    {7, -9, 36},
    {-2, -6, 38},
    {5, -14, 32},
    {13, -21, 33},
    {16, -27, 39},
    {10, -22, 36},
    {4, -16, 33},
    {4, -16, 35},
    {-2, -17, 38},
    {-6, -14, 41},
    {2, -8, 34},
    {-18, -16, 37},
    {-8, -19, 38},
    {-2, -19, 32},
    {-8, -26, 35},
    {-8, -15, 40},
    {-18, -18, 33},
    {-18, -17, 37},
    {-17, -16, 41},
    {-6, -12, 45},
    {-16, -9, 37},
    {-21, -6, 40},
    {-25, -3, 42},
    {-15, 5, 41},
    {-21, 7, 41},
    {-26, 5, 34},
    {-25, 0, 29},
    {-24, -2, 34},
    {-22, 0, 42},
    {-30, 4, 35},
    {-32, 15, 34},
    {-21, 14, 31},
    {-19, 7, 32},
    {-16, 10, 40},
    {-9, 12, 43},
    {-18, 22, 40},
    {-25, 18, 34},
    {-14, 12, 31},
    {-13, 24, 31},
    {-2, 23, 31},
    {-3, 14, 38},
    {-2, 15, 40},
    {3, 12, 48},
    {7, 10, 48},
    {11, 11, 39},
    {11, 10, 38},
    {15, 12, 38},
    {8, 23, 36},
    {3, 24, 37},
    {11, 21, 42},
    {12, 17, 33},
    {14, 21, 29},
    {10, 25, 31},
    {7, 21, 37},
    {14, 14, 39},
    {16, 12, 45},
    {13, 6, 41},
    {10, 1, 36},
    {-6, -4, 40},
    {23, -4, 35},
    {23, 7, 32},
    {25, 1, 35},
    {25, -3, 37},
    {17, -5, 37},
    {15, -11, 39},
    {12, -14, 40},
    {8, -16, 41},
    {18, -15, 36},
    {12, -12, 41},
    {10, -8, 42},
    {8, -12, 54},
    {1, -17, 52},
    {8, -17, 42},
    {5, -22, 44},
    {2, -18, 46},
    {-3, -13, 50},
    {-1, -18, 45},
    {-7, -18, 39},
    {-1, -23, 42},
    {-8, -21, 43},
    {-7, -15, 48},
    {-11, -11, 56},
    {-5, -10, 42},
    {-14, -8, 51},
    {-9, 0, 46},
    {-18, 2, 49},
    {-16, -4, 41},
    {-19, -3, 45},
    {-21, 2, 47},
    {-5, 3, 56},
    {-6, 10, 57},
    {-12, 11, 53},
    {-17, 10, 56},
    {-10, 11, 52},
    {-10, 10, 52},
    {-3, 14, 52},
    {-10, 20, 48},
    {-13, 16, 54},
    {-6, 11, 58},
    {1, 12, 60},
    {4, 17, 57},
    {-7, 18, 53},
    {1, 21, 62},
    {2, 13, 59},
    {9, 13, 63},
    {10, 12, 55},
    {10, 17, 61},
    {8, 7, 57},
    {10, 0, 58},
    {18, 6, 54},
    {11, 14, 51},
    {13, 10, 54},
    {14, 3, 51},
    {18, -2, 54},
    {13, -4, 44},
    {17, -2, 44},
    {22, -6, 52},
    {17, -8, 58},
    {9, -2, 56},
    {8, -13, 57},
    {8, -14, 52},
    {17, -11, 48},
    {14, -8, 55},
    {9, -16, 52},
    {5, -8, 53},
    {3, -6, 60},
    {-8, -8, 57},
    {-8, -14, 57},
    {-3, -20, 60},
    {-6, -15, 65},
    {-11, -14, 58},
    {-10, -7, 51},
    {-16, -14, 48},
    {-15, -13, 58},
    {-10, -5, 63},
    {-19, -1, 60},
    {-17, -4, 70},
    {-13, 2, 64},
    {-10, 6, 65},
    {-13, 4, 71},
    {3, 6, 61},
    {-7, 7, 67},
    {-5, 13, 79},
    {-4, 19, 68},
    {3, 13, 64},
    {3, 19, 60},
    {-4, 23, 72},
    {-2, 16, 74},
    {9, 12, 73},
    {6, 8, 74},
    {10, 9, 74},
    {16, 8, 74},
    {11, 3, 66},
    {11, 13, 63},
    {18, 6, 64},
    {13, 1, 66},
    {13, -4, 71},
    {8, -5, 70},
    {7, -12, 72},
    {2, -9, 63},
    {13, -10, 62},
    {15, -18, 72},
    {6, -19, 67},
    {2, -23, 71},
    {-4, -15, 75},
    {-2, -9, 74},
    {-7, -2, 68},
    {-16, -4, 72},
    {-12, -8, 79},
    {-15, 0, 77},
    {-12, 5, 75},
    {-9, 9, 77},
    {-8, 12, 79},
    {-1, 17, 80},
    {-3, 11, 93},
    {3, 11, 84},
    {12, 12, 81},
    {10, 6, 91},
    {11, 13, 87},
    {15, 11, 81},
    {11, 5, 77},
    {10, 2, 76},
    {7, -5, 78},
    {6, -9, 83},
    {0, -10, 84},
    {-4, 4, 79},
    {-5, 3, 87},
    {-4, 7, 92},
    {-1, 11, 105}
};


class AngularPattern
{
  const int16_t PERIOD = 10000;  // ms per rev
  const int16_t NUM_STRIPES = 2;
  const int16_t ANGLE_STRIPES = 120;  // deg

  int16_t angles[NUM_LEDS];  // deg
  int16_t hue, sat, val;
  int16_t phase;  // deg

  public:
  AngularPattern(uint32_t t, uint8_t val_ = 255)
  {
    reset(t, val_);
  }
  
  void reset(uint32_t t, uint8_t val_ = 255)
  {
    float cx = 0, cy = 0, cz = 64;
    float nx = 1, ny = 0, nz = 0;
    
   for (int i = 0; i < NUM_LEDS; i++)
   {
//     angles[i] = round(atan2(cartesian[i][1], cartesian[i][0]) * 180 / 3.14);
     float x = nx * (cartesian[i][0] - cx) + ny * (cartesian[i][1] - cy) + nz * (cartesian[i][2] - cz);
     float y = cartesian[i][2] - cz;
     angles[i] = round(atan2(x, y) * 180 / 3.14);
   }
     
      
    hue = random(255);
    sat = 0;//random(255);
    val = val_;
    update(t);
  }

  inline bool update(uint32_t t)
  {
    phase = t * 360 / PERIOD;
    return true;
  }

  inline void get_hsv(int i, uint8_t& hue_, uint8_t& sat_, uint8_t& val_)
  {
   int16_t ang = (angles[i] + phase) % (360 / NUM_STRIPES);
   sat_ = sat;
   hue_ = hue;
   val_ = constrain(map(abs(ang - ANGLE_STRIPES/2), 0, ANGLE_STRIPES/2, val, 0), 0, val);
  }
};


class LinearPattern
{
  const int16_t LEN = height;  // len unit
  const int32_t TIME_FADE = 500;
  int16_t PERIOD = 2000;  // ms per len
  int16_t NUM_STRIPES = 2;
  int16_t LEN_STRIPES = LEN / 4;

  int16_t dist[NUM_LEDS];  // dist to base plane
  int16_t hue, sat, val;
  int16_t valMax;
  int16_t phase;  // deg
  int32_t timeout = 0;
  uint32_t t0 = 0;

  public:
  char msg[80];

  LinearPattern(uint32_t t, uint8_t val_ = 255)
  {
    reset(t, val_);
  }

  void reset(uint32_t t, uint8_t val_ = 255)
  {
    PERIOD = random(2000, 3000);
    NUM_STRIPES = random(2, 4);
//    PERIOD = 2265;
//    NUM_STRIPES = 2;
    
    LEN_STRIPES = LEN / 2;
    t0 = t;
    timeout = t + random(10000, 12000);
    
    hue = random(255);
    sat = random(32);
    valMax = val_;
    
    float nx = random(-100, 100),
          ny = random(-100, 100),
          nz = random(-100, 100);
//    float nx = -87, 
//          ny = -72, 
//          nz = -7;
    sprintf(msg, "N: (%d %d %d), PERIOD: %d, NUM_STRIPES: %d, timeout: %d, sat: %d", 
      int(nx), int(ny), int(nz), PERIOD, NUM_STRIPES, timeout-t, sat);

    if (nx == 0 && ny == 0 && nz == 0) nz = 1.0;
    float n = NUM_STRIPES / sqrt(nx*nx + ny*ny + nz*nz);
    nx *= n;
    ny *= n;
    nz *= n;

    for (int i = 0; i < NUM_LEDS; i++)
      dist[i] = round(nx * cartesian[i][0] + ny * cartesian[i][1] + nz * cartesian[i][2]);
}

  inline bool update(uint32_t t)
  {
    phase = t * LEN / PERIOD;
    if (t - t0 <= TIME_FADE)            val = map(t - t0, 0, TIME_FADE, 0, valMax);
    else if (t > timeout)               val = 0;
    else if (timeout - t <= TIME_FADE)  val = map(timeout - t, 0, TIME_FADE, 0, valMax);
    else                                val = valMax;

    return (t < timeout);
  }

  inline void get_hsv(int i, uint8_t& hue_, uint8_t& sat_, uint8_t& val_)
  {
    int16_t ang = (dist[i] + phase) % LEN;
    sat_ = sat;
    hue_ = hue;
    val_ = constrain(map(abs(ang - LEN_STRIPES/2), 0, LEN_STRIPES/2, val, 0), 0, val);
  }
};


class ExplosionPattern
{
 public:
 const uint32_t DT_MIN = 900,
                DT_MAX = 1200;
 const uint32_t DT_UP = 200;
 const int EXPLOSION_SIZE = height / 2;
 
 int16_t distanceInv[NUM_LEDS];  // 0-255 (255 at center)
 uint32_t t0;
 uint32_t ti;
 uint32_t timeout;
 uint8_t hue;
 uint16_t brightness;
 bool overlay;

 ExplosionPattern(uint32_t t, bool overlay_ = false)
 {
   overlay = overlay_;
   reset(t);
 }
 
 void reset(uint32_t t)
 {
   t0 = t;
   timeout = t + random(DT_MIN, DT_MAX);
   hue = 150;//random(0, 255);

   // pick one bulb as explosion center
   int iCenter = random(NUM_LEDS);
   int16_t x = cartesian[iCenter][0], 
           y = cartesian[iCenter][1], 
           z = cartesian[iCenter][2];

   for (int i = 0; i < NUM_LEDS; i++)
   {
     int16_t dist = sqrt(\
       (cartesian[i][0] - x) * (cartesian[i][0] - x) + \
       (cartesian[i][1] - y) * (cartesian[i][1] - y) + \
       (cartesian[i][2] - z) * (cartesian[i][2] - z));
     
     distanceInv[i] = dist > EXPLOSION_SIZE ? 0 : \
       map(dist, 0, EXPLOSION_SIZE, 1000, 0);
   }
   update(t);
 }

 inline bool update(uint32_t t)
 {
  ti = constrain(map(t, t0, timeout, 0, 250), 0, 1000);
   if (t > timeout)
   {
     brightness = 0;
     return false;
   }
   if (t - t0 < DT_UP)
     brightness = map(t, t0, t0 + DT_UP, 0, 1000);
   else
     brightness = map(t, t0 + DT_UP, timeout, 1000, 0); 
   return true;
 }

 inline void get_hsv(int i, uint8_t& hue_, uint8_t& sat_, uint8_t& val_)
 {
  int phase = (distanceInv[i] * 1000 / EXPLOSION_SIZE + ti) % 1000;


     
   const int16_t sat = 255;
   if (overlay)  // mix colors
   {
     int16_t valNew = distanceInv[i] * brightness * 255 / 1000 / 1000;  // height
//    int16_t valNew = phase * 255 / 1000;
     int16_t satNew = (val_ * sat_ + valNew * sat) / (val_ + valNew + 1);  // radius
     int16_t hueNew = (sat_ * hue_ + satNew * hue) / (sat_ + satNew + 1);  // angle
    //  if (abs(hue_ - hue) > 127)
    //    hueNew += 127;
       
     hue_ = hueNew % 255;
     val_ = constrain(valNew + val_, 0, 255);
     sat_ = constrain(satNew, 0, 255);
   }
   else
   {
     val_ = distanceInv[i] * brightness * 255 / 1000 / 1000;
     hue_ = hue;
     sat_ = sat;
   }
 }
};
