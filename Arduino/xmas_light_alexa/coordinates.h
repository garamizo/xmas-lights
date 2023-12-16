#define NUM_LEDS    300  // 9 strips, 50 bulbs each

const float height = 128;  // tree height
const float diameter = 76;  // tree base diameter


const int8_t cartesian[NUM_LEDS][3] = {
  {-2, -1, 50},
{18, -23, 31},
{10, -5, 0},
{7, -23, 7},
{15, -27, 5},
{12, -36, 0},
{22, -26, 22},
{-9, 17, 36},
{-21, -16, 12},
{-5, 9, 19},
{-13, -20, 2},
{-19, -29, 0},
{-21, -30, 0},
{-26, -21, 0},
{-33, -13, 3},
{-38, 7, 22},
{-29, -5, 2},
{-35, -5, 8},
{-25, -7, 0},
{-36, 1, 9},
{-15, -7, 16},
{-32, 10, 10},
{-23, 13, 2},
{-21, 4, 0},
{8, -2, 7},
{-3, 16, 0},
{-6, 13, 14},
{-2, 10, 5},
{1, 38, 16},
{3, 33, 10},
{10, 27, 0},
{11, 34, 12},
{7, 30, 14},
{10, 24, 17},
{10, 18, 10},
{-4, -5, 29},
{23, 15, 14},
{27, 10, 20},
{24, 7, 17},
{24, -2, 16},
{26, 2, 20},
{22, -8, 14},
{28, -1, 24},
{24, -12, 18},
{24, -9, 12},
{26, -5, 15},
{33, -14, 13},
{22, -14, 17},
{26, -22, 20},
{17, -18, 15},
{11, -14, 7},
{12, -12, 12},
{4, -32, 17},
{-1, -26, 19},
{-3, -14, 16},
{-6, -31, 18},
{-3, -30, 17},
{-19, -26, 26},
{-20, -19, 27},
{-25, -25, 19},
{-13, 15, 23},
{-31, -6, 21},
{-27, -16, 20},
{-31, -6, 24},
{-16, -10, 16},
{-3, -14, 3},
{-33, 6, 17},
{-33, 6, 18},
{-34, 14, 19},
{-22, 16, 15},
{-21, 22, 11},
{-18, 5, 0},
{-3, 14, 28},
{-9, 23, 13},
{-1, 29, 17},
{8, 30, 24},
{2, 24, 22},
{-2, 23, 30},
{8, 24, 19},
{17, 20, 20},
{25, 13, 24},
{17, 11, 24},
{24, 2, 26},
{19, -3, 32},
{22, -10, 28},
{22, -12, 26},
{17, -4, 26},
{11, -6, 26},
{15, -17, 27},
{11, -20, 21},
{8, -30, 27},
{7, -32, 25},
{8, -13, 53},
{5, -21, 27},
{-3, -20, 25},
{-4, -18, 25},
{-10, -15, 33},
{-21, -9, 30},
{-23, -2, 31},
{-14, 1, 25},
{-20, 3, 22},
{-25, 4, 27},
{-20, 12, 19},
{-10, 7, 22},
{-13, 12, 16},
{-7, 23, 25},
{-11, 29, 31},
{-4, 24, 32},
{0, 27, 31},
{8, 20, 29},
{7, 15, 26},
{8, -13, 53},
{19, 7, 33},
{30, 9, 27},
{28, 4, 29},
{25, 5, 36},
{25, -2, 40},
{17, -5, 41},
{19, -14, 36},
{18, -10, 40},
{9, -21, 41},
{3, -14, 40},
{4, -19, 34},
{-4, -25, 38},
{-5, -23, 39},
{-10, -14, 38},
{-13, -13, 33},
{-16, -14, 38},
{-22, -2, 39},
{5, -9, 27},
{-18, 3, 37},
{-14, 6, 38},
{-16, 21, 39},
{-7, 18, 33},
{-6, 15, 28},
{0, 23, 32},
{4, 25, 39},
{6, 9, 37},
{5, 17, 38},
{12, 6, 37},
{11, 15, 38},
{18, 9, 36},
{20, 6, 37},
{18, -1, 48},
{22, 0, 42},
{22, -4, 41},
{22, -5, 48},
{11, -8, 43},
{9, -8, 52},
{2, -15, 51},
{-4, -18, 51},
{-18, -20, 51},
{-16, -3, 47},
{-15, -4, 52},
{-25, 2, 50},
{-18, 4, 44},
{-13, 7, 41},
{-12, 16, 46},
{-9, 13, 47},
{2, 15, 45},
{7, 21, 44},
{11, 18, 47},
{3, 11, 48},
{15, 10, 55},
{13, 1, 49},
{13, 3, 57},
{21, 0, 55},
{17, -10, 53},
{11, -12, 56},
{17, -15, 53},
{8, -9, 54},
{4, -20, 52},
{2, -20, 54},
{-6, -8, 55},
{-13, -17, 58},
{-9, -12, 62},
{-10, -12, 62},
{-15, 0, 55},
{-13, -3, 59},
{-11, 4, 50},
{-1, 5, 60},
{-13, 16, 53},
{-10, 23, 61},
{-10, 17, 61},
{-2, 19, 55},
{4, 19, 54},
{-5, 7, 57},
{8, 2, 51},
{16, 3, 61},
{13, -2, 66},
{18, -3, 61},
{10, -11, 62},
{10, -11, 64},
{7, -12, 75},
{7, -13, 68},
{0, -16, 70},
{-5, -17, 76},
{-16, -10, 74},
{-14, -8, 72},
{-12, -1, 66},
{-16, 12, 61},
{-9, 7, 68},
{-8, 18, 66},
{-5, 13, 66},
{0, 12, 65},
{7, 13, 74},
{4, 11, 60},
{12, 11, 65},
{17, 11, 74},
{8, 8, 70},
{12, 0, 71},
{14, -6, 71},
{4, -4, 76},
{13, -9, 76},
{12, -16, 76},
{7, -1, 70},
{-4, -9, 82},
{-8, -7, 79},
{-10, -7, 81},
{-19, 1, 90},
{-13, 3, 83},
{-15, 7, 81},
{-10, 10, 81},
{-5, 17, 74},
{-4, 19, 84},
{3, 10, 83},
{5, 8, 77},
{12, 9, 83},
{16, 3, 88},
{9, -1, 90},
{5, -3, 93},
{13, -5, 87},
{9, -10, 89},
{2, -12, 83},
{-10, -12, 90},
{-16, -6, 90},
{-3, -3, 92},
{-2, 0, 94},
{-1, 8, 90},
{3, 15, 96},
{12, 10, 98},
{9, 3, 97},
{10, -4, 104},
{2, -11, 96},
{0, 0, 102},
{-5, 9, 99},
{-2, 13, 91},
{0, 9, 103},
{0, 7, 111},
{3, -4, 113},
{-3, 14, 101},
{4, 9, 103},
{11, 13, 92},
{11, 4, 85},
{20, 3, 91},
{9, -9, 88},
{15, -9, 92},
{8, -13, 53},
{-1, -17, 96},
{-5, -18, 96},
{-9, -12, 88},
{-9, -1, 84},
{-13, 0, 84},
{-14, 9, 89},
{-11, 12, 80},
{-4, 20, 78},
{-1, 14, 68},
{3, 17, 65},
{5, 12, 62},
{14, 12, 66},
{15, 8, 53},
{16, -5, 61},
{19, -6, 63},
{11, -13, 69},
{13, -15, 57},
{10, -19, 55},
{0, -16, 54},
{-5, -28, 55},
{-5, -28, 50},
{-15, -9, 36},
{-13, -16, 46},
{-24, -12, 51},
{-21, -4, 43},
{-26, 5, 44},
{-15, 7, 42},
{-19, 16, 41},
{-15, 23, 39},
{-5, 21, 39},
{-5, 26, 28},
{4, 29, 28},
{3, 25, 18},
{14, 27, 11},
{18, 26, 8},
{23, 22, 0},
{32, 14, 0},
{15, 0, 9},
{38, 1, 6},
{32, -5, 13},
{37, -8, 7},
{30, -17, 6}};


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
