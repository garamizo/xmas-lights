#define NUM_LEDS 450

class xmas_light {
  public:
  float height = 76 / 39.3701;
  float diameter = 33 / 39.3701;

  float X[NUM_LEDS], Y[NUM_LEDS], Z[NUM_LEDS], Q[NUM_LEDS], R[NUM_LEDS];

  // connector position
  void create_bulbs();
  
};

void xmas_light::create_bulbs()
{
  float Qc[] = {
    -3.141592653589793e+00,
     4.537856055185257e+00,
     1.247910415175946e+01,
     2.076941809873252e+01,
     3.063052837250048e+01,
     4.101523742186674e+01,
     5.340707511102648e+01,
     6.736970912698112e+01,
     8.717919613711676e+01,
     1.314232926751730e+02
  };

  float Zc[] = {
                         0,
     1.523999177040444e-01,
     2.285998765560666e-01,
     3.301998216920963e-01,
     4.698997462541370e-01,
     6.603996433841925e-01,
     8.635995336562519e-01,
     1.092199410212318e+00,
     1.295399300484378e+00,
     1.930398957584563e+00
  };
}

void setup() {
  // put your setup code here, to run once:

}

void loop() {
  // put your main code here, to run repeatedly:

}
