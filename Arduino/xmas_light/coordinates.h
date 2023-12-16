const float colormap_hot[][30] = {
  {11.00000, 32.00000, 55.00000, 79.00000, 102.00000, 126.00000, 147.00000, 171.00000, 194.00000, 218.00000, 242.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000},
  {0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 10.00000, 31.00000, 55.00000, 79.00000, 102.00000, 126.00000, 149.00000, 170.00000, 194.00000, 218.00000, 241.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000},
  {0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 15.00000, 50.00000, 82.00000, 117.00000, 153.00000, 188.00000, 223.00000, 255.00000}
};

const float colormap_x[50] = {0.00000, 5.20408, 10.40816, 15.61224, 20.81633, 26.02041, 31.22449, 36.42857, 41.63265, 46.83673, 52.04082, 57.24490, 62.44898, 67.65306, 72.85714, 78.06122, 83.26531, 88.46939, 93.67347, 98.87755, 104.08163, 109.28571, 114.48980, 119.69388, 124.89796, 130.10204, 135.30612, 140.51020, 145.71429, 150.91837, 156.12245, 161.32653, 166.53061, 171.73469, 176.93878, 182.14286, 187.34694, 192.55102, 197.75510, 202.95918, 208.16327, 213.36735, 218.57143, 223.77551, 228.97959, 234.18367, 239.38776, 244.59184, 249.79592, 255.00000};

const float colormap_hsv[][50] = {{255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 254.00000, 232.00000, 203.00000, 173.00000, 144.00000, 114.00000, 79.00000, 49.00000, 20.00000, 2.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 4.00000, 25.00000, 55.00000, 90.00000, 120.00000, 149.00000, 179.00000, 208.00000, 244.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000},
  {0.00000, 30.00000, 59.00000, 89.00000, 118.00000, 154.00000, 183.00000, 213.00000, 241.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 252.00000, 223.00000, 187.00000, 158.00000, 128.00000, 99.00000, 69.00000, 34.00000, 8.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000},
  {0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 12.00000, 45.00000, 75.00000, 104.00000, 134.00000, 163.00000, 199.00000, 228.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 255.00000, 237.00000, 207.00000, 177.00000, 142.00000, 113.00000, 83.00000, 53.00000, 24.00000}
};

const float cartesian[][3] = {
  {0.36729, -0.43757, -0.20551},
{0.36729, -0.43757, -0.20551},
{0.21109, -0.07843, -0.84433},
{0.15871, -0.43964, -0.68636},
{0.32886, -0.50909, -0.73020},
{0.27061, -0.67456, -0.82900},
{0.45813, -0.49884, -0.39250},
{-0.18863, 0.33739, -0.10184},
{-0.39491, -0.31371, -0.58399},
{-0.10293, 0.19271, -0.43996},
{-0.23450, -0.39141, -0.79140},
{-0.35543, -0.55359, -0.84496},
{-0.40027, -0.58121, -0.82464},
{-0.48781, -0.41406, -0.83621},
{-0.64363, -0.26646, -0.77219},
{-0.74851, 0.14153, -0.38534},
{-0.56820, -0.09955, -0.78572},
{-0.68199, -0.11050, -0.67488},
{-0.47219, -0.12780, -1.00503},
{-0.70083, 0.02719, -0.65868},
{-0.30174, -0.13336, -0.50329},
{-0.62552, 0.19538, -0.62209},
{-0.46197, 0.27352, -0.78493},
{-0.41247, 0.10447, -0.80973},
{0.17170, -0.02686, -0.67594},
{-0.05941, 0.33792, -0.85770},
{-0.12092, 0.26972, -0.54322},
{-0.03912, 0.23006, -0.70380},
{0.01532, 0.82723, -0.47687},
{0.04692, 0.66436, -0.59981},
{0.18989, 0.57104, -0.81923},
{0.21360, 0.70254, -0.56147},
{0.12708, 0.62066, -0.52102},
{0.19427, 0.49124, -0.47370},
{0.20529, 0.37993, -0.60407},
{-0.08450, -0.10722, -0.24612},
{0.45252, 0.33062, -0.53772},
{0.53788, 0.24047, -0.41556},
{0.47714, 0.18346, -0.46311},
{0.47355, -0.02281, -0.50193},
{0.52279, 0.07184, -0.42537},
{0.44910, -0.13869, -0.53229},
{0.56233, -0.00408, -0.34135},
{0.48811, -0.22209, -0.47006},
{0.48035, -0.14591, -0.58357},
{0.51692, -0.07649, -0.50992},
{0.66343, -0.25459, -0.54878},
{0.45026, -0.25078, -0.48956},
{0.52939, -0.41650, -0.41774},
{0.36263, -0.32677, -0.52871},
{0.23317, -0.25779, -0.67505},
{0.26435, -0.22868, -0.57859},
{0.11899, -0.60804, -0.48899},
{-0.01824, -0.49949, -0.45982},
{-0.05688, -0.27598, -0.50951},
{-0.11208, -0.60987, -0.47155},
{-0.04358, -0.57695, -0.49528},
{-0.35455, -0.50629, -0.32742},
{-0.38881, -0.37877, -0.29625},
{-0.48791, -0.48907, -0.45385},
{-0.26570, 0.29985, -0.36973},
{-0.61621, -0.12981, -0.41144},
{-0.51218, -0.33141, -0.42675},
{-0.60106, -0.13583, -0.35765},
{-0.30980, -0.19459, -0.51673},
{-0.04148, -0.26270, -0.76065},
{-0.64869, 0.13194, -0.49754},
{-0.65046, 0.12672, -0.47157},
{-0.67048, 0.27938, -0.45103},
{-0.44556, 0.32605, -0.51911},
{-0.43520, 0.43699, -0.59846},
{-0.35461, 0.12850, -0.87914},
{-0.08256, 0.28110, -0.26420},
{-0.18442, 0.46906, -0.54957},
{-0.03991, 0.57962, -0.47041},
{0.15567, 0.61366, -0.32313},
{0.03802, 0.48340, -0.37447},
{-0.06605, 0.46986, -0.21408},
{0.15659, 0.50218, -0.43983},
{0.34041, 0.41949, -0.40676},
{0.49901, 0.28969, -0.34198},
{0.32873, 0.24035, -0.32766},
{0.47134, 0.07304, -0.29458},
{0.37997, -0.05149, -0.18584},
{0.45341, -0.18342, -0.26953},
{0.45056, -0.22166, -0.31529},
{0.34382, -0.07543, -0.29527},
{0.21978, -0.11899, -0.30351},
{0.31794, -0.31947, -0.29697},
{0.23949, -0.37474, -0.40807},
{0.18366, -0.58604, -0.29194},
{0.15594, -0.60910, -0.34006},
{0.18206, -0.24733, 0.22266},
{0.12391, -0.40627, -0.28824},
{-0.04627, -0.39554, -0.33866},
{-0.08461, -0.34336, -0.33709},
{-0.19584, -0.29235, -0.18904},
{-0.42174, -0.19348, -0.24604},
{-0.45333, -0.04590, -0.22612},
{-0.29066, 0.03309, -0.33931},
{-0.39083, 0.06994, -0.38941},
{-0.50060, 0.08984, -0.29458},
{-0.39746, 0.25271, -0.43797},
{-0.21455, 0.14998, -0.39470},
{-0.27189, 0.25434, -0.50003},
{-0.15510, 0.45752, -0.32616},
{-0.24017, 0.57681, -0.19958},
{-0.11019, 0.48911, -0.18976},
{-0.03192, 0.54683, -0.19427},
{0.15805, 0.40717, -0.23978},
{0.14470, 0.31779, -0.29189},
{0.18206, -0.24733, 0.22266},
{0.36956, 0.16151, -0.16385},
{0.58308, 0.21184, -0.27136},
{0.55211, 0.12216, -0.23936},
{0.49781, 0.13639, -0.11060},
{0.48509, -0.01986, -0.03778},
{0.34707, -0.09879, -0.00828},
{0.38078, -0.27112, -0.10981},
{0.36167, -0.19640, -0.02837},
{0.20246, -0.39831, -0.02601},
{0.06616, -0.27925, -0.04804},
{0.10137, -0.36115, -0.16472},
{-0.07504, -0.48610, -0.07765},
{-0.09078, -0.45679, -0.06963},
{-0.19861, -0.27281, -0.07755},
{-0.24789, -0.26010, -0.18040},
{-0.31679, -0.27749, -0.09431},
{-0.44399, -0.04786, -0.06945},
{0.12535, -0.16862, -0.29077},
{-0.35229, 0.06737, -0.10338},
{-0.28372, 0.13150, -0.08305},
{-0.34097, 0.40505, -0.05165},
{-0.15768, 0.36590, -0.17228},
{-0.14317, 0.30614, -0.26298},
{-0.02125, 0.45937, -0.17901},
{0.06108, 0.51121, -0.04754},
{0.11875, 0.19750, -0.08762},
{0.09484, 0.34702, -0.07170},
{0.23300, 0.14437, -0.09248},
{0.20462, 0.31301, -0.07255},
{0.36064, 0.21083, -0.10533},
{0.40123, 0.13709, -0.08255},
{0.35699, -0.02198, 0.12287},
{0.42863, 0.01216, 0.00275},
{0.44269, -0.07721, -0.01823},
{0.42951, -0.09082, 0.11562},
{0.22856, -0.16027, 0.02153},
{0.17827, -0.15395, 0.20521},
{0.06107, -0.30760, 0.17998},
{-0.09000, -0.37143, 0.16602},
{-0.34671, -0.41582, 0.16147},
{-0.32823, -0.07563, 0.08626},
{-0.30163, -0.10144, 0.20067},
{-0.50158, 0.04102, 0.14422},
{-0.35864, 0.08287, 0.04146},
{-0.27027, 0.15098, -0.00877},
{-0.25926, 0.32285, 0.09317},
{-0.18878, 0.25554, 0.10685},
{0.04000, 0.30727, 0.06488},
{0.12146, 0.41873, 0.05186},
{0.21550, 0.37554, 0.11866},
{0.05900, 0.22687, 0.11792},
{0.28965, 0.21154, 0.26373},
{0.26249, 0.04176, 0.14815},
{0.25704, 0.06789, 0.30501},
{0.40623, 0.02553, 0.25794},
{0.34900, -0.19161, 0.20933},
{0.22425, -0.24492, 0.26540},
{0.34345, -0.30139, 0.21519},
{0.16512, -0.18953, 0.24490},
{0.09901, -0.39847, 0.20033},
{0.06065, -0.40344, 0.23349},
{-0.12240, -0.16807, 0.24656},
{-0.26253, -0.35850, 0.30300},
{-0.19151, -0.25434, 0.38629},
{-0.20396, -0.23964, 0.38819},
{-0.31499, 0.00626, 0.25700},
{-0.26320, -0.08694, 0.33426},
{-0.24038, 0.07874, 0.16595},
{-0.04780, 0.09808, 0.36408},
{-0.28329, 0.30408, 0.22413},
{-0.21893, 0.45268, 0.37087},
{-0.21628, 0.33004, 0.36840},
{-0.06046, 0.37455, 0.26077},
{0.06502, 0.38923, 0.23656},
{-0.12554, 0.13990, 0.30273},
{0.15143, 0.05107, 0.17457},
{0.30388, 0.06946, 0.37033},
{0.26184, -0.05058, 0.47758},
{0.35239, -0.07036, 0.36928},
{0.20614, -0.22054, 0.39256},
{0.19220, -0.22237, 0.42391},
{0.13692, -0.25558, 0.63529},
{0.14474, -0.26857, 0.50144},
{0.01062, -0.31869, 0.54011},
{-0.10283, -0.36216, 0.65570},
{-0.33310, -0.22018, 0.60995},
{-0.29828, -0.18201, 0.57558},
{-0.25662, -0.03373, 0.46909},
{-0.33613, 0.23387, 0.37003},
{-0.19584, 0.12570, 0.50258},
{-0.19050, 0.34107, 0.47131},
{-0.12791, 0.25049, 0.47343},
{-0.03318, 0.24278, 0.46532},
{0.11694, 0.25523, 0.63012},
{0.07394, 0.22946, 0.36691},
{0.23389, 0.22960, 0.46350},
{0.31553, 0.23115, 0.63544},
{0.15785, 0.15910, 0.54677},
{0.23157, -0.00545, 0.57583},
{0.27130, -0.11635, 0.56381},
{0.07842, -0.10094, 0.65535},
{0.25552, -0.19474, 0.65532},
{0.23736, -0.31427, 0.66610},
{0.14232, -0.02358, 0.54932},
{-0.10684, -0.20310, 0.77338},
{-0.17797, -0.16020, 0.72007},
{-0.21823, -0.17076, 0.75985},
{-0.39344, 0.00850, 0.92529},
{-0.28222, 0.04937, 0.79298},
{-0.32958, 0.13506, 0.75166},
{-0.21965, 0.19314, 0.76427},
{-0.13360, 0.33053, 0.63253},
{-0.10824, 0.35984, 0.81910},
{0.04828, 0.19726, 0.80994},
{0.08146, 0.16579, 0.69197},
{0.22925, 0.18553, 0.80575},
{0.29650, 0.05495, 0.90607},
{0.16395, -0.03411, 0.93237},
{0.09979, -0.08200, 1.00202},
{0.24579, -0.12064, 0.87781},
{0.16488, -0.20880, 0.91172},
{0.04443, -0.24743, 0.79459},
{-0.22362, -0.25693, 0.93789},
{-0.32614, -0.14730, 0.93647},
{-0.08297, -0.08690, 0.98147},
{-0.05934, -0.00637, 1.00529},
{-0.05398, 0.15144, 0.94447},
{0.04007, 0.28550, 1.05801},
{0.21862, 0.20170, 1.10352},
{0.16358, 0.04851, 1.07355},
{0.19488, -0.10412, 1.20927},
{0.03505, -0.23492, 1.05151},
{-0.01245, -0.01753, 1.17591},
{-0.13103, 0.16459, 1.12432},
{-0.06553, 0.24427, 0.95952},
{-0.00853, 0.17106, 1.20360},
{-0.03912, 0.11160, 1.35471},
{0.05528, -0.11422, 1.38967},
{-0.09840, 0.26187, 1.14666},
{0.05504, 0.17438, 1.18859},
{0.18933, 0.25846, 0.97887},
{0.20596, 0.08572, 0.84785},
{0.37466, 0.07088, 0.96770},
{0.16495, -0.19601, 0.90361},
{0.28225, -0.20000, 0.98563},
{0.18206, -0.24733, 0.22266},
{-0.03065, -0.36366, 1.04068},
{-0.11139, -0.37253, 1.03925},
{-0.19259, -0.26885, 0.88897},
{-0.19534, -0.05062, 0.81576},
{-0.27321, -0.03195, 0.82589},
{-0.29938, 0.15721, 0.91757},
{-0.25447, 0.23063, 0.74329},
{-0.11519, 0.38783, 0.71911},
{-0.05640, 0.27784, 0.50864},
{0.04782, 0.34108, 0.45415},
{0.09276, 0.24080, 0.39857},
{0.27389, 0.23903, 0.48090},
{0.28460, 0.17271, 0.21947},
{0.31923, -0.11013, 0.38415},
{0.37511, -0.12733, 0.41628},
{0.22559, -0.27016, 0.52723},
{0.25791, -0.30010, 0.28944},
{0.20672, -0.37620, 0.24360},
{0.01195, -0.32621, 0.22296},
{-0.09225, -0.55785, 0.23563},
{-0.10468, -0.56029, 0.14970},
{-0.29983, -0.19590, -0.11701},
{-0.26574, -0.33529, 0.06988},
{-0.48442, -0.25374, 0.16905},
{-0.41688, -0.10408, 0.01312},
{-0.52030, 0.09319, 0.04263},
{-0.30170, 0.13508, -0.00369},
{-0.38820, 0.31433, -0.00955},
{-0.31503, 0.45185, -0.04377},
{-0.12391, 0.42398, -0.05383},
{-0.12454, 0.53124, -0.25241},
{0.07326, 0.58756, -0.25031},
{0.05507, 0.51036, -0.44702},
{0.26852, 0.56400, -0.58983},
{0.35086, 0.55297, -0.64636},
{0.45248, 0.47732, -0.81410},
{0.62465, 0.31643, -0.81898},
{0.29981, 0.03333, -0.63141},
{0.75859, 0.07690, -0.67776},
{0.64896, -0.06656, -0.55864},
{0.73424, -0.12542, -0.66735},
{0.60783, -0.31170, -0.68976}};
