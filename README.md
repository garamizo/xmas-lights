# xmas-lights

Embelish a christmas tree with colorfull and geometric patterns as a function of the position of each light bulb.

![](img/final.gif) 

## Map Bulb Coordinates

Calculate the spatial position of each bulb from the interpolation along the LED strip length. However, the position of the LED strip connectors must be measured manually. See [map_bulbs.m](map_bulbs.m) for more details.

To execute all light patterns, the bulb positions must be known in cartesian, cylindrical, and Y-axis cylindrical coordinates.

## Color Pattern Design

The tree color hue changes over time and along the angle parameter of the cylindrical coordinate. In addition, this color hue is "masked" by a geometrical pattern that periodically alternate in shape. See [animate_pattern.m](animate_pattern.m) for more details.

![](img/pattern.gif)

**Background Colors:**

The color of each bulb is defined by its Hue, Saturation, and Value (HSV). The Saturation and Value are kept at maximum while the Hue is a function of

`hue := 127 + 127 * sin(Kt*time + Kq*angle)`

where `Kt`, `Kq`, and `angle` are the time constant, angle constant, and angle parameter of the bulb in cylindrical coordinates. 

**Mask Pattern:**

For each bulb, the brightness state is defined as

`bright := sin(Kt*time + Kc*coord)`

Where `Kt`, `Kc`, and `coord` are the time constant, parameter constant, and coordinate of bulb. The bulb is off if `bright` is less than 0, otherwise it has the `hue` defined previously.

Different parameter constant and coordinates are used depending on the pattern type:

| Pattern | Function of parameter |
| ------- | ----------- |
| Radial  | Angle, cylindrical |
| Waterfall | Z |
| Circles | Radius, Y-axis cylindrical |
| Ray | Angle, Y-axis cylindrical | 

## Physical Implementation

**Materials:**

- 9 LED strips (~~WS2811 controller, 12V~~)
- Microcontroller (Teensy 3.2)
- Power supply, 12V/6A and 5V (Computer supply, ATX)
- Christmas tree (1.9 m tall, 0.84 m diameter of base)

**Important Remarks:**

- Add a power connection (directly from the power supply) every 2 LED strips to avoid voltage drop along daisy-chain. Otherwise color hue is distorted (colors that needs more voltage are more affected).
- Mind the data wire end when connecting to the microcontroller. Use the ~~female~~ end.
- The microcontroller, the LED strips, and the power supply must all share the same ground voltage.
- Jump start the ATX power supply connecting green pin to ground (see [Instructables](https://www.instructables.com/id/How-to-power-up-an-ATX-Power-Supply-without-a-PC/))

**Software Challenges:**
- Arduino has sub-par math processing capabilities (probably due to trigonometric functions). Use Teensy instead.
- Avoid having to calculate bulb coordinates on Arduino. Use MATLAB to calculate it and export to C++ code (see [map_bulbs.m](map_bulbs.m) and [print_coordinates.m](print_coordinates.m)).

