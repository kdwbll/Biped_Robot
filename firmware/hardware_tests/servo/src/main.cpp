#include <Arduino.h>
#include <ESP32Servo.h>
Servo motor;
String input;
bool enabled = false;
void setup() {
  Serial.begin(115200);
  Serial.println("Servo test: type 1500 to enable at center, 1400-1600 to move, or off to stop pulses.");
  Serial.println("Use GPIO13. Power servo separately and connect grounds.");
}
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      input.trim();
      if (input == "off") { motor.detach(); enabled = false; Serial.println("Pulses off"); }
      else if (input == "1500" || input == "1400" || input == "1450" || input == "1550" || input == "1600") {
        if (!enabled && input != "1500") Serial.println("First send 1500.");
        else { if (!enabled) { motor.setPeriodHertz(50); motor.attach(13, 500, 2500); enabled = true; } motor.writeMicroseconds(input.toInt()); Serial.println("Pulse us: " + input); }
      } else Serial.println("Use 1500, 1400, 1450, 1550, 1600, or off.");
      input = "";
    } else if (input.length() < 16) input += c;
  }
}
