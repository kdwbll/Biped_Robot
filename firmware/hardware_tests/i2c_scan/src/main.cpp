#include <Arduino.h>
#include <Wire.h>
void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
}
void loop() {
  Serial.println("Scanning I2C on SDA 21, SCL 22...");
  int found = 0;
  for (int address = 1; address < 127; ++address) {
    Wire.beginTransmission(address);
    if (Wire.endTransmission() == 0) {
      Serial.printf("Found: 0x%02X\n", address);
      ++found;
    }
  }
  if (!found) Serial.println("No devices found.");
  delay(5000);
}
