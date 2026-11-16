const int BUZZER_PIN = 8;
String command;

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.begin(9600);
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  command = Serial.readStringUntil('\n');
  command.trim();
  if (command == "ALERT") {
    digitalWrite(BUZZER_PIN, HIGH);
  } else if (command == "CLEAR") {
    digitalWrite(BUZZER_PIN, LOW);
  }
}