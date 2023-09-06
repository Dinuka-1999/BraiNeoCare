#include <SPI.h> //external Lib <>
#include "config.h"   // Settings
#include "ADS1299.hh" // own Lib

#include <WebServer.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <FastLED.h>

#define rgb_pin 48

//set up to connect to an existing network (e.g. mobile hotspot from laptop that will run the python code)
const char* ssid = "virus";
const char* password = "nima12345";
WiFiUDP Udp;
unsigned int localUdpPort = 4210;  //  port to listen on
char incomingPacket[255];  // buffer for incoming packets

struct lead_struct{
  int32_t l0;
  int32_t l1;
  int32_t l2;
  int32_t l3;
  int32_t l4;
  int32_t l5;
  int32_t l6;
  int32_t l7;
  int32_t l8;
};

struct lead_struct leads;
CRGB led[1];
ADS1299 ADS1;
void IRAM_ATTR DRDY_ISR(void);
void ESPmemcheck();
void ADSerrorcheck();
bool ADS_connected = false;
bool data_ready = false;

void setup()
{
  Serial.begin(115200);
  Serial.println("MOSI: ");
  Serial.println(MOSI);
  Serial.println("MISO: ");
  Serial.println(MISO);
  Serial.println("SCK: ");
  Serial.println(SCK);
  Serial.println("SS: ");
  Serial.println(SS); 
  ESPmemcheck();
  ADS1.setup_master(PIN_NUM_DRDY_1, PIN_CS_1);
  // ADS1.RESET();
  // delay(50); // wait for clock to settle
  ADSerrorcheck();
  // delay(50); // wait for clock to settle
  digitalWrite(PIN_NUM_STRT, HIGH); // Synchronize Start of ADC's
  ADS1.RDATAC();
  attachInterrupt(PIN_NUM_DRDY_1, DRDY_ISR, FALLING);
  

  FastLED.addLeds<WS2812, rgb_pin, GRB>(led,1);
  FastLED.setBrightness(20);
  int status = WL_IDLE_STATUS;
  WiFi.begin(ssid, password);
  log_d("");

  led[0] = CRGB(255, 0, 0);FastLED.show();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    log_d(".");
  }
  log_d("Connected to wifi");
  Udp.begin(localUdpPort);
  log_d("Now listening at IP %s, UDP port %d\n", WiFi.localIP().toString().c_str(), localUdpPort);
  led[0] = CRGB(0, 0, 255);FastLED.show();

  bool readPacket = false;
  while (!readPacket) {
    int packetSize = Udp.parsePacket();
    if (packetSize)
     {
      log_d("Received %d bytes from %s, port %d\n", packetSize, Udp.remoteIP().toString().c_str(), Udp.remotePort());
      int len = Udp.read(incomingPacket, 255);
      if (len > 0)
      {
        incomingPacket[len] = 0;
      }
      log_d("UDP packet contents: %s\n", incomingPacket);
      readPacket = true;
    } 
  }
  led[0] = CRGB(0, 255, 0);FastLED.show();

  // ADS1.activateTestSignals(CH1SET);
  // ADS1.activateTestSignals(CH2SET);
  // ADS1.activateTestSignals(CH3SET);
  // ADS1.activateTestSignals(CH4SET);
  // ADS1.activateTestSignals(CH5SET);
  // ADS1.activateTestSignals(CH6SET);
  // ADS1.activateTestSignals(CH7SET);
  // ADS1.activateTestSignals(CH8SET);
}

void loop()
{
  if (data_ready == true)
  {
    results ADS_1;
    ADS_1 = ADS1.updateResponder();
    
    leads.l0 = ADS_1.rawresults[0];
    leads.l1 = ADS_1.rawresults[1];
    leads.l2 = ADS_1.rawresults[2];
    leads.l3 = ADS_1.rawresults[3];
    leads.l4 = ADS_1.rawresults[4];
    leads.l5 = ADS_1.rawresults[5];
    leads.l6 = ADS_1.rawresults[6];
    leads.l7 = ADS_1.rawresults[7];
    leads.l8 = ADS_1.rawresults[8];
    byte* data = (byte*)&leads;
    Udp.beginPacket(Udp.remoteIP(), Udp.remotePort());
    Udp.write(data, sizeof(float)*9);
    Udp.endPacket();

    // Serial.print(ADS_1.rawresults[0]);
    // Serial.print(",");
    // Serial.print(ADS_1.rawresults[1]);
    // Serial.print(",");
    // Serial.print(ADS_1.rawresults[2]);
    // Serial.print(",");
    // Serial.print(ADS_1.rawresults[3]);
    // Serial.print(",");
    // Serial.print(ADS_1.rawresults[4]);
    // Serial.print(",");
    // Serial.print(ADS_1.rawresults[5]);
    // Serial.print(",");
    // Serial.print(ADS_1.rawresults[6]);
    // Serial.print(",");
    // Serial.print(ADS_1.rawresults[7]);
    // Serial.print(",");
    // Serial.println(ADS_1.rawresults[8]);

    data_ready = false;
  }
  if (ADS_connected == false)
  {
    log_d("ADS not connected");
  }
}

void IRAM_ATTR DRDY_ISR(void)
{
  data_ready = true;
}

void ESPmemcheck()
{
  log_d("Total heap: %d", ESP.getHeapSize());
  log_d("Free heap: %d", ESP.getFreeHeap());
  log_d("Minimum free heap: %d", ESP.getMinFreeHeap());
  log_d("Total PSRAM: %d", ESP.getPsramSize());
  log_d("Free PSRAM: %d", ESP.getFreePsram());
}

void ADSerrorcheck()
{
  if (ADS1.getDeviceID() != 0b00111110)
  {
    log_d("ADS1299 Nr.1 not found... restart \n");
    log_d("Device id: %d", ADS1.getDeviceID());
    delay(500);
    ESP.restart();
  }
  else
  {
    Serial.print("ID: ");
    Serial.println(ADS1.RREG(ID));
    Serial.print("CONFIG1: ");
    Serial.println(ADS1.RREG(CONFIG1));
    Serial.print("CONFIG2: ");
    Serial.println(ADS1.RREG(CONFIG2));
    Serial.print("CONFIG3: ");
    Serial.println(ADS1.RREG(CONFIG3));
    Serial.print("LOFF: ");
    Serial.println(ADS1.RREG(LOFF));
    Serial.print("CH1SET: ");
    Serial.println(ADS1.RREG(CH1SET));
    Serial.print("CH2SET: ");
    Serial.println(ADS1.RREG(CH2SET));
    Serial.print("CH3SET: ");
    Serial.println(ADS1.RREG(CH3SET));
    Serial.print("CH4SET: ");
    Serial.println(ADS1.RREG(CH4SET));
    Serial.print("CH5SET: ");
    Serial.println(ADS1.RREG(CH5SET));
    Serial.print("CH6SET: ");
    Serial.println(ADS1.RREG(CH6SET));
    Serial.print("CH7SET: ");
    Serial.println(ADS1.RREG(CH7SET));
    Serial.print("CH8SET: ");
    Serial.println(ADS1.RREG(CH8SET));
    Serial.print("BIAS_SENSP: ");
    Serial.println(ADS1.RREG(BIAS_SENSP));
    Serial.print("BIAS_SENSN: ");
    Serial.println(ADS1.RREG(BIAS_SENSN));
    Serial.print("LOFF_FLIP: ");
    Serial.println(ADS1.RREG(LOFF_FLIP));
    Serial.print("LOFF_STATP: ");
    Serial.println(ADS1.RREG(LOFF_STATP));
    Serial.print("LOFF_STATN: ");
    Serial.println(ADS1.RREG(LOFF_STATN));
    Serial.print("LOFF_GPIO: ");
    Serial.println(ADS1.RREG(GPIO));
    Serial.print("LOFF_MISC1: ");
    Serial.println(ADS1.RREG(MISC1));
    Serial.print("LOFF_CONFIG4: ");
    Serial.println(ADS1.RREG(CONFIG4));
    ADS_connected = true;
  }
}
