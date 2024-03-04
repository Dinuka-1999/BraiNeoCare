#include <SPI.h> //external Lib <>
#include "config.h"   // Settings
#include "ADS1299.hh" // own Lib

#include <WiFi.h>

#define led_red 36
#define led_green 35

const char* ssid     = "BraiNeoCare";
const char* password     = "braingroup13";
const uint16_t portNumber = 50000; 
IPAddress local_ip(192,168,4,21);
IPAddress gateway(192,168,4,21);
IPAddress subnet(255,255,255,0);
WiFiServer server(portNumber);
WiFiClient client;
bool connected = false;

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


bool ADS_connected = false;
bool data_ready = false;
struct lead_struct leads;
int led_stat = 1;
ADS1299 ADS1;
void IRAM_ATTR DRDY_ISR(void);
void ESPmemcheck();
void ADSerrorcheck();
void led_status();

void setup()
{
  pinMode(led_green, OUTPUT);
  pinMode(led_red, OUTPUT);
  xTaskCreate(
    led_status,
    "led status",
    1000,
    NULL,
    2,
    NULL
  );
  led_stat = 1;
  Serial.begin(115200);
  ESPmemcheck();
  ADS1.setup_master(PIN_NUM_DRDY_1, PIN_CS_1);
  ADSerrorcheck();

  // attachInterrupt(PIN_NUM_DRDY_1, DRDY_ISR, FALLING);

  WiFi.mode(WIFI_AP);
  WiFi.softAPConfig(local_ip, gateway, subnet);
  WiFi.softAP(ssid, password, 10, false, 1);
  IPAddress IP = WiFi.softAPIP();
  Serial.print(" -> IP address: "); Serial.println(IP);
  server.begin();

  led_stat = 2;
}

void loop()
{
  
  if (!connected) {
    // listen for incoming clients
    client = server.available();
    if (client) {
      Serial.println("Got a client !");
      if (client.connected()) {
        Serial.println("and it's connected!");
        connected = true;

        ADS1.WREG(CONFIG3, client.readStringUntil('\n').toInt());    // b’x1xx 1100  Turn on BIAS amplifier, set internal BIASREF voltage
        delay(100); // wait for oscillator startup 20us
        ADS1.WREG(CONFIG1, client.readStringUntil('\n').toInt()); //F5 Output CLK signal for second ADS, 500 SPS
        ADS1.WREG(CONFIG2, client.readStringUntil('\n').toInt()); //F5 Output CLK signal for second ADS, 500 SPS
        ADS1.WREG(CH1SET, client.readStringUntil('\n').toInt());
        ADS1.WREG(CH2SET, client.readStringUntil('\n').toInt());
        ADS1.WREG(CH3SET, client.readStringUntil('\n').toInt());
        ADS1.WREG(CH4SET, client.readStringUntil('\n').toInt()); //0b00000101
        ADS1.WREG(CH5SET, client.readStringUntil('\n').toInt());
        ADS1.WREG(CH6SET, client.readStringUntil('\n').toInt());
        ADS1.WREG(CH7SET, client.readStringUntil('\n').toInt());
        ADS1.WREG(CH8SET, client.readStringUntil('\n').toInt()); //0b10000001
        ADS1.WREG(BIAS_SENSP, client.readStringUntil('\n').toInt());
        ADS1.WREG(BIAS_SENSN, client.readStringUntil('\n').toInt());
        ADS1.WREG(MISC1, client.readStringUntil('\n').toInt()); 
        delay(100);

        led_stat = 4;
        ADS1.START();
        ADS1.RDATAC();
      } else {
        Serial.println("but it's not connected!");
        client.stop();  // close the connection:
      }
    }
    else{
      if (WiFi.softAPgetStationNum()){
        led_stat = 3;
      }
      else{
        led_stat = 2;
      }
    }
  } 
  else {
    if (client.connected()) {
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
        client.write(data, sizeof(float)*9);

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
        Serial.println("ADS not connected");
      }
    } else {
      ADS1.SDATAC();
      ADS1.STOP();
      Serial.println("Client is gone");
      client.stop();  // close the connection:
      connected = false;
    }
  }
  
}

void IRAM_ATTR DRDY_ISR(void)
{
  data_ready = true;
}

void ESPmemcheck()
{
  Serial.print("Total heap: ");
  Serial.println(ESP.getHeapSize());
  Serial.print("Free heap: ");
  Serial.println(ESP.getFreeHeap());
  Serial.print("Minimum free heap: ");
  Serial.println(ESP.getMinFreeHeap());
  Serial.print("Total PSRAM: ");
  Serial.println(ESP.getPsramSize());
  Serial.print("Free PSRAM: ");
  Serial.println(ESP.getFreePsram());
}

void ADSerrorcheck()
{
  if (ADS1.getDeviceID() != 0b00111110)
  {
    Serial.println("ADS1299 Nr.1 not found... restart \n");
    Serial.print("Device id: ");
    Serial.println(ADS1.getDeviceID());
    delay(2000);
    ESP.restart();
  }
  else
  {
    Serial.print("ID: ");
    Serial.println(ADS1.RREG(ID), BIN);
    Serial.print("CONFIG1: ");
    Serial.println(ADS1.RREG(CONFIG1), BIN);
    Serial.print("CONFIG2: ");
    Serial.println(ADS1.RREG(CONFIG2), BIN);
    Serial.print("CONFIG3: ");
    Serial.println(ADS1.RREG(CONFIG3), BIN);
    Serial.print("LOFF: ");
    Serial.println(ADS1.RREG(LOFF), BIN);
    Serial.print("CH1SET: ");
    Serial.println(ADS1.RREG(CH1SET), BIN);
    Serial.print("CH2SET: ");
    Serial.println(ADS1.RREG(CH2SET), BIN);
    Serial.print("CH3SET: ");
    Serial.println(ADS1.RREG(CH3SET), BIN);
    Serial.print("CH4SET: ");
    Serial.println(ADS1.RREG(CH4SET), BIN);
    Serial.print("CH5SET: ");
    Serial.println(ADS1.RREG(CH5SET), BIN);
    Serial.print("CH6SET: ");
    Serial.println(ADS1.RREG(CH6SET), BIN);
    Serial.print("CH7SET: ");
    Serial.println(ADS1.RREG(CH7SET), BIN);
    Serial.print("CH8SET: ");
    Serial.println(ADS1.RREG(CH8SET), BIN);
    Serial.print("BIAS_SENSP: ");
    Serial.println(ADS1.RREG(BIAS_SENSP), BIN);
    Serial.print("BIAS_SENSN: ");
    Serial.println(ADS1.RREG(BIAS_SENSN), BIN);
    Serial.print("LOFF_FLIP: ");
    Serial.println(ADS1.RREG(LOFF_FLIP), BIN);
    Serial.print("LOFF_STATP: ");
    Serial.println(ADS1.RREG(LOFF_STATP), BIN);
    Serial.print("LOFF_STATN: ");
    Serial.println(ADS1.RREG(LOFF_STATN), BIN);
    Serial.print("LOFF_GPIO: ");
    Serial.println(ADS1.RREG(GPIO), BIN);
    Serial.print("LOFF_MISC1: ");
    Serial.println(ADS1.RREG(MISC1), BIN);
    Serial.print("LOFF_CONFIG4: ");
    Serial.println(ADS1.RREG(CONFIG4), BIN);
    ADS_connected = true;
  }
}

void led_status(void * parameters){
  while(1){
    if (led_stat==1){
      digitalWrite(led_red, HIGH);
      vTaskDelay(1000 / portTICK_PERIOD_MS);
      digitalWrite(led_red, LOW);
    }
    else if (led_stat==2)
    {
      digitalWrite(led_green, HIGH);
      vTaskDelay(1000 / portTICK_PERIOD_MS);
      digitalWrite(led_green, LOW);
    }
    else if (led_stat==3)
    {
      digitalWrite(led_red, HIGH);
      vTaskDelay(100 / portTICK_PERIOD_MS);
      digitalWrite(led_red, LOW);
      vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    else if (led_stat==4)
    {
      digitalWrite(led_green, HIGH);
      vTaskDelay(100 / portTICK_PERIOD_MS);
      digitalWrite(led_green, LOW);
      vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
  }
}