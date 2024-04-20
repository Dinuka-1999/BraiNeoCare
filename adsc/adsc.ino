#define  DEBUG  0     // Turn on debug statements to the serial output

#if DEBUG
  #define DEBUG_PRINT(...) Serial.print(__VA_ARGS__)
  #define DEBUG_PRINTLN(...) Serial.println(__VA_ARGS__)
#else
  #define DEBUG_PRINT(...)
  #define DEBUG_PRINTLN(...)
#endif


#include <SPI.h> //external Lib <>
#include "config.h"   // Settings
#include "ADS1299.hh" // own Lib
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

#include <WiFi.h>

const char* ssid     = "BraiNeoCare";
const char* password     = "braingroup13";
const uint16_t portNumber = 50000; 
IPAddress local_ip(192,168,4,21);
IPAddress gateway(192,168,4,21);
IPAddress subnet(255,255,255,0);
WiFiServer server(portNumber);
WiFiClient client;
Adafruit_MPU6050 mpu;

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
  float a1;
  float a2;
  float a3;
  float g1;
  float g2;
  float g3;
  float temp;
  int32_t bat;
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
void setup_mpu();

void setup()
{
  pinMode(led_green, OUTPUT);
  pinMode(led_red, OUTPUT);
  pinMode(PIN_BAT_MON, INPUT);

  xTaskCreate(led_status,"led status",3000,NULL,2,NULL);
  led_stat = 1;
  if (DEBUG){
    Serial.begin(115200);
  }
  ESPmemcheck();

  ADS1.setup_master(PIN_NUM_DRDY_1, PIN_CS_1);
  ADSerrorcheck();
  attachInterrupt(PIN_NUM_DRDY_1, DRDY_ISR, FALLING);

  setup_mpu();
  
  WiFi.mode(WIFI_AP);
  WiFi.softAPConfig(local_ip, gateway, subnet);
  WiFi.softAP(ssid, password, 10, false, 1);
  IPAddress IP = WiFi.softAPIP();
  DEBUG_PRINT(" -> IP address: "); DEBUG_PRINTLN(IP);
  server.begin();

  led_stat = 2;
}

void loop()
{
  
  if (!connected) {
    // listen for incoming clients
    client = server.available();
    if (client) {
      DEBUG_PRINTLN("Got a client !");
      if (client.connected()) {
        DEBUG_PRINTLN("and it's connected!");
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
        ADSerrorcheck();
        led_stat = 4;
        ADS1.START();
        ADS1.RDATAC();
      } else {
        DEBUG_PRINTLN("but it's not connected!");
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

        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);

        leads.a1 = a.acceleration.x;
        leads.a2 = a.acceleration.y;
        leads.a3 = a.acceleration.z;
        leads.g1 = g.gyro.x;
        leads.g2 = g.gyro.y;
        leads.g3 = g.gyro.z;
        leads.temp = temp.temperature;
        leads.bat = analogRead(PIN_BAT_MON);

        byte* data = (byte*)&leads;
        client.write(data, sizeof(leads));

        DEBUG_PRINT("ADS_0:");
        DEBUG_PRINT(ADS_1.rawresults[0]);
        DEBUG_PRINT(",ADS_1:");
        DEBUG_PRINT(ADS_1.rawresults[1]);
        DEBUG_PRINT(",ADS_2:");
        DEBUG_PRINT(ADS_1.rawresults[2]);
        DEBUG_PRINT(",ADS_3:");
        DEBUG_PRINT(ADS_1.rawresults[3]);
        DEBUG_PRINT(",ADS_4:");
        DEBUG_PRINT(ADS_1.rawresults[4]);
        DEBUG_PRINT(",ADS_5:");
        DEBUG_PRINT(ADS_1.rawresults[5]);
        DEBUG_PRINT(",ADS_6:");
        DEBUG_PRINT(ADS_1.rawresults[6]);
        DEBUG_PRINT(",ADS_7:");
        DEBUG_PRINT(ADS_1.rawresults[7]);
        DEBUG_PRINT(",ADS_8:");
        DEBUG_PRINTLN(ADS_1.rawresults[8]);

        // DEBUG_PRINT(",AccelerationX:");
        // DEBUG_PRINT(a.acceleration.x);
        // DEBUG_PRINT(",AccelerationY:");
        // DEBUG_PRINT(a.acceleration.y);
        // DEBUG_PRINT(",AccelerationZ:");
        // DEBUG_PRINT(a.acceleration.z);

        // DEBUG_PRINT(",RotationX:");
        // DEBUG_PRINT(g.gyro.x);
        // DEBUG_PRINT(",RotationY:");
        // DEBUG_PRINT(g.gyro.y);
        // DEBUG_PRINT(",RotationZ:");
        // DEBUG_PRINT(g.gyro.z);

        // DEBUG_PRINT(",Temperature:");
        // DEBUG_PRINT(temp.temperature);

        // DEBUG_PRINT(",Battery:");
        // DEBUG_PRINTLN(analogRead(PIN_BAT_MON));
        
        data_ready = false;
      }
      if (ADS_connected == false)
      {
        DEBUG_PRINTLN("ADS not connected");
      }
    } else {
      ADS1.SDATAC();
      ADS1.STOP();
      DEBUG_PRINTLN("Client is gone");
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
  DEBUG_PRINT("Total heap: ");
  DEBUG_PRINTLN(ESP.getHeapSize());
  DEBUG_PRINT("Free heap: ");
  DEBUG_PRINTLN(ESP.getFreeHeap());
  DEBUG_PRINT("Minimum free heap: ");
  DEBUG_PRINTLN(ESP.getMinFreeHeap());
  DEBUG_PRINT("Total PSRAM: ");
  DEBUG_PRINTLN(ESP.getPsramSize());
  DEBUG_PRINT("Free PSRAM: ");
  DEBUG_PRINTLN(ESP.getFreePsram());
}

void ADSerrorcheck()
{
  while (ADS1.getDeviceID() != 0b00111110)
  {
    DEBUG_PRINTLN("ADS1299 Nr.1 not found \n");
    DEBUG_PRINT("Device id: ");
    DEBUG_PRINTLN(ADS1.getDeviceID());
    delay(2000);
  }
  DEBUG_PRINT("ID: ");
  DEBUG_PRINTLN(ADS1.RREG(ID), BIN);
  DEBUG_PRINT("CONFIG1: ");
  DEBUG_PRINTLN(ADS1.RREG(CONFIG1), BIN);
  DEBUG_PRINT("CONFIG2: ");
  DEBUG_PRINTLN(ADS1.RREG(CONFIG2), BIN);
  DEBUG_PRINT("CONFIG3: ");
  DEBUG_PRINTLN(ADS1.RREG(CONFIG3), BIN);
  DEBUG_PRINT("LOFF: ");
  DEBUG_PRINTLN(ADS1.RREG(LOFF), BIN);
  DEBUG_PRINT("CH1SET: ");
  DEBUG_PRINTLN(ADS1.RREG(CH1SET), BIN);
  DEBUG_PRINT("CH2SET: ");
  DEBUG_PRINTLN(ADS1.RREG(CH2SET), BIN);
  DEBUG_PRINT("CH3SET: ");
  DEBUG_PRINTLN(ADS1.RREG(CH3SET), BIN);
  DEBUG_PRINT("CH4SET: ");
  DEBUG_PRINTLN(ADS1.RREG(CH4SET), BIN);
  DEBUG_PRINT("CH5SET: ");
  DEBUG_PRINTLN(ADS1.RREG(CH5SET), BIN);
  DEBUG_PRINT("CH6SET: ");
  DEBUG_PRINTLN(ADS1.RREG(CH6SET), BIN);
  DEBUG_PRINT("CH7SET: ");
  DEBUG_PRINTLN(ADS1.RREG(CH7SET), BIN);
  DEBUG_PRINT("CH8SET: ");
  DEBUG_PRINTLN(ADS1.RREG(CH8SET), BIN);
  DEBUG_PRINT("BIAS_SENSP: ");
  DEBUG_PRINTLN(ADS1.RREG(BIAS_SENSP), BIN);
  DEBUG_PRINT("BIAS_SENSN: ");
  DEBUG_PRINTLN(ADS1.RREG(BIAS_SENSN), BIN);
  DEBUG_PRINT("LOFF_FLIP: ");
  DEBUG_PRINTLN(ADS1.RREG(LOFF_FLIP), BIN);
  DEBUG_PRINT("LOFF_STATP: ");
  DEBUG_PRINTLN(ADS1.RREG(LOFF_STATP), BIN);
  DEBUG_PRINT("LOFF_STATN: ");
  DEBUG_PRINTLN(ADS1.RREG(LOFF_STATN), BIN);
  DEBUG_PRINT("LOFF_GPIO: ");
  DEBUG_PRINTLN(ADS1.RREG(GPIO), BIN);
  DEBUG_PRINT("LOFF_MISC1: ");
  DEBUG_PRINTLN(ADS1.RREG(MISC1), BIN);
  DEBUG_PRINT("LOFF_CONFIG4: ");
  DEBUG_PRINTLN(ADS1.RREG(CONFIG4), BIN);
  ADS_connected = true;
}

void setup_mpu()
{
  Wire.begin(PIN_IMU_SDA, PIN_IMU_SCL);

  while (!mpu.begin()) {
    DEBUG_PRINTLN("IMU init failed");
    delay(1000);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_5_HZ);
}

void led_status(void * parameters){
  while(1){
    int bat_val = analogRead(PIN_BAT_MON);
    if (bat_val<1400){ //4.2V --> 1740 , 3.3V --> 1350, 0.1V ~ 40
      digitalWrite(led_red, HIGH);
      vTaskDelay(50 / portTICK_PERIOD_MS);
      digitalWrite(led_red, LOW);
      vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    else if (led_stat==1){
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
      digitalWrite(led_green, HIGH);
      digitalWrite(led_red, HIGH);
      vTaskDelay(1000 / portTICK_PERIOD_MS);
      digitalWrite(led_green, LOW);
      digitalWrite(led_red, LOW);
    }
    else if (led_stat==4)
    {
      digitalWrite(led_green, HIGH);
      vTaskDelay(50 / portTICK_PERIOD_MS);
      digitalWrite(led_green, LOW);
      vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
  }
}