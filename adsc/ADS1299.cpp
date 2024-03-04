#include "ADS1299.hh"

ADS1299::ADS1299()
{                                                            // Standard constructor for initial definitions
  for (int i = 0; i < sizeof ChGain / sizeof ChGain[0]; ++i) // get length of array from size
  {
    ChGain[i] = GAIN; // initial Gain for all channels
  }
  LSB = ((2 * 4.5) / GAIN) / (16777216 - 1); // 2*Vref+-/GAIN/(2^24bits -1)
  outputCount = 0;
  // **** ----- SPI Setup ----- **** //
  // initialise instance of the SPIClass attached to VSPI
  pinMode(VSPI_SCLK, OUTPUT);
  pinMode(VSPI_MOSI, OUTPUT);
  pinMode(VSPI_MISO, INPUT);
  // vspi = new SPIClass(VSPI);
}

/**
 * @brief Sets up master ADS1299, configures SPI and ADS1299 for
 * @details supply slave ADS1299 with clock and BIAS
 * @param _DRDY DATA Ready pin goes LOW, when DATA is available
 * @param _CS  CHIP SELECT pin, goes LOW, when MCU communicates with ADS
 */
void ADS1299::setup_master(int _DRDY, int _CS)
{
  CS = _CS;
  DRDY = _DRDY;
  pinMode(CS, OUTPUT);
  pinMode(PIN_NUM_RST, OUTPUT);
  pinMode(PIN_NUM_STRT, OUTPUT);
  pinMode(PIN_NUM_PWD, OUTPUT);
  pinMode(PIN_NUM_CLKSEL, OUTPUT);
  pinMode(DRDY, INPUT);
  // set up slave select pins as outputs as the Arduino API
  // doesn't handle automatically pulling SS low
  digitalWrite(VSPI_SCLK, LOW);
  digitalWrite(VSPI_MOSI, LOW);
  //digitalWrite(SS, HIGH);
  // SPI Setup
  SPI.begin(VSPI_SCLK, VSPI_MISO, VSPI_MOSI); // Initialize SPI library
  SPI.setBitOrder(MSBFIRST);  // Most significant Bit first
  SPI.setFrequency(spiClk);   // Sets SPI clock
  SPI.setDataMode(SPI_MODE1); //// 1...2.4 MHz, clock polarity = 0; clock phase = 1 (pg. 8)

  // at startup all inputs must be low
  digitalWrite(PIN_NUM_RST, LOW);
  digitalWrite(PIN_NUM_STRT, LOW);
  // LDO power down
  digitalWrite(PIN_NUM_PWD, LOW);

  digitalWrite(PIN_NUM_CLKSEL, HIGH);
  delay(1000); // wait for oscillator startup 20us

  digitalWrite(PIN_NUM_PWD, HIGH);
  digitalWrite(PIN_NUM_RST, HIGH);
  delay(1000); // wait for oscillator startup 20us
  delayMicroseconds(20 * TCLK_cycle); // Recommended 18 Tclk before using device
  RESET();
  SDATAC();                           // DEVICE wakes up in RDATAC so no Registers could be written.
  delayMicroseconds(20 * TCLK_cycle); // Recommended 18 Tclk before using device
  
}

/**
 * @brief Sets up slave ADS1299, configures SPI and ADS1299 for
 * @details supply slave ADS1299 with clock and BIAS
 * @param _DRDY DATA Ready pin goes LOW, when DATA is available
 * @param _CS  CHIP SELECT pin, goes LOW, when MCU communicates with ADS
 */
void ADS1299::setup_slave(int _DRDY, int _CS)
{
  CS = _CS;
  DRDY = _DRDY;
  // Set direction register for VSPI_SCLK and VSPI_MOSI pin.
  // VSPI_MISO pin automatically overrides to INPUT.
  // initialise vspi with default pins
  // SCLK = 18, VSPI_MISO = 19, VSPI_MOSI = 23, SS = 5
  SPI.begin(VSPI_SCLK, VSPI_MISO, VSPI_MOSI); // Initialize SPI library
  SPI.setBitOrder(MSBFIRST);  // Most significant Bit first
  SPI.setFrequency(spiClk);   // Sets SPI clock
  SPI.setDataMode(SPI_MODE1); //// 1...2.4 MHz, clock polarity = 0; clock phase = 1 (pg. 8)

  pinMode(DRDY, INPUT);
  pinMode(CS, OUTPUT);
  // set up slave select pins as outputs as the Arduino API
  // doesn't handle automatically pulling SS low
  digitalWrite(VSPI_SCLK, LOW);
  digitalWrite(VSPI_MOSI, LOW);
  //digitalWrite(SS, HIGH);

  delayMicroseconds(20 * TCLK_cycle); // Recommended 18 Tclk before using device
  SDATAC();                           // DEVICE wakes up in RDATAC so no Registers could be written.

  delayMicroseconds(20 * TCLK_cycle); // Recommended 18 Tclk before using device
  // Setting registers for slave
  // CLOCK: CLKSEL pin 0 (through J); COnf1 1101 0xxx
  // WREG(CONFIG1,0xF5); // Output CLK signal on
  // BIAS: power down the bias amp
  WREG(CONFIG1, 0x95); // No CLKOUT, 500 SPS
  WREG(MISC1, 0x10);   // connect SRB1 to neg Electrodes
  WREG(CONFIG3,0xE8 ); //0x68 0 11 0 1 0 0 0  Turn down BIAS amplifier, set internal BIASREF voltage
}

// ADS1299 SPI Command Definitions (Datasheet, Pg. 35)
/*--------------------------------------------------*/
/*---------------------- System Commands ---------------------*/
/*--------------------------------------------------*/

/**
 * @brief Wakes up the ADC
 *
 */
void ADS1299::WAKEUP()
{
  digitalWrite(CS, LOW);
  SPI.transfer(_WAKEUP);
  digitalWrite(CS, HIGH);
  delayMicroseconds(4 * TCLK_cycle); // wait 4 Tclk (Pg. 35)
}
/**
 * @brief Put the ADC in Standby
 *  5.1 mW instead of 22 mW in running mode
 */
void ADS1299::STANDBY()
{
  digitalWrite(CS, LOW);
  SPI.transfer(_STANDBY);
  digitalWrite(CS, HIGH);
}

/**
 * Reset all Registers
 * sends command 0x06
 * @return nothing
 */
void ADS1299::RESET()
{
  digitalWrite(CS, LOW);
  SPI.transfer(_RESET);
  digitalWrite(CS, HIGH);
  delayMicroseconds(18 * TCLK_cycle); // Recommended 18 Tclk before using device
}
/**
 * @brief Send SPI command or pull START pin LOW, sync multiple ADS
 *
 */
void ADS1299::START()
{
  digitalWrite(CS, LOW);
  //SPI.transfer(_START);
  digitalWrite(PIN_NUM_STRT, LOW);
  delayMicroseconds(4 * TCLK_cycle); // wait 4 clk cycles after this command (DS pg.36)
  digitalWrite(PIN_NUM_STRT, HIGH);
  delayMicroseconds(4 * TCLK_cycle); // wait 4 clk cycles after this command (DS pg.36)
  digitalWrite(CS, HIGH);
}
/**
 * @brief STOP data conversion, allows REGISTER reading/writing
 *
 */
void ADS1299::STOP()
{
  digitalWrite(CS, LOW);
  SPI.transfer(_STOP);
  digitalWrite(PIN_NUM_STRT, LOW);
  delayMicroseconds(4 * TCLK_cycle); // wait 4 clk cycles after this command (DS pg.36)
  digitalWrite(CS, HIGH);
}
/**
 * @brief Read one data chunk from ADS. Transmits 0001 0010 (12h)
 */
void ADS1299::RDATA()
{
  digitalWrite(CS, LOW);
  SPI.transfer(_RDATA);
  digitalWrite(CS, HIGH);
}

/**
 * @brief Read Data Continuously mode. Transmits 0001 0000 (10h)
 * Reads data from ADC Channels continious
 * @return nothing
 */
void ADS1299::RDATAC()
{
  digitalWrite(CS, LOW);
  SPI.transfer(_RDATAC);
  digitalWrite(CS, HIGH);
}

/**
 * @brief Stop Read Data Continuously mode. Transmits 0001 0001 (11h)
 * @return nothing
 */
void ADS1299::SDATAC()
{
  digitalWrite(CS, LOW);
  SPI.transfer(_SDATAC);
  digitalWrite(CS, HIGH);
  delayMicroseconds(4 * TCLK_cycle); // wait 4 clk cycles after this command (DS pg.36)
}

/*--------------------------------------------------*/
/*---------------------- Register Commands ---------------------*/
/*--------------------------------------------------*/

/**
 * @brief Read Register from ADS1299.
 * It answers with the contained values, call SDATAC before reading/writing registers
 * @param _address Single Register adress to read
 */
byte ADS1299::RREG(byte _address)
{  SDATAC();                               //  RDATAC must be stopped before reading 
  byte opcode1 = _address + 0x20;         //  RREG expects 001rrrrr where rrrrr = _address
  digitalWrite(CS, LOW);                  //  open SPI
  SPI.transfer(opcode1);                  //  opcode1
  SPI.transfer(0x00);                     //  opcode2
  regData[_address] = SPI.transfer(0x00); //  update mirror location with returned byte
  digitalWrite(CS, HIGH);                 //  close SPI
  RDATAC();                                // Continue reading
  return regData[_address];               // return requested register value
}

/**
 * @brief Write Settings as 1 Byte of Data to Registers
 *
 * @param _address of Regsiter
 * @param _value of Register
 */
void ADS1299::WREG(byte _address, byte _value)
{
  digitalWrite(CS,LOW);
  // write one Register
  uint8_t opcode1 = _address + 0x40;
  // Send WREG command & address
  SPI.transfer(opcode1);
  // Send number of registers to write
  SPI.transfer(0x00);
  // Write the value to the register
  SPI.transfer(_value);
  digitalWrite(CS, HIGH);
  // a 4 t_CLK (~2 us) period must separate the end of one byte (or command) and the next
} //

/*
void ADS1299::WREG(byte _address, byte _value, byte _numRegistersMinusOne) {
  //overloaded version for multiple registers

}  //
*/

/*--------------------------------------------------*/
/*----------------------  Utility Functions ---------------------*/
/*--------------------------------------------------*/
/**
 * @brief Sets the adjecent Voltage value of the smallest bit of a measurement
 *
 * @param gain currently used Gain value for channel
 * @param vref currently used Voltage reference
 */
void ADS1299::calculateLSB(uint8_t gain, float vref)
{ // Gain as int, vref in mV
  // LSB = (vref) / gain / (16777216 - 1);             // 16.777.216 = 2^24 <-Analog | 2^23 = 8.388.608
  LSB = 1000 * (vref) / (8388608);
}



/**
 * @brief Ask ADS1299 for device ID;
 * @details  returns REV_ID[2:0] 1 DEV_ID[1:0] NU_CH[1:0]
 * @result  desired answer is 0b 0011 1110
 * @return byte ID value
 */
byte ADS1299::getDeviceID()
{
  digitalWrite(CS, LOW); // Low to communicated

  SPI.transfer(_SDATAC);          // 0001 0001 SDATAC Stop Data continous
  SPI.transfer(_RREG);            // 0010 0000 RREG Read Register
  SPI.transfer(0x00);             // 0000 0000 Asking for 1 byte ID
  byte data = SPI.transfer(0x00); // byte to read (hopefully 0b???1 1110) should be 3E or 62
  // Device ID you should get:                    REV_ID[2:0] 1 DEV_ID[1:0] NU_CH[1:0]
  digitalWrite(CS, HIGH); // HIGH to stop communication
  return data;
}

/**
 * @brief  activate testsignal on passed channel; changes CONFIG1/2/3 !
 *
 * @param _channeladdress
 */

void ADS1299::activateTestSignals(byte _channeladdress)
{
  SDATAC();            // 0001 0001 Stop Data reading, to write new settings
  // WREG(CONFIG3, 0xEC); // internal Reference Voltage 1110 0000 no bias
  //WREG(CONFIG1, 0xD6); // 0x96= 1001 0110 Daisy En 250 SPS |
  WREG(CONFIG2, 0xD0); // Config2: 0100(x40)0010(x02)->x00-> 11010000(D0) internal test signal
  // 0xD5 for faster higher test signal
  WREG(_channeladdress, 0x05); // CHnSET: 0100 0101 Setting: 00000101 no PGA, just activate a (1 mV x V REF / 2.4) Square-Wave Test Signal
  // RDATAC();             // read data continous
}

/**
 * @brief setup device for singleended measurement against SRB1
 *
 */

void ADS1299::setSingleended(int configvalue)
{ SDATAC();           
// GAIN 1, normal electrode input -> 0x00
  WREG(CH1SET, configvalue); // measures normal on CH1
  WREG(CH2SET, configvalue); // measures normal on CH1
  WREG(CH3SET, configvalue); // measures normal on CH1
  WREG(CH4SET, configvalue); // measures normal on CH1
  WREG(CH5SET, configvalue); // measures normal on CH1
  WREG(CH6SET, configvalue); // measures normal on CH1
  WREG(CH7SET, configvalue); // measures normal on CH1
  WREG(CH8SET, configvalue);
  RDATAC();

}


/**
 * @brief Reads a single 24 bit HEX and converts to signed 4 byte 
 * 
 * @return signed raw values, have to be scaled by LSB
 */
int32_t ADS1299::readData()
{
	uint8_t ADC_data[3]; 
  ADC_data[0] = SPI.transfer(0x00); //ADC_data[0] holds the MSB
  ADC_data[1] = SPI.transfer(0x00);
  ADC_data[2] = SPI.transfer(0x00); //ADC_data[2] holds the LSB
  
	// Return the 32-bit sign-extended conversion result 
	int32_t signByte;
	if (ADC_data[0] & 0x80u)	{ signByte = 0xFF000000; }
	else						{ signByte = 0x00000000; }

	int32_t upperByte	= ((int32_t) ADC_data[0] & 0xFF) << 16;
	int32_t middleByte	= ((int32_t) ADC_data[1] & 0xFF) << 8;
	int32_t lowerByte	= ((int32_t) ADC_data[2] & 0xFF) << 0;
   int32_t allNumber = (signByte | upperByte | middleByte | lowerByte);
//  double voltage = (double) allNumber* 0.000536441802978515625; // in mV value for gain [1]
	return allNumber;
}

struct results ADS1299::updateResponder(){
 const int numpckts = 9;
  static double output[numpckts];

  //calculateLSB(1, 4.5); // Gain,4.5 Vref set
  if (digitalRead(PIN_NUM_STRT == HIGH))
  { // read only if data can be available 
      digitalWrite(CS, LOW);
    
      for (int i = 0; i < numpckts; i++)
      {
       res.rawresults[i]= readData();
      }
      digitalWrite(CS,HIGH);
      /*
      char buffer[262];
      snprintf(buffer, 262, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d\n",
                output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8],
               output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8], packetloss);
      udp.beginPacket(udpAddress, udpPort);
      udp.print(buffer);
      udp.endPacket();
      packetloss++; 
    */
    
  }
  return res;
} 



/*--------------------------------------------------*/
/*----------------------  Deprecated functions ---------------------*/
/*--------------------------------------------------*/

void ADS1299::Task_data(void *param) // const
{
  // working

  while (1)
  {
    Serial.println("Hello from ADS task");
    vTaskDelay(3300 / portTICK_PERIOD_MS);
  }
}

/**
 * @brief Converts hexadecimal output of ADS to voltage value
 *
 * @param hexdata incoming 24 bit value
 * @return float output signed voltage in mV (LSB is ~nV)
 */
float ADS1299::convertHEXtoVolt(long hexdata)
{
  int sign = hexdata & 0x800000; // AND data with MSB sign; from 8 to F is -
  float voltage = 0.0;
  if (sign == 0)
  {
    voltage = hexdata * LSB;
  } // positive range
  else
  { // 2^24 negative range
    voltage = (hexdata - 0xFFFFFF) * LSB;
  }
  return voltage;
}

/**
 * @brief retrieve the most recent data
 *
 *
*/ 
// float *ADS1299::updateData(){  
//   // keep array memory adress
//   static float results_mV1[9]={0};
//   float results_mV[9]={0};
//   if (digitalRead(PIN_NUM_STRT == HIGH))
//   { // read only if data can be available
        
//         calculateLSB(1, 4.5); // Gain,4.5 Vref set
//       digitalWrite(CS, LOW);

//       long output[9]; // Output Array in Hex
//       long dataPacket = 0; //24 bit ADC reading
      
//       for (int i = 0; i < 9; i++)
//       {
//         for (int j = 0; j < 3; j++)
//         {
//           byte dataByte = SPI.transfer(0x00);        // issue 3 SCLK cycles to read 24 bit
//           dataPacket = (dataPacket << 8) | dataByte; // right shift each Byte
//         }
//         output[i] = dataPacket; // save 9 entries, 3 Bytes each
//         dataPacket = 0;
//       }

//       for (int i = 1; i < 9; i++)
//       { // exclude ADS status bits at i=0 [0]
//         // Serial.print(output[i], DEC);
//         results_mV1[i] = convertHEXtoVolt(output[i]); // convertHEXtoVolt()
//       }
      
      
//       char buffer[262];
//       snprintf(buffer, 262, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%ld\n",
//               results_mV[1], results_mV[2], results_mV[3], results_mV[4], results_mV[5], results_mV[6], results_mV[7], results_mV[8],
//                results_mV[1], results_mV[2], results_mV[3], results_mV[4], results_mV[5], results_mV[6], results_mV[7], results_mV[8], outputCount);
//       udp.beginPacket(udpAddress, udpPort);
//       udp.print(buffer);
//       udp.endPacket();
//       outputCount++; 
    
//     digitalWrite(CS, HIGH);
//   }
//   return results_mV1;
// }
 



void ADS1299::TI_setup()
{
  // TI SETUP https://e2e.ti.com/support/data-converters-group/data-converters/f/data-converters-forum/634236/ads1298-cannot-get-ecg-signal
  { /*
  CONFIG1 	0x46
  CONFIG2 	0x10
  CONFIG3 	0x41
  LOFF 	Default
  CH1Set 	0x81
  CH2Set 	0x60
  CH3-8Set 	0x81
  */

    RREG(CONFIG2);
    // WREG(CONFIG3,0xEC); // 6C: power down Buffer | EC: enable Bias
    activateTestSignals(CH4SET); // measure testsignal on CH4
    WREG(CH1SET, 0x00);          // measures normal on CH1
    WREG(CH2SET, 0x03);          // measures MVDD on CH2
    WREG(CH3SET, 0x01);          // shorted  on CH3
    // ADS1.activateTestSignals(CH4SET); //measure testsignal on CH4
    WREG(CH5SET, 0x01);     // shorted  on CH5
    WREG(CH6SET, 0x01);     // shorted  on CH6
    WREG(MISC1, 0x00);      // Not connect SRB1 to neg Electrodes
    WREG(BIAS_SENSN, 0x01); // CH1 n bias
    START();
    delay(1000);
  }
}

