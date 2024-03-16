#ifndef _config_h
#define _config_h

#define PIN_CS_1 10 // Pin No. 11 low active
#define PIN_NUM_RST 4 // active LOW
#define PIN_NUM_STRT 2 // HIGH active
#define PIN_NUM_DRDY_1 15 // active LOW
#define PIN_NUM_PWD 16 // active LOW
// #define PIN_NUM_CLKSEL 5 
#define PIN_BAT_MON 5
#define PIN_IMU_SCL 6
#define PIN_IMU_SDA 7
#define led_red 36
#define led_green 35
//SCLK = 12, MISO = 13, MOSI = 11, SS = 10
#define VSPI_MISO   13
#define VSPI_MOSI   11
#define VSPI_SCLK   12
// #define VSPI_SS     SS 
#define spiClk 5000000//2400000 //valid: 2.4 Mhz; to 1 MHz
#define TCLK_cycle (1) //us for 1MHz SPIclk


#define SPI_DATA_MODE 0x04     //clock polarity = 0; clock phase = 1 (pg. 8)
#define SPI_MODE_MASK 0x0C     // CPOL = bit 3, CPHA = bit 2 on SPCR
#define SPI_CLOCK_MASK 0x03    // SPR1 = bit 1, SPR0 = bit 0 on SPCR
#define SPI_2XCLOCK_MASK 0x01  // SPI2X = bit 0 on SPSR

// ADS Registers
//SPI Command Definition Byte Assignments (Datasheet, pg. 35)
#define _WAKEUP 0x02   // Wake-up from standby mode
#define _STANDBY 0x04  // Enter Standby mode
#define _RESET 0x06    // Reset the device, 
#define _START 0x08    // Start and restart (synchronize) conversions
#define _STOP 0x0A     // Stop conversion
#define _RDATAC 0x10   // Enable Read Data Continuous mode (default mode at power-up)
#define _SDATAC 0x11   // Stop Read Data Continuous mode
#define _RDATA 0x12    // Read data by command; supports multiple read back

#define _RREG 0x20  // (also = 00100000) is the first opcode that the address must be added to for RREG communication
#define _WREG 0x40  // 01000000 in binary (Datasheet, pg. 35)

//Register Addresses
#define ID 0x00
#define CONFIG1 0x01
#define CONFIG2 0x02
#define CONFIG3 0x03
#define LOFF 0x04
#define CH1SET 0x05
#define CH2SET 0x06
#define CH3SET 0x07
#define CH4SET 0x08
#define CH5SET 0x09
#define CH6SET 0x0A
#define CH7SET 0x0B
#define CH8SET 0x0C
#define BIAS_SENSP 0x0D
#define BIAS_SENSN 0x0E
#define LOFF_SENSP 0x0F
#define LOFF_SENSN 0x10
#define LOFF_FLIP 0x11
#define LOFF_STATP 0x12
#define LOFF_STATN 0x13
#define GPIO 0x14
#define MISC1 0x15
#define MISC2 0x16
#define CONFIG4 0x17

// SETTINGS for Registers
// ads1299 config 1 register bits
#define CFG1RES7                                0x80    // 10000000
#define DAISY_EN                                0x40    // 01000000
#define CLK_EN                                  0x20    // 00100000
#define CFG1RES43                               0x10    // 00010000
#define SAMPLING_RATE_00250HZ                   0x06    // 00000110
#define SAMPLING_RATE_00500HZ                   0x05    // 00000101
#define SAMPLING_RATE_01000HZ                   0x04    // 00000100
#define SAMPLING_RATE_02000HZ                   0x03    // 00000011
#define SAMPLING_RATE_04000HZ                   0x02    // 00000010
#define SAMPLING_RATE_08000HZ                   0x01    // 00000001
#define SAMPLING_RATE_16000HZ                   0x00    // 00000000
  
// ads1299 config 2 register bits  
#define CFG2RES75                               0xC0   // 11000000 
#define INT_CAL                                 0x10   // 00010000
#define CFG2RES3                                0x00   // 00000000
#define CAL_AMP                                 0x04   // 00000100
#define CAL_FREQ_221                            0x00   // 00000000 
#define CAL_FREQ_220                            0x01   // 00000001
#define CAL_FREQ_DC                             0x03   // 00000011
 
// ads1299 config 3 register bits
#define PD_REFBUF                               0x80 // 10000000
#define CFG3RES_65                              0x42 // 01100000
#define BIAS_MEAS                               0x00 // 00010000
#define BIASREF_INT                             0x08 // 00001000
#define PD_BIAS                                 0x04 // 00000100
#define BIAS_LOFF_SENS                          0x00 // 00000000
#define BIAS_STAT                               0x01 // 00000001

// ads1299 config 4 register bits
#define CFG4RES_74                              0xF0 // 11110000
#define SINGLE_SHOT                             0x08 // 00010000 
#define CFG4RES_2                               0x04 // 00000100
#define PD_LOFF_COMP                            0x02 // 00000010
#define CFG4RES_0                               0x00 // 00000001

// ads1299 individual channel settings bits
#define CHANNEL_POWER_DOWN                      0x80 // 10000000
#define SRB2                                    0x08  //00001000

#define ADS1299_INPUT_NORMAL                    0x00 // 00000000
#define ADS1299_INPUT_SHORTED                   0x01 // 00000001
#define ADS1299_INPUT_MEAS_BIAS                 0x02 // 00000010
#define ADS1299_INPUT_SUPPLY                    0x03 // 00000011
#define ADS1299_INPUT_TEMP                      0x04 // 00000100
#define ADS1299_INPUT_TESTSIGNAL                0x05 // 00000101
#define ADS1299_INPUT_SET_BIASP                 0x06 // 00000110
#define ADS1299_INPUT_SET_BIASN                 0x07 // 00000111

//Gain
#define ADS1299_PGA_GAIN01                      0x00 // 00000000
#define ADS1299_PGA_GAIN02                      0x10 // 00010000
#define ADS1299_PGA_GAIN04                      0x20 // 00100000
#define ADS1299_PGA_GAIN06                      0x30 // 00110000
#define ADS1299_PGA_GAIN08                      0x40 // 01000000
#define ADS1299_PGA_GAIN12                      0x50 // 01010000
#define ADS1299_PGA_GAIN24                      0x60 // 01100000

#define GAIN  1
#define SCALE_FACT 1000000 *((4.5/8388607)/GAIN)  // Vref= 4.5V; 2^23-1 = 8388607

struct LEDparameter{
int brightness = 0;  // how bright the LED is
int fadeAmount = 5;
int PWM_FREQUENCY = 1000;  // this variable is used to define the time period
int PWM_CHANNEL = 0;       // this variable is used to select the channel number
int PWM_RESOUTION = 8;     // this will define the resolution of the signal which is 8 in this case
int dutyCycle = 127;       // it will define the width of signal or also the on time
long outputCount =0;
};

#endif