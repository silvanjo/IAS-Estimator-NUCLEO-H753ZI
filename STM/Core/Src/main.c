/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "arm_math.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define ADC_BUFFER_SIZE 2048
#define FFT_SIZE ADC_BUFFER_SIZE

/* MOPA Algorithm parameters */
#define SAMPLE_RATE 2000.0f        /* Sample rate in Hz */
#define MIN_OMEGA 10.0f            /* Minimum angular frequency (Hz) */
#define MAX_OMEGA 100.0f           /* Maximum angular frequency (Hz) */
#define N_OMEGA 100                /* Number of omega candidates */
#define N_ORDERS 3                 /* Number of harmonic orders */
#define SIGMA_HZ 10.0f             /* Gaussian prior smoothing parameter */
#define PRIOR_WEIGHT 2.0f          /* Weight for Gaussian prior */

/* Voltage measurement */
#define VREF 3.3f
#define ADC_MAX 65535.0f           /* 16-bit ADC max value (2^16 - 1) */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

COM_InitTypeDef BspCOMInit;
ADC_HandleTypeDef hadc1;
DMA_HandleTypeDef hdma_adc1;

TIM_HandleTypeDef htim3;

/* USER CODE BEGIN PV */

/* ADC DMA buffer - must be in AXI SRAM (DMA cannot access DTCMRAM on H7) */
__attribute__((section(".ram_d1"))) uint16_t adc_input_buffer[ADC_BUFFER_SIZE];

volatile uint8_t buffer_half_full_flag = 0;
volatile uint8_t buffer_full_flag = 0;

// FFT buffers and instance
float32_t fft_input[FFT_SIZE];
float32_t fft_output[FFT_SIZE * 2];  // Complex output: real and imaginary
float32_t fft_magnitude[FFT_SIZE];
float32_t hann_window[FFT_SIZE];

arm_rfft_fast_instance_f32 fft_instance;

/* MOPA Algorithm buffers */
static const uint8_t mopa_orders[N_ORDERS] = {1, 2, 3};  /* Harmonic orders to consider */
float32_t mopa_omega[N_OMEGA];              /* Omega candidate values */
float32_t mopa_pdf[N_OMEGA];                /* PDF for current frame */
float32_t mopa_biased_pdf[N_OMEGA];         /* Biased PDF with Gaussian prior */
float32_t mopa_prev_ias = 0.0f;             /* Previous IAS estimate for smoothing */
float32_t mopa_current_ias = 0.0f;          /* Current IAS estimate */
volatile uint8_t mopa_ias_ready = 0;        /* Flag: IAS estimate ready */
uint8_t mopa_initialized = 0;               /* Flag: MOPA omega vector initialized */

/* Spectrum parameters (calculated from FFT) */
float32_t spectrum_df = 0.0f;               /* Frequency resolution (Hz per bin) */
float32_t spectrum_max_freq = 0.0f;         /* Maximum frequency in spectrum */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM3_Init(void);

/* USER CODE BEGIN PFP */

static void mopa_init(void);
static void normalize_spectrum(void);
static float32_t interp_spectrum(float32_t freq);
static void compute_pdf_map(void);
static float32_t extract_ias(void);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/*
HELPER FUNCTIONS
*/

// Initialize Hann window coefficients
void init_hann_window(void)
{
  const float32_t pi = 3.14159265358979323846f;
  for (uint32_t i = 0; i < FFT_SIZE; i++)
  {
    // Hann window: w(n) = 0.5 * (1 - cos(2*pi*n/(N-1)))
    hann_window[i] = 0.5f * (1.0f - arm_cos_f32(2.0f * pi * i / (FFT_SIZE - 1)));
  }
}

/*
MOPA FUNCTIONS
*/

// Process FFT on ADC buffer segment with Hann windowing
void process_fft(uint16_t* input, uint32_t offset)
{
  uint32_t half_size = FFT_SIZE / 2;
  uint32_t old_data_offset;

  // Determine the offset of the old data
  if (offset == 0) {
      old_data_offset = half_size;
  } else {
      old_data_offset = 0;
  }

  /* Construct Input Buffer: [OLD DATA] + [NEW DATA] */

  // First, copy data without windowing to calculate DC offset
  float32_t dc_sum = 0.0f;
  for (uint32_t i = 0; i < half_size; i++)
  {
      fft_input[i] = (float32_t) input[old_data_offset + i];
      dc_sum += fft_input[i];
  }
  for (uint32_t i = 0; i < half_size; i++)
  {
      fft_input[i + half_size] = (float32_t) input[offset + i];
      dc_sum += fft_input[i + half_size];
  }

  // Calculate and remove DC offset
  float32_t dc_offset = dc_sum / (float32_t)FFT_SIZE;

  // Remove DC offset and apply Hann window
  for (uint32_t i = 0; i < FFT_SIZE; i++)
  {
      fft_input[i] = (fft_input[i] - dc_offset) * hann_window[i];
  }

  // Perform FFT
  arm_rfft_fast_f32(&fft_instance, fft_input, fft_output, 0);

  // Calculate magnitude spectrum
  arm_cmplx_mag_f32(fft_output, fft_magnitude, FFT_SIZE / 2);

  // Normalize the magnitude spectrum (for FFT scaling)
  float32_t norm_factor = 2.0f / (float32_t)FFT_SIZE;
  for (uint32_t i = 0; i < FFT_SIZE / 2; i++) {
    fft_magnitude[i] *= norm_factor;
  }

  // Run MOPA algorithm
  if (mopa_initialized) {
    // Normalize spectrum for MOPA (RMS normalization)
    normalize_spectrum();

    // Compute PDF map
    compute_pdf_map();

    // Extract IAS estimate
    mopa_current_ias = extract_ias();
    mopa_prev_ias = mopa_current_ias;
    mopa_ias_ready = 1;
  }
}

/* Initialize MOPA algorithm - compute omega vector and spectrum parameters */
static void mopa_init(void) {
  /* Compute omega candidate vector */
  float32_t d_omega = (MAX_OMEGA - MIN_OMEGA) / (float32_t)(N_OMEGA - 1);
  for (uint16_t i = 0; i < N_OMEGA; i++) {
    mopa_omega[i] = MIN_OMEGA + i * d_omega;
  }

  /* Compute spectrum parameters based on FFT size and sample rate */
  spectrum_df = SAMPLE_RATE / (float32_t)FFT_SIZE;
  spectrum_max_freq = spectrum_df * ((float32_t)FFT_SIZE / 2.0f - 1.0f);

  mopa_initialized = 1;
}

/* Normalize the magnitude spectrum by its RMS value */
static void normalize_spectrum(void) {
  float32_t sum_sq = 0.0f;
  uint16_t n_bins = FFT_SIZE / 2;

  /* Calculate sum of squares */
  for (uint16_t i = 0; i < n_bins; i++) {
    sum_sq += fft_magnitude[i] * fft_magnitude[i];
  }

  /* Calculate RMS and normalize */
  float32_t rms;
  arm_sqrt_f32(sum_sq / (float32_t)n_bins, &rms);

  if (rms > 1e-10f) {
    float32_t inv_rms = 1.0f / rms;
    for (uint16_t i = 0; i < n_bins; i++) {
      fft_magnitude[i] *= inv_rms;
    }
  }
}

/* Linear interpolation of spectrum at a given frequency */
static float32_t interp_spectrum(float32_t freq) {
  /* Handle boundary cases */
  if (freq <= 0.0f) {
    return fft_magnitude[0];
  }
  if (freq >= spectrum_max_freq) {
    return fft_magnitude[FFT_SIZE / 2 - 1];
  }

  /* Calculate bin index */
  float32_t bin_float = freq / spectrum_df;
  uint16_t idx = (uint16_t)bin_float;

  if (idx >= FFT_SIZE / 2 - 1) {
    idx = FFT_SIZE / 2 - 2;
  }

  /* Linear interpolation */
  float32_t f0 = idx * spectrum_df;
  float32_t f1 = (idx + 1) * spectrum_df;
  float32_t v0 = fft_magnitude[idx];
  float32_t v1 = fft_magnitude[idx + 1];

  float32_t t = (freq - f0) / (f1 - f0);
  return v0 + t * (v1 - v0);
}

/* Compute PDF map for all omega candidates */
static void compute_pdf_map(void) {
  float32_t max_log = -1e30f;

  /* Compute log-PDF for each omega candidate */
  for (uint16_t i = 0; i < N_OMEGA; i++) {
    float32_t w = mopa_omega[i];
    float32_t log_pdf = 0.0f;

    /* Sum log of spectrum values at harmonic frequencies */
    for (uint8_t o = 0; o < N_ORDERS; o++) {
      float32_t freq = mopa_orders[o] * w;
      if (freq < spectrum_max_freq) {
        float32_t s = interp_spectrum(freq);
        if (s < 1e-10f) s = 1e-10f;
        log_pdf += logf(s);
      }
    }

    mopa_pdf[i] = log_pdf;
    if (log_pdf > max_log) {
      max_log = log_pdf;
    }
  }

  /* Convert to linear scale and normalize */
  float32_t sum = 0.0f;
  for (uint16_t i = 0; i < N_OMEGA; i++) {
    float32_t val = expf(mopa_pdf[i] - max_log);
    mopa_pdf[i] = val;
    sum += val;
  }

  if (sum > 0.0f) {
    float32_t inv_sum = 1.0f / sum;
    for (uint16_t i = 0; i < N_OMEGA; i++) {
      mopa_pdf[i] *= inv_sum;
    }
  }
}

/* Extract IAS estimate using Gaussian prior smoothing */
static float32_t extract_ias(void) {
  float32_t max_val = 0.0f;
  uint16_t peak_idx = 0;

  if (mopa_prev_ias == 0.0f) {
    /* First frame: simple argmax */
    for (uint16_t i = 0; i < N_OMEGA; i++) {
      if (mopa_pdf[i] > max_val) {
        max_val = mopa_pdf[i];
        peak_idx = i;
      }
    }
  } else {
    /* Subsequent frames: apply Gaussian prior biasing */
    float32_t sigma_sq_2 = 2.0f * SIGMA_HZ * SIGMA_HZ;

    for (uint16_t i = 0; i < N_OMEGA; i++) {
      float32_t w = mopa_omega[i];
      float32_t diff = w - mopa_prev_ias;
      float32_t gaussian = expf(-(diff * diff) / sigma_sq_2);
      mopa_biased_pdf[i] = mopa_pdf[i] * (1.0f + PRIOR_WEIGHT * gaussian);

      if (mopa_biased_pdf[i] > max_val) {
        max_val = mopa_biased_pdf[i];
        peak_idx = i;
      }
    }
  }

  return mopa_omega[peak_idx];
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_ADC1_Init();
  MX_TIM3_Init();
  /* USER CODE BEGIN 2 */

  // Initialize FFT instance
  arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE);

  // Initialize Hann window coefficients
  init_hann_window();

  // Initialize MOPA algorithm
  mopa_init();

  /* USER CODE END 2 */

  /* Initialize leds */
  BSP_LED_Init(LED_GREEN);
  BSP_LED_Init(LED_BLUE);
  BSP_LED_Init(LED_RED);

  /* Initialize USER push-button, will be used to trigger an interrupt each time it's pressed.*/
  BSP_PB_Init(BUTTON_USER, BUTTON_MODE_EXTI);

  /* Initialize COM1 port (115200, 8 bits (7-bit data + 1 stop bit), no parity */
  BspCOMInit.BaudRate   = 115200;
  BspCOMInit.WordLength = COM_WORDLENGTH_8B;
  BspCOMInit.StopBits   = COM_STOPBITS_1;
  BspCOMInit.Parity     = COM_PARITY_NONE;
  BspCOMInit.HwFlowCtl  = COM_HWCONTROL_NONE;
  if (BSP_COM_Init(COM1, &BspCOMInit) != BSP_ERROR_NONE)
  {
    Error_Handler();
  }

  {
    const char *msg = "MOPA algorithm initialized\r\n";
    HAL_UART_Transmit(&hcom_uart[COM1], (uint8_t*)msg, strlen(msg), HAL_MAX_DELAY);
  }

  // Calibrate ADC
  HAL_ADCEx_Calibration_Start(&hadc1, ADC_CALIB_OFFSET, ADC_SINGLE_ENDED);

  // Start ADC with DMA in circular mode
  HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc_input_buffer, ADC_BUFFER_SIZE);

  // Start Timer3 to trigger ADC conversions at 2 kHz
  HAL_TIM_Base_Start(&htim3);

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */

  while (1)
  {
    if (mopa_ias_ready) {
      mopa_ias_ready = 0;
      // Convert float to integer 
      int ias_int = (int)mopa_current_ias;
      int ias_frac = (int)((mopa_current_ias - ias_int) * 100);
      if (ias_frac < 0) ias_frac = -ias_frac;

      char uart_buf[32];
      int len = sprintf(uart_buf, "IAS: %d.%02d Hz\r\n", ias_int, ias_frac);
      HAL_UART_Transmit(&hcom_uart[COM1], (uint8_t*)uart_buf, len, HAL_MAX_DELAY);
    }

    // Check if ADC buffer is half full
    if (buffer_half_full_flag)
    {
      buffer_half_full_flag = 0;
      process_fft(adc_input_buffer, 0);
    }

    // Check if ADC buffer is full
    if (buffer_full_flag)
    {
      buffer_full_flag = 0;
      process_fft(adc_input_buffer, ADC_BUFFER_SIZE / 2);
    }
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 60;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_MultiModeTypeDef multimode = {0};
  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Common config
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV4;
  hadc1.Init.Resolution = ADC_RESOLUTION_16B;
  hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc1.Init.LowPowerAutoWait = DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConv = ADC_EXTERNALTRIG_T3_TRGO;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_RISING;
  hadc1.Init.ConversionDataManagement = ADC_CONVERSIONDATA_DMA_CIRCULAR;
  hadc1.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  hadc1.Init.LeftBitShift = ADC_LEFTBITSHIFT_NONE;
  hadc1.Init.OversamplingMode = DISABLE;
  hadc1.Init.Oversampling.Ratio = 1;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the ADC multi-mode
  */
  multimode.Mode = ADC_MODE_INDEPENDENT;
  if (HAL_ADCEx_MultiModeConfigChannel(&hadc1, &multimode) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_2;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_1CYCLE_5;
  sConfig.SingleDiff = ADC_SINGLE_ENDED;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  sConfig.OffsetSignedSaturation = DISABLE;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 119;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 999;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim3, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_UPDATE;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Stream0_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream0_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOF_CLK_ENABLE();

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

// DMA half-transfer complete callback
void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef* hadc)
{
  if (hadc->Instance == ADC1)
  {
    buffer_half_full_flag = 1;
  }
}

// DMA transfer complete callback
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc)
{
  if (hadc->Instance == ADC1)
  {
    buffer_full_flag = 1;
  }
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
