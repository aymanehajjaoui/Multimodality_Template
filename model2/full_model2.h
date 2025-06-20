#pragma once
#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#define True 1
#define False 0

#define _clamp_to(type, number) clamp_to_number_t_ ## type (number)
#define clamp_to(type, number) _clamp_to(type, number)
#define _scale(type, number, scale_factor) scale_number_t_ ## type (number, scale_factor)
#define scale(type, number, scale_factor) _scale(type, number, scale_factor)

// Idea 1: Write the smallest min max interval of the net, could be an issue for hybrid int type network
// Idea 2: listing any interval and add type in name in a switch case like <- better but painfull
// #define NUMBER_MIN		// Max value for this numeric type
// #define NUMBER_MAX		// Min value for this numeric type

// // Idea 1: List of all types and write any corresponding function 
// typedef  number_t;		// Standard size numeric type used for weights and activations
// typedef  long_number_t;	// Long numeric type used for intermediate results

#define NUMBER_MIN_INT8_T -128
#define NUMBER_MAX_INT8_T 127

static inline int16_t min_int8_t(
    int16_t a,
    int16_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int16_t max_int8_t(
    int16_t a,
    int16_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int16_t scale_number_t_int8_t(
  int16_t number, int scale_factor) {
  if (scale_factor < 0)
    return number << - scale_factor;
  else 
    return number >> scale_factor;
}
static inline int8_t clamp_to_number_t_int8_t(
  int16_t number) {
	return (int8_t) max_int8_t(
      NUMBER_MIN_INT8_T,
      min_int8_t(
        NUMBER_MAX_INT8_T, number));
}

#define NUMBER_MIN_INT32_T -2147483648
#define NUMBER_MAX_INT32_T 2147483647

static inline int64_t min_int32_t(
    int64_t a,
    int64_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int64_t max_int32_t(
    int64_t a,
    int64_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int64_t scale_number_t_int32_t(
  int64_t number, int scale_factor) {
  if (scale_factor < 0)
    return number << - scale_factor;
  else 
    return number >> scale_factor;
}
static inline int32_t clamp_to_number_t_int32_t(
  int64_t number) {
	return (int32_t) max_int32_t(
      NUMBER_MIN_INT32_T,
      min_int32_t(
        NUMBER_MAX_INT32_T, number));
}




static inline void int64_t_to_float(int64_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int32_t_to_float(int32_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int16_t_to_float(int16_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}

static inline void int8_t_to_float(int8_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}
#endif //__NUMBER_H__

#ifdef __cplusplus
} // extern "C"
#endif
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_H_
#define _CONV1D_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       48
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int8_t conv1d_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       48
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 6
#define INPUT_SCALE_FACTOR 6
#define OUTPUT_SCALE_FACTOR 6
#define NUMBER_T int8_t
#define LONG_NUMBER_T int16_t


static inline void conv1d(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 

      output_acc = scale(NUMBER_T, (LONG_NUMBER_T)bias[k], -INPUT_SCALE_FACTOR);


      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }
      
#ifdef ACTIVATION_LINEAR
      output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
        output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
        output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
      }
#endif
    }
  }

#else



  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 4 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q7_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q7_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q7_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q7_basic_nonsquare(
#endif
#endif
                                      (q7_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q7_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q7_t*)bias, //bias
                                      INPUT_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q7_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q7((q7_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q7((q7_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    1
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3


const int8_t  conv1d_bias[CONV_FILTERS] = {-7, -12, -3, 33, -8, 17, 19, 12, -20, 10, 37, -9, 2, 21, 22, 1}
;

const int8_t  conv1d_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS] = {{{8}
, {-2}
, {20}
}
, {{-21}
, {-14}
, {-19}
}
, {{-9}
, {12}
, {7}
}
, {{13}
, {8}
, {-6}
}
, {{7}
, {15}
, {-14}
}
, {{-21}
, {18}
, {-4}
}
, {{19}
, {-11}
, {6}
}
, {{-1}
, {17}
, {-24}
}
, {{22}
, {15}
, {9}
}
, {{-11}
, {-14}
, {2}
}
, {{11}
, {12}
, {4}
}
, {{-19}
, {5}
, {20}
}
, {{26}
, {-1}
, {-20}
}
, {{-16}
, {-11}
, {19}
}
, {{18}
, {10}
, {28}
}
, {{7}
, {-23}
, {18}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_H_
#define _BATCH_NORMALIZATION_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       46

typedef int8_t batch_normalization_output_type[46][16];

#if 0
void batch_normalization(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       46
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 6
#define INPUT_SCALE_FACTOR 6
#define OUTPUT_SCALE_FACTOR 6
#define NUMBER_T int8_t
#define LONG_NUMBER_T int16_t


static inline void batch_normalization(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = scale(NUMBER_T, (LONG_NUMBER_T)bias[z], -INPUT_SCALE_FACTOR);
      tmp += (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      tmp = scale(NUMBER_T, tmp, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[x][z] = clamp_to(NUMBER_T, tmp);
#elif defined(ACTIVATION_RELU)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
        tmp = scale(NUMBER_T, tmp, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
        output[x][z] = clamp_to(NUMBER_T, tmp);
      }
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int8_t batch_normalization_bias[16] = {-20, -32, -23, -35, -11, -34, -12, -41, -17, -27, -25, -18, -18, -28, -24, -5}
;
const int8_t batch_normalization_kernel[16] = {39, 18, 58, 48, 69, 53, 47, 79, 21, 42, 32, 65, 54, 56, 22, 54}
;
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_H_
#define _MAX_POOLING1D_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   46
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int8_t max_pooling1d_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   46
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 6
#define OUTPUT_SCALE_FACTOR 6
#define NUMBER_T int8_t
#define LONG_NUMBER_T int16_t


static inline void max_pooling1d(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef WITH_CMSIS_NN
// Not really CMSIS-NN since using arm_relu_q* is not more efficient, but use SSAT anyway
#if ACC_SCALE_FACTOR - OUTPUT_SCALE_FACTOR > 0
      output[pos_x][k] = __SSAT(max[k] >> (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#else
      output[pos_x][k] = __SSAT(max[k] << (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#endif
#else
      max[k] = scale(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, max[k]);
#endif
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_1_H_
#define _CONV1D_1_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       23
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int8_t conv1d_1_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_1(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_1_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_1.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       23
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 6
#define INPUT_SCALE_FACTOR 6
#define OUTPUT_SCALE_FACTOR 6
#define NUMBER_T int8_t
#define LONG_NUMBER_T int16_t


static inline void conv1d_1(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 

      output_acc = scale(NUMBER_T, (LONG_NUMBER_T)bias[k], -INPUT_SCALE_FACTOR);


      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }
      
#ifdef ACTIVATION_LINEAR
      output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
        output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
        output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
      }
#endif
    }
  }

#else



  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 4 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q7_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q7_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q7_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q7_basic_nonsquare(
#endif
#endif
                                      (q7_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q7_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q7_t*)bias, //bias
                                      INPUT_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q7_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q7((q7_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q7((q7_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3


const int8_t  conv1d_1_bias[CONV_FILTERS] = {10, 8, 2, -14, 4, -11, 9, -2, -5, -8, -1, -6, 1, 2, 5, -2, -9, 21, 2, -21, 23, -9, 1, -13, 5, -9, 6, -5, 0, -14, -8, 2}
;

const int8_t  conv1d_1_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS] = {{{2, -14, -4, 4, -6, 2, 10, 3, -1, -8, 8, 2, 9, 8, -13, -1}
, {1, -12, -6, 9, 4, -5, 10, 15, 8, -30, -1, 0, -2, -22, 9, -18}
, {-20, 20, -13, 4, 17, -13, -7, 27, -16, 7, -5, 25, 35, 12, -16, -6}
}
, {{-7, -20, 14, -6, -10, -3, 0, -20, 4, -20, 9, -17, -12, -35, -3, -20}
, {2, -25, -18, -23, -24, -12, -5, 17, 0, -22, -13, -24, -9, -3, 10, 20}
, {15, -6, 10, -7, -8, 10, 30, 12, 1, -19, -4, 14, 3, 1, 9, 13}
}
, {{8, -4, -6, 0, 1, -6, -16, 5, 14, 4, 11, 18, 9, 3, 2, 3}
, {16, -4, 1, -3, 14, 7, -11, 6, -14, 19, -4, -26, 8, 23, 5, 13}
, {24, -8, -32, -27, -22, -1, 3, -50, -12, -21, -23, -31, -38, -3, -3, 0}
}
, {{-23, -4, -2, -18, -16, -8, -29, 5, -10, 0, -10, 1, -22, 3, -15, 0}
, {9, 6, 17, 1, -16, -2, -1, -24, 19, -8, 18, 12, -9, -4, 17, -10}
, {14, -2, -3, -6, 4, -15, -15, -18, -8, 8, -16, 8, 2, 19, 3, 15}
}
, {{-4, -4, -1, -13, 11, 12, -14, 0, -18, 17, -16, 24, 7, 23, -17, 0}
, {19, -9, 11, 1, 1, 6, 4, -12, -11, -5, 2, 9, -14, 18, 15, 11}
, {7, -1, 5, 11, 0, -22, -1, 2, -12, -13, -3, 13, 0, -22, -9, 6}
}
, {{-11, -9, -5, 2, -8, 8, -9, 15, -34, -27, -14, 17, -6, -5, -1, 11}
, {13, -14, -7, 7, 5, 6, 11, -4, -4, -10, -9, -11, -1, -15, 1, 10}
, {-21, 19, 2, -15, -7, -5, -21, -9, -6, 36, -23, 0, 2, -11, -15, 16}
}
, {{11, -18, -1, 15, 5, 20, 7, 17, 13, -1, 2, -2, 11, 6, -1, 19}
, {16, 8, -12, -6, -10, 8, 2, -21, 10, -3, 13, 6, -15, -1, 11, -2}
, {2, -29, -4, 6, 1, -35, 2, 3, -2, 3, -7, -11, 9, 10, 20, 6}
}
, {{3, -10, 7, 1, 12, 2, 8, -10, 13, 18, 11, -2, 10, 35, 5, 6}
, {-16, -18, -20, 3, 7, -6, 19, -7, -6, 5, 0, -21, 21, 8, -7, 3}
, {-1, -42, 5, -4, 14, 13, 1, -6, -30, -16, -17, 0, -11, 12, -24, 7}
}
, {{6, -6, 6, -4, 18, 27, 7, -5, -14, -8, 8, 23, 10, 5, 1, 6}
, {1, 16, -10, 9, -14, -29, -5, -17, 0, -5, 8, 3, -20, -57, -3, -64}
, {-15, 8, 26, -6, -1, -22, -10, -9, 10, -2, 2, 7, -9, -11, -11, 31}
}
, {{22, 23, -2, -32, -37, 4, -3, -52, 1, 9, 2, -9, -29, -8, 15, -18}
, {-2, -13, 15, -3, 11, 11, -7, 5, 6, 4, 20, 9, -1, -4, -10, 3}
, {-4, -19, -11, -10, 16, 4, 14, -14, -14, -38, -19, -18, 11, -3, 4, 9}
}
, {{13, -10, 0, -1, 9, 12, 1, 2, 4, -11, 13, 10, 0, 1, 12, -18}
, {-3, -22, -3, 2, -7, 3, 9, 2, -6, 6, -1, 2, 3, 6, -25, 44}
, {2, 9, -17, -9, -21, 19, 19, -11, 5, -26, 8, -13, -30, -21, 19, -30}
}
, {{-23, 6, 2, 12, 11, 14, 5, -7, -4, 13, -7, -5, 5, 30, -22, -13}
, {-2, -5, -7, -23, 6, 20, 0, -13, -7, 22, -19, -17, -1, 11, 7, 1}
, {-21, -21, -2, 21, 14, -2, -8, -2, -20, -18, 10, -4, 18, 6, -16, 0}
}
, {{-11, -10, 9, 11, 6, -20, -2, -12, -7, 4, 6, -14, 21, -21, 14, 9}
, {10, 10, -11, -20, -21, 5, 12, -20, 5, -23, 8, -11, 6, -21, -11, -1}
, {-2, -9, -6, -8, 11, 2, -3, 28, -14, 8, -19, 15, 20, -10, -18, 3}
}
, {{20, 3, -2, 11, 16, -20, 11, 9, -11, -9, 7, -6, 1, 13, 18, 0}
, {6, -6, 2, -13, -5, 0, 7, -9, 21, -12, -3, -17, -19, -42, -6, -35}
, {-39, 6, 5, -20, -38, -13, -30, 33, -14, 23, -25, 10, -16, 0, -22, 2}
}
, {{19, 21, -13, -11, 5, -24, -24, 4, 17, 10, 15, -30, -11, -24, 20, -20}
, {-17, -10, -13, -4, -1, -11, 10, -2, -7, -15, -3, -2, 10, 4, -18, 24}
, {21, 30, 1, -24, -12, -1, 0, 3, 8, 9, -13, 4, -11, 2, 0, -12}
}
, {{-1, -11, -1, 15, 13, -17, 12, -10, -18, -31, 10, 0, 19, -30, -13, -14}
, {-25, -11, 13, -12, 9, -13, -23, 1, -3, 16, -13, 24, 5, 5, -16, -2}
, {-22, 11, 17, -16, -2, -8, 2, -9, -11, 10, -12, 2, 3, 6, -2, -7}
}
, {{7, -15, 4, 20, 12, 15, -11, -3, -8, -26, -9, 4, 10, 9, -3, 16}
, {-11, 6, -13, -18, -22, 3, 13, -73, 0, -13, -4, -18, -19, 7, 1, -13}
, {21, 11, 8, -1, 15, -23, -29, -15, 2, 7, 16, 15, -17, -7, 14, 6}
}
, {{-17, 0, 4, -19, -21, -3, 1, 3, -3, -1, -15, 0, -6, 6, -20, -3}
, {-5, 0, 6, -1, 10, 15, 2, 32, 0, -15, -5, 5, -1, -15, -17, -1}
, {18, 9, 8, 13, -11, -5, 8, -15, 6, 26, 9, 12, -1, 27, 15, 13}
}
, {{-24, 3, -9, -14, 0, 18, -7, -20, -1, 21, -6, 13, 2, 15, -22, 13}
, {-4, -10, -18, -17, -9, 1, 15, 0, -10, -12, -1, -18, -25, -2, 1, 8}
, {3, 1, 9, 5, -23, -5, 2, -9, 1, 3, -12, 1, -2, 12, -19, 21}
}
, {{-8, 1, 3, 15, 11, 7, 7, -9, 8, 8, 3, -18, 25, -13, 1, 11}
, {8, -4, -9, -4, -25, -28, 8, 16, 20, -10, -15, -36, -15, -45, 1, -10}
, {-12, -18, -17, -8, -20, -11, 5, 2, -14, -14, 2, -2, -16, -10, 16, 16}
}
, {{6, 1, -22, -19, 7, -22, -16, 7, -21, 7, -5, -16, 5, 4, 1, 16}
, {-3, 26, -6, -18, -5, -21, -1, -25, -14, 22, -16, -3, 14, 11, 7, 1}
, {6, -10, -4, -6, 0, -3, 2, 2, 8, 3, 10, -9, -10, -3, -1, 0}
}
, {{11, 16, 2, 20, 4, -18, 10, -18, 22, -8, -3, -19, 3, -32, 13, -41}
, {5, 8, -3, -7, 2, -16, -21, 15, 10, 3, 17, -8, -1, 1, -4, 17}
, {-18, -17, 4, 6, 11, 18, 11, 5, -12, -19, -1, -3, 2, 6, 5, -2}
}
, {{7, -3, -1, 15, 26, 16, 0, 4, -29, -5, -8, -1, 24, 11, 7, 10}
, {-18, 1, 8, -14, -17, 2, 9, 11, -13, 10, -18, 3, -10, 2, -26, -13}
, {5, -2, -7, 3, -6, -3, 13, 23, -22, -9, -2, -1, -12, -19, -6, 14}
}
, {{2, -5, -10, -11, -8, 21, 15, 17, -9, -18, -29, -7, -12, -5, -5, -5}
, {6, -12, -3, 15, -8, -1, 4, 0, -12, 0, -2, -18, -10, 0, 7, 15}
, {4, -12, -20, -5, -16, 19, 27, 2, -1, -12, -11, -15, -16, -2, -12, 18}
}
, {{1, -22, 0, -9, 14, 28, -2, 8, -4, 16, -1, 0, 10, 36, -9, 10}
, {7, 1, -4, -8, 4, 11, -6, -12, -4, -5, 14, 0, 7, -5, 7, -25}
, {5, 25, -21, -1, -19, -18, -7, 1, -2, 3, 1, -42, -16, -25, 13, -13}
}
, {{-3, -1, -14, -20, -10, -6, -1, 2, -9, -19, -17, -27, -25, -1, -2, 2}
, {-4, -1, -8, -14, -24, 3, -5, 22, -9, -8, -15, 9, -42, -2, -5, -10}
, {-19, -6, 1, -2, -2, 2, -5, -8, 1, 12, -10, 10, -4, 11, -4, 12}
}
, {{-7, 2, 2, 17, -6, 16, 5, -3, -4, -2, 5, 7, -7, -2, -9, -12}
, {2, 4, 9, 2, 4, 4, 11, 12, -29, 8, -17, 3, -7, 15, -26, 10}
, {2, -32, 14, -15, -21, 7, 0, -21, -1, -6, 9, 25, -8, 25, 18, 21}
}
, {{7, -37, 9, 0, -4, -20, 0, 11, -16, -24, -7, -12, 21, -11, -4, -4}
, {-8, -35, 2, 14, -21, 13, -13, 11, -7, -12, -6, 15, 10, 3, -7, -8}
, {-2, -35, 15, 2, 8, 8, 19, -3, 2, -12, 12, 6, 11, -2, -1, 16}
}
, {{-19, -6, 6, -17, 1, -9, -8, 6, -3, 20, -31, 5, 4, -3, -38, 31}
, {-14, 3, 14, 9, 15, 8, 5, 0, -24, -15, 10, 8, -8, -3, -22, -16}
, {-20, -24, -11, -14, 20, 4, 10, 5, -24, 14, -8, -3, 4, 26, -38, 13}
}
, {{16, 24, -13, 10, 9, 14, 4, -2, 18, 15, 10, -11, 10, 25, 10, 23}
, {6, -8, -17, -24, -23, 6, 1, -13, -6, -11, -28, -13, -19, 0, -6, 6}
, {-12, -16, -12, -10, 0, 20, -17, -6, -12, -17, -15, -20, -27, 6, -32, -2}
}
, {{-27, -12, 14, -18, -2, 17, -6, -6, -12, 3, -8, 0, -13, -6, -15, -7}
, {0, -16, -25, -14, -12, -14, -14, 6, -8, 2, -12, -29, -2, -3, -17, 7}
, {-32, 12, 18, -19, 8, 12, -10, 23, -15, 9, -10, 21, 15, -12, -39, -8}
}
, {{23, -18, 5, -15, -6, 4, 3, 1, -12, -29, -6, -1, -9, -21, 8, 6}
, {-11, 6, -10, -13, -2, -20, -2, -19, -1, -9, -17, 13, 2, 7, -8, 11}
, {-20, -12, 4, -6, -7, 18, -17, 10, -22, 9, 6, 23, -3, 11, -5, 22}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_1_H_
#define _BATCH_NORMALIZATION_1_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       21

typedef int8_t batch_normalization_1_output_type[21][32];

#if 0
void batch_normalization_1(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_1_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_1_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_1.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       21
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 6
#define INPUT_SCALE_FACTOR 6
#define OUTPUT_SCALE_FACTOR 6
#define NUMBER_T int8_t
#define LONG_NUMBER_T int16_t


static inline void batch_normalization_1(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_1_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = scale(NUMBER_T, (LONG_NUMBER_T)bias[z], -INPUT_SCALE_FACTOR);
      tmp += (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      tmp = scale(NUMBER_T, tmp, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[x][z] = clamp_to(NUMBER_T, tmp);
#elif defined(ACTIVATION_RELU)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
        tmp = scale(NUMBER_T, tmp, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
        output[x][z] = clamp_to(NUMBER_T, tmp);
      }
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int8_t batch_normalization_1_bias[32] = {-41, -58, -41, -14, -26, -29, -28, -45, -20, -32, -24, -35, -14, -29, -30, -8, -44, -44, -36, -32, -56, -29, -28, -43, -23, -36, -5, -21, -47, -43, -42, -22}
;
const int8_t batch_normalization_1_kernel[32] = {43, 109, 113, 95, 38, 107, 35, 52, 102, 107, 85, 67, 61, 67, 76, 61, 118, 35, 72, 127, 59, 71, 57, 127, 66, 103, 35, 60, 60, 114, 72, 70}
;
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_1_H_
#define _MAX_POOLING1D_1_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   21
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int8_t max_pooling1d_1_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_1(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_1_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_1.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   21
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 6
#define OUTPUT_SCALE_FACTOR 6
#define NUMBER_T int8_t
#define LONG_NUMBER_T int16_t


static inline void max_pooling1d_1(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef WITH_CMSIS_NN
// Not really CMSIS-NN since using arm_relu_q* is not more efficient, but use SSAT anyway
#if ACC_SCALE_FACTOR - OUTPUT_SCALE_FACTOR > 0
      output[pos_x][k] = __SSAT(max[k] >> (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#else
      output[pos_x][k] = __SSAT(max[k] << (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#endif
#else
      max[k] = scale(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, max[k]);
#endif
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_2_H_
#define _CONV1D_2_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       10
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int8_t conv1d_2_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_2(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_2_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_2.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       10
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 6
#define INPUT_SCALE_FACTOR 6
#define OUTPUT_SCALE_FACTOR 6
#define NUMBER_T int8_t
#define LONG_NUMBER_T int16_t


static inline void conv1d_2(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 

      output_acc = scale(NUMBER_T, (LONG_NUMBER_T)bias[k], -INPUT_SCALE_FACTOR);


      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }
      
#ifdef ACTIVATION_LINEAR
      output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
        output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
        output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
      }
#endif
    }
  }

#else



  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 4 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q7_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q7_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q7_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q7_basic_nonsquare(
#endif
#endif
                                      (q7_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q7_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q7_t*)bias, //bias
                                      INPUT_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q7_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q7((q7_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q7((q7_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    32
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  3


const int8_t  conv1d_2_bias[CONV_FILTERS] = {1, -12, -10, 11, 8, 12, 1, 3, -7, -7, 3, -4, 7, 9, 5, 2, 0, -3, 4, -7, 2, -4, 5, 9, 12, 9, 5, 3, 10, 10, 0, -13, -1, 6, 6, 1, -2, 6, 7, 4, -12, 12, 13, 3, -3, 7, -1, -10, 9, -1, -11, 0, 7, 9, 12, -8, 0, -9, -3, 8, -11, -1, -4, 5}
;

const int8_t  conv1d_2_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS] = {{{6, 4, -12, 7, -7, -7, 2, -14, 6, -4, -2, -14, 8, 2, -9, -7, -2, -3, -11, -4, -19, 1, -8, 1, -3, -5, -13, 5, -21, -8, -4, -6}
, {-8, -3, -6, -2, 1, -7, -1, 17, 3, -3, -1, -10, -6, 2, -10, 8, -2, -13, 7, 14, -11, 12, 4, 5, -4, -3, -8, 3, 5, -8, 6, -4}
, {-11, -1, 10, -13, -3, 2, -13, 6, 2, -6, -2, -9, 0, 6, -5, -1, -5, -4, 2, 19, -12, -9, 0, -3, 10, -1, -7, -5, 3, 0, -10, 6}
}
, {{-2, 5, -11, 0, -14, -9, -11, -11, -2, 6, -7, -28, -1, -3, -14, -24, -4, -1, 2, -8, -13, -12, -18, 11, -1, 3, -3, -22, -2, -11, 9, 3}
, {2, 8, -6, 4, 6, 11, -9, -8, 10, 3, -15, -17, 0, 10, -2, 1, -13, -3, -19, 4, -3, 1, -10, 0, -1, 5, -7, -3, -8, -12, 1, 10}
, {3, -1, -9, -8, 6, 5, -9, -1, 1, -4, -12, 7, -2, 4, -3, 17, -9, -1, 0, -3, 2, 13, 1, -4, 0, 10, -8, 1, -4, -13, 10, 14}
}
, {{-27, 1, 14, 4, 2, -13, -23, -5, 17, -19, -24, -23, -7, 3, -4, -6, -32, -4, 3, 17, -10, 4, -4, -3, 13, 7, -24, -1, 7, -20, 5, 5}
, {-11, 8, -8, -16, -5, 7, -10, -1, 16, -8, -2, 5, -15, 17, -17, -5, -8, 9, -26, -3, 1, 1, -9, -8, 8, 16, -1, 15, -6, -10, -6, 12}
, {2, 4, 2, 14, 20, -15, 1, 0, -1, -2, 8, 18, -13, -1, -8, 2, 0, -10, 9, 3, 3, 5, 0, -8, 3, 6, 10, -2, 8, -9, 6, -2}
}
, {{15, -4, -10, -8, 10, 5, 5, 4, -3, 3, -9, 11, -2, 3, -6, 18, -2, -5, -7, -6, 0, 13, -3, -10, 0, -8, 5, 13, 6, -5, -5, 26}
, {4, 8, -7, -1, 2, -11, 3, 6, -3, 6, -6, -5, 8, 7, -11, 7, -9, 4, -2, -8, -4, 3, -2, -11, -14, -5, 4, 12, 4, -10, -1, -9}
, {-31, -9, 1, -11, -26, -6, -9, -17, 6, 0, 11, -26, -10, -3, -14, -7, -8, -21, -6, 12, -6, 1, -4, -1, 10, 14, -30, -2, -9, 18, 2, -23}
}
, {{-6, -3, -7, 0, 1, 9, -2, -4, 5, -7, 8, 0, -5, -7, -11, 6, 5, -5, 12, -1, 7, -5, 18, -7, -5, -10, 5, 7, 8, 17, -4, 4}
, {9, -6, -1, -1, -8, -6, -5, 4, 0, -1, -5, 8, -4, -7, -3, -9, -8, -5, 1, 4, -8, 5, -7, 3, -1, 1, -7, 1, 7, -2, 2, -4}
, {8, -5, 17, 6, -13, 21, -12, -11, -8, 12, -14, -6, 11, 1, -2, 19, 3, -14, -8, 5, 0, 5, -8, 6, 5, 5, -10, -9, -13, 17, -8, 10}
}
, {{-6, 6, 12, -9, -13, 3, -16, -10, -31, 30, -14, -6, 4, -13, -2, -9, 2, 0, -31, -7, -1, -25, -14, 1, -18, 17, -4, -6, -23, -9, -6, -1}
, {-16, -6, -5, -17, -11, -2, -13, -2, -17, 8, -15, 4, -1, 0, 4, 7, -7, 3, 1, -5, 17, -13, 11, -3, -12, 6, -5, 3, -2, 11, -9, 16}
, {4, -5, -2, 0, 5, -1, -4, 3, -1, -2, 3, 2, -3, 6, -1, 1, -18, 1, 1, -9, -6, -11, 9, -20, 16, 12, 21, 8, 17, -2, 15, -10}
}
, {{5, -10, 7, 4, 3, -7, -3, -2, -4, -13, -7, 6, 4, -6, -12, 1, -12, 1, -4, -10, 4, -10, 5, -11, 9, -5, 3, 2, 8, -2, 3, 2}
, {-9, -9, 6, -5, 4, -6, 1, 11, 7, 4, 2, 17, -6, -6, -14, 6, -18, -2, 2, -2, 4, -8, 2, 0, -8, 3, 11, -7, 12, -7, -5, 4}
, {-29, 2, 11, -4, -7, -3, -6, -4, 21, -5, 0, -10, -4, 7, -25, -9, 5, -23, 8, 11, -19, -14, 0, 1, -5, -8, -2, -3, 11, 0, -7, -9}
}
, {{-22, 10, 0, 1, -21, 1, -9, 10, -9, -19, -2, -20, -10, -11, 3, -14, -3, -2, -5, -9, 2, -2, -11, 9, -27, 7, 2, 2, 1, -11, 4, 11}
, {-1, 7, 0, -2, 13, -2, -5, 2, -1, 5, 2, 4, -1, -3, 5, -2, 19, -6, 0, -4, -6, -17, 5, -2, -12, -2, 7, 2, 3, 10, -7, -9}
, {-15, -9, -18, 1, 10, 8, -3, -16, 10, 2, 2, 2, -7, 2, 0, 1, 4, -16, 7, -2, -14, -3, -3, 15, 14, -4, -8, -10, -2, -6, 10, -6}
}
, {{2, -5, -18, 7, 7, -8, 2, -1, 10, -8, 13, 0, 1, 19, -14, 15, -26, -6, -1, 26, -1, 33, -4, 0, 18, 13, 2, -7, 1, -13, -2, 14}
, {-3, 11, 7, 5, 1, -15, -8, 13, -8, -5, -18, 7, 0, -10, -12, -13, -7, 3, 4, 14, -15, 8, -22, 7, 2, -10, -5, 6, -8, 0, -15, 4}
, {-13, 10, -10, -10, 6, -8, -9, 15, 9, -14, -3, 10, -3, 2, -45, 11, -29, -13, -28, 14, -5, 17, -2, 3, -4, -3, -3, 6, 8, 8, 8, 9}
}
, {{-13, 16, -5, -2, 0, 1, -5, 15, -3, -3, -13, 4, -6, -3, -4, 11, -9, -11, 2, -6, -6, -3, -13, 4, -11, -8, -5, 18, 10, -8, 0, 13}
, {3, -2, 10, 0, -7, 14, -13, -6, -3, -11, -4, -4, -6, -4, -8, 6, -2, 9, -18, -9, -3, -15, 15, -1, -12, 15, -10, 9, 6, -12, 10, -6}
, {-3, -18, -17, 5, -7, 4, 3, -31, 5, -7, 15, -23, -2, 13, 18, -8, 10, 9, 0, 1, 2, -7, -2, -18, 1, 7, 9, -14, -1, -10, 14, 3}
}
, {{-12, 7, 1, 12, -25, -5, -8, 0, -11, -5, 0, -8, 5, -3, -5, -1, -4, -7, -5, -2, -6, 16, -15, -22, -14, -1, 4, -11, -11, 5, -1, -2}
, {0, -7, 3, -3, -3, 3, -11, 0, -4, 9, -15, -1, 8, -6, 4, 11, 2, 12, -5, 0, -4, 6, 15, -9, 0, 0, -3, -1, 12, 11, 9, 6}
, {-36, -5, 8, 14, 3, -12, -12, -9, 0, 8, 11, -17, -1, 10, 13, 2, 3, -4, 8, 2, -1, -3, -2, -3, 8, 7, 5, 2, -4, 3, -15, -9}
}
, {{0, 5, 1, -9, 3, 16, -4, -3, 2, -11, -13, -3, -3, -11, -15, -4, -3, 4, -4, -12, 4, -8, 9, 10, 9, 5, -5, -4, 15, -9, 6, 6}
, {-11, -1, 8, -1, -11, 21, -6, -28, -2, -10, -10, -21, -5, -3, -3, 4, -1, 12, 7, -6, 9, -3, 12, 1, 13, -1, -7, -16, 4, 8, -1, 1}
, {-19, -7, -9, 4, -13, -4, 2, -22, -6, -10, -7, -4, -1, 2, 1, 0, -9, -2, -5, 1, -3, -2, 13, -3, -10, -2, -8, -18, 5, 0, 7, 9}
}
, {{-8, 2, -1, 2, -4, 9, -19, -1, 11, 0, -10, 5, 8, 5, -5, 13, -13, 0, 6, 13, -1, -7, 10, -6, 7, 2, -3, -2, 20, 12, 15, 8}
, {-7, 6, -14, -6, -8, 0, -13, -12, 3, -8, -15, 7, -10, -6, 3, -4, -22, 4, -1, -12, 1, -9, -13, 1, 2, 3, 3, -2, -3, -9, -4, -9}
, {3, -3, -9, 11, 6, 0, -16, 10, -3, -4, -5, -2, 5, 14, 3, 15, -21, 1, -3, 4, 0, 3, -5, -17, -10, 2, 4, -7, 1, 6, 0, 9}
}
, {{2, -13, 4, -11, 8, -4, -12, 3, -11, -11, -4, 14, 1, -11, -6, 2, 7, 9, 7, -15, 6, -21, 18, 13, -4, -14, 9, 11, 7, -11, 4, 10}
, {4, -6, 4, -11, -14, 6, 9, 3, -12, 3, -1, 8, 4, 0, 4, -3, 6, 9, -10, -20, 6, -12, 7, -15, -18, -5, -13, 4, 10, -1, 3, -4}
, {-11, -6, 5, 0, -12, -4, -8, -15, -13, -10, -13, -1, 3, 7, 1, 0, -10, -15, 4, -4, 0, -11, -2, 3, -4, 0, -20, -10, 1, 1, 2, -17}
}
, {{-18, 15, -7, -19, 4, 13, -6, -1, 15, -12, -11, -12, 5, 1, -9, 16, -2, -10, -29, 17, -4, -11, -13, -9, -5, 6, 10, -3, -5, -19, -8, 16}
, {-9, 0, 7, 6, 10, -2, -9, -1, -1, 5, -11, -3, -7, 4, 9, -13, 6, -21, 3, 0, 1, -13, -7, 3, 1, 6, -3, -14, -10, 0, 0, 4}
, {4, 2, 1, 4, 7, 9, -15, -21, 1, 8, -3, -7, -4, 14, 6, 7, 3, -12, -7, 11, 4, -8, -7, -5, 4, 10, -1, -19, -17, -4, -2, -8}
}
, {{-19, 0, -5, -4, -11, -7, -11, -12, -10, -21, 10, -6, -15, -9, 4, -27, -12, 13, 4, -11, 6, -14, 1, -3, -9, 0, -3, 3, 9, -5, 0, 1}
, {-17, -6, 2, 6, -1, -7, -9, -1, 0, -15, 6, 4, 3, -3, 3, -11, -7, 0, -1, 0, -5, -27, -3, 5, -7, -3, -1, -4, 26, 3, -13, -2}
, {0, -1, -2, -4, 10, 1, 8, 7, 2, -7, 4, 13, -5, -1, -1, -1, -11, -14, 13, -5, -9, -4, 13, -11, -2, -11, 11, 6, 11, -10, 18, 8}
}
, {{-6, 8, -18, -15, -15, -5, -6, -8, -10, -8, 5, -1, 4, 0, 0, -2, 0, -17, -6, -3, -3, -17, 2, 9, -19, 5, -11, -4, 2, 9, -1, -16}
, {-3, 7, -7, 0, -15, 9, 3, -6, -12, 4, -9, -5, 5, -4, -5, -9, 13, 7, -3, -10, 3, -15, 16, -1, -8, -18, -3, 6, -7, -22, -6, -9}
, {13, -1, 11, 10, -4, 17, 4, -11, -4, 3, -16, 7, -6, 2, 12, 3, -8, 11, -1, -5, 12, -1, -3, -19, -10, 2, -3, -5, 8, 10, -5, 0}
}
, {{-12, 5, 9, -8, -26, 2, -12, -4, -21, -13, -18, -23, -6, -8, -13, -20, 3, 4, -11, -6, 5, -9, 8, 4, -15, -4, -9, -4, -11, 10, -13, 23}
, {-2, 10, -3, -14, -10, 8, -7, -7, -6, 5, -13, 4, -9, 19, -4, -8, -12, -9, -8, 1, -3, -14, 4, 0, 8, -4, -8, -8, 12, -4, 8, -1}
, {-8, -14, 13, -20, -1, 1, -3, 9, -21, 3, -20, 8, -7, -14, -1, 12, 14, 0, -6, -26, 6, -6, 7, 7, -27, -6, 5, 3, -8, 12, -12, 12}
}
, {{-13, 1, -5, 2, -7, -3, 2, 4, -9, -4, 7, -3, 13, -17, -13, -18, 10, -10, 15, -7, -11, -21, 17, 3, -2, -1, -6, 5, -4, 10, 4, -4}
, {-1, -1, 1, 3, -1, 4, 13, -12, -1, 3, 15, 1, -3, 0, -9, 0, 5, -8, -1, 3, 7, -3, -9, 11, -12, -18, 7, -4, -10, 3, 0, -6}
, {-21, -1, -27, -17, 4, 3, 1, 7, 1, -15, 3, 1, 11, 21, -8, 4, -29, 3, -14, 0, -13, 19, 7, 10, 2, -6, 2, 7, 19, -4, 12, 7}
}
, {{-5, -10, 1, -14, -9, -7, -5, 7, -19, 4, -9, -8, -22, 14, -10, -14, -6, -1, 7, -5, -6, -11, 8, 9, -11, 2, -9, -8, -7, 22, -7, 0}
, {5, 8, -1, -9, -14, -5, -2, 23, 15, -2, -4, -4, -8, 1, 10, -1, -3, 1, 2, 19, 11, 7, -4, -13, 1, -2, -5, 6, 2, -3, 1, 2}
, {-9, 4, 6, -20, -13, -4, -5, 9, -4, 5, 3, 3, -1, -24, -10, -14, 1, -12, -9, -16, -6, 2, 3, 10, -3, -3, -5, 4, -1, 0, 4, 6}
}
, {{-12, -17, -9, 12, -7, -5, -8, -9, 4, 2, 6, -3, -6, -9, 7, -5, -6, -10, -6, 1, 6, -4, 2, -11, 0, 7, 0, -6, 5, -11, -5, 0}
, {-20, -1, -8, -10, 4, -5, 8, -2, 9, -21, -1, 10, 0, 3, 4, 2, -19, 1, 5, 4, -2, 8, 1, -2, 14, -1, 8, -6, 2, -11, 3, -2}
, {0, -5, -9, -3, 2, 7, -17, 3, 8, 0, -16, -1, 0, -1, -3, 4, -27, 7, 4, 7, -3, 9, 0, -11, 1, 6, -1, 9, 9, -22, 19, 2}
}
, {{-12, -7, 2, 11, -7, -9, -13, -18, -3, 12, -8, 0, 16, 8, 17, -17, -2, -16, 16, -1, 11, 2, -10, -2, -2, 15, 3, -26, -12, 18, -4, 1}
, {-1, -13, -4, 4, -10, 1, -3, 8, 2, 1, -5, -3, -9, 3, -3, -7, -14, 18, -3, 5, -5, -10, 7, -5, 4, 8, -7, -11, 16, -1, 2, 6}
, {12, -8, 6, -18, -6, 16, -6, 20, 8, 1, -7, 9, 3, -6, -11, 5, -9, -8, -4, 1, -1, -3, 12, 7, 3, -3, 7, 10, 11, -19, -1, 10}
}
, {{3, -6, -4, -31, 8, 10, 0, 13, -2, -10, 1, 13, -16, -21, -22, -3, -3, 9, 6, -7, -4, -7, 15, 0, 4, -33, 10, 11, 16, -6, 8, 11}
, {-6, -7, 5, -17, 7, 7, 1, 4, -10, 7, -4, 2, -7, -14, -14, 2, -4, -6, -3, -9, -13, -10, 11, 6, 0, -6, -4, 1, -8, -1, -4, 0}
, {-21, 8, 15, -13, -20, 3, -7, 1, -1, 12, -9, -21, 8, -3, -21, -7, 4, -29, -10, 8, -20, -11, -6, 6, 22, 4, -46, -28, -16, 16, -12, -12}
}
, {{-5, 3, 12, 3, -26, -1, -6, -2, -1, -16, 8, -9, 3, -8, 3, -2, 12, -6, 14, -6, 0, -8, 23, -1, 3, -1, 6, -13, -4, 22, -2, -3}
, {-7, 7, -19, 0, -8, -9, -3, 0, 3, -1, 4, -3, -13, -7, 7, 1, -10, -14, 0, -4, -1, -3, -3, -5, -1, 1, 2, 1, 9, -5, 3, 6}
, {-16, -7, -20, 4, 6, 1, 8, -11, 8, 0, 1, 6, -15, -9, 0, -10, -8, -8, 7, 8, -16, 5, 1, 4, 7, -6, -8, 6, -7, 3, 14, 0}
}
, {{6, -7, -1, -9, 9, 8, 3, 14, 5, -9, 8, 3, -1, 1, -4, 8, -3, 11, 6, 19, -9, -1, 2, -10, 18, 1, 8, 9, 19, -7, 2, 8}
, {-7, -5, 0, -10, -8, -2, 0, -4, 5, 2, -4, -19, 3, 0, 7, -7, 2, -23, -3, 3, -4, -8, -10, -7, 7, -3, -11, -1, 3, -11, 1, 6}
, {2, -1, 1, 0, 5, -7, 3, 10, -19, -2, 2, 11, 10, 9, 2, 4, 4, -1, 4, -4, -7, -2, -9, -14, -11, -5, 9, 5, -5, -10, -6, 4}
}
, {{-2, 9, -26, -20, -4, 11, 5, -10, -1, -2, -6, 15, -17, -1, -17, -7, 0, -4, -4, -19, -11, -13, -10, 5, -11, -4, -8, 13, -8, 7, 15, 10}
, {-4, 10, 6, 12, -8, -27, -11, 0, -9, 3, -15, 4, -18, 5, -12, 24, -18, -19, 3, 0, 3, 2, -4, -1, -10, 11, -5, -2, 5, -22, 11, -2}
, {-11, -3, -32, 4, -7, 16, -20, -4, 31, -8, -7, -4, -14, 34, -15, 7, -17, -17, -23, 14, 1, 14, -9, -20, 4, 18, -6, 5, -5, -22, 2, 11}
}
, {{-4, 0, 8, 6, 1, 0, -3, -18, 14, -5, -1, 8, -4, 10, 2, 3, -17, 2, -3, -5, 6, -2, -16, -22, 10, 9, -1, -9, -9, 3, -1, -1}
, {-3, 9, 4, 2, -2, -20, -2, 4, 5, -17, -8, 12, 3, -9, 11, -9, -11, -1, 11, 0, -2, 3, -7, -13, -5, 3, 4, -6, 9, -1, 5, -10}
, {-1, -13, -8, 4, -1, 16, -10, 1, 10, 0, -11, 11, -11, 4, -16, 9, -8, -13, -2, 2, -3, 2, -5, -15, 7, 11, 1, -5, 1, 4, -1, 13}
}
, {{-8, 0, -2, -1, 7, 2, -8, 5, -3, -1, -5, 22, -16, 2, 3, -8, -2, 5, 15, -16, 15, -6, 13, -8, -7, -11, 13, -13, 15, 7, -3, 5}
, {3, -8, 0, -7, -6, -9, 12, 13, -1, 2, -1, 3, -1, -17, -6, -5, -4, -12, -1, 1, -12, 5, -2, -1, -6, -1, 3, 8, -1, -11, -4, -10}
, {-12, 13, 5, -13, -16, 13, -2, -12, 0, 6, 7, -12, -1, 6, -4, -6, -1, -11, -13, -5, -7, 11, 1, 3, 11, 8, -20, 1, -12, -2, -12, -2}
}
, {{2, 9, -11, 5, -5, -12, -9, -20, 7, 3, -2, -16, 8, -15, 1, -17, -8, -19, -7, -11, -18, 4, -13, 7, -6, 14, 6, -7, 6, -11, 6, 11}
, {5, 10, -2, -7, 2, 6, 1, -4, 1, 3, -3, -3, -1, -13, 8, 7, 5, 1, -2, -4, -7, 10, -1, -6, -5, 0, -3, 6, 8, -16, 9, 2}
, {12, -14, 0, -11, 2, 15, -4, 19, 2, 7, 2, 12, 7, 6, 6, 14, -7, 8, 12, 10, 5, -14, 7, -11, -11, -18, 9, 7, 12, -3, -3, -7}
}
, {{-2, -9, -10, 2, 7, 5, -12, -3, 6, 13, -22, 13, -10, 2, 14, 6, -10, -11, -5, -6, 15, -11, -8, -16, 13, -3, 5, -4, -5, -13, -4, -5}
, {4, 8, -10, 12, 1, 6, -5, 9, 5, 6, -12, 7, -2, 9, 1, 4, -15, -13, 8, 0, 2, -12, -11, -2, -5, -10, -2, -1, -5, -13, 3, 0}
, {5, 11, -9, -11, 2, -4, -12, 16, 4, 23, -13, 2, 13, 3, -3, 0, -15, -3, -13, -2, 2, 1, 4, -14, -5, -5, 2, 5, 15, -11, -11, -2}
}
, {{-20, 0, -9, -15, -22, -9, 4, -9, 0, -12, 2, -12, -11, -11, -8, -25, -3, -3, -8, -7, -10, -16, 5, -4, -6, 0, -9, 9, 4, 5, 7, -7}
, {-11, 10, -2, -12, -5, -4, -8, -4, 2, -1, 8, -5, -4, -3, -8, -8, -8, -10, -1, -3, -16, -3, 12, -6, -9, 1, -8, -6, 14, -8, 7, 11}
, {-6, 4, 0, -18, -16, -2, 12, -1, -6, -1, 6, -13, -8, 0, -3, -13, -19, -4, -1, 8, -7, 11, 8, 9, 7, -7, -12, -4, 9, 12, 9, 4}
}
, {{13, 8, 8, -15, -29, 7, -12, -10, -1, 2, -9, -2, 3, -2, 9, -11, 7, 6, -11, -15, -10, -18, 2, 14, -21, -25, -14, -20, -6, -16, -3, -11}
, {7, 7, -3, -2, 0, 10, -9, 6, -15, 6, -16, -2, 18, -1, 8, 6, -4, -9, -10, 2, 2, 3, -5, -19, -16, -4, -16, -6, 4, -5, 2, 10}
, {-2, -29, -6, -20, 0, -3, -12, 3, -11, 13, -16, 2, 2, 2, -3, 7, -4, 6, -12, 8, 9, -4, -6, -8, -18, -1, -17, -6, -2, 1, -5, -14}
}
, {{5, -4, -11, -14, 0, 5, -8, 13, 1, -13, 1, 7, 0, 0, -11, 4, -9, -1, -9, 7, -6, -14, 23, -9, 8, -12, 3, 6, 7, -23, 9, 17}
, {-2, 12, -7, 7, -1, 7, 1, 12, 14, 1, -3, 3, -5, 0, -3, 1, -14, 1, -7, 5, -8, -2, -3, -2, 6, -1, 2, 3, -4, -17, 0, 21}
, {-50, 16, 2, 12, -3, -18, -7, -17, 9, -16, 9, -19, -33, 1, -11, 2, 19, -17, 3, 6, -19, -6, -13, 8, -1, 16, -12, 0, -7, 8, -13, -11}
}
, {{-10, 8, -3, 10, 17, -9, -13, -4, 16, 0, -1, -12, -6, 8, 1, 3, 3, -10, 0, 9, -8, 9, -12, -9, 12, 13, 0, 8, 0, -8, -3, 8}
, {-13, -2, -4, 9, -2, -7, -4, -13, -1, 10, 3, -12, 16, -2, -2, -8, 1, 1, -1, -8, -13, 1, -5, 11, 6, 2, -1, 1, -12, 13, 9, -4}
, {5, -2, -5, -3, -6, 22, 1, 18, 2, -2, -2, 1, 4, -19, -4, 3, 3, -1, 14, -13, 4, 2, -8, -4, -1, -6, 10, 3, 16, -3, 9, 7}
}
, {{-13, 17, 6, 6, -21, 6, -25, -11, -3, 20, -23, -8, -1, -4, 10, -23, -6, -21, -9, 1, -1, -9, -18, -1, 0, 7, 8, -18, -25, 13, -12, -2}
, {-3, -18, -9, -4, 9, 8, -3, -5, -3, 12, 4, -11, 0, -5, 0, 5, -2, -8, 9, -2, 10, -13, -3, -12, 4, -13, 7, -7, 3, 2, -1, 7}
, {-10, -12, -25, -2, 9, 2, 4, 19, 10, 0, 1, 10, -4, -5, -9, 3, 3, 21, -4, 0, 6, 3, 13, -8, 4, 2, 16, 12, 17, -26, -7, 27}
}
, {{-21, 17, 19, 1, -25, -11, -16, -25, 0, -13, -10, -10, 16, -5, 4, -5, -6, -15, -5, -3, -1, 3, -20, 6, -25, 5, -3, -17, -10, 11, -10, -10}
, {10, -3, 10, -22, 5, 1, -10, 0, -21, 13, -9, 6, -4, -7, -7, 9, 4, 1, 11, -10, 10, -10, 4, -1, -1, -16, 6, 6, -9, 14, -3, 6}
, {9, -7, -4, 0, 4, -12, -5, 22, -9, 4, -4, 16, 2, -12, -10, 0, 10, 8, 10, -20, -3, -9, 3, -10, -11, -17, 25, 0, 10, -7, -7, -6}
}
, {{-5, -3, -12, -5, 0, 3, -3, -1, 8, -4, 9, 0, 2, 10, -8, -1, -17, -1, -8, 11, 2, 3, -4, -3, 14, 10, -11, 8, 12, 3, 8, 1}
, {-7, -11, 0, 3, -9, 7, 1, -6, 10, -9, -9, -9, 9, 1, 2, -3, -3, 3, -7, 10, -10, 1, -8, -13, 13, 4, -5, -5, -2, -5, -1, 4}
, {-5, -5, 22, -6, -17, -9, -21, 22, -13, 5, -9, 1, -9, -2, -21, 3, -16, -14, 1, -5, -3, -3, -1, 10, -29, 12, -8, -18, 4, 26, -15, -10}
}
, {{-2, 13, 12, -3, 1, -11, -12, -5, 5, -7, 1, 0, -6, -2, -9, 18, 6, -8, 6, -12, 0, -10, 7, -4, -14, -12, -9, 6, -1, -2, -10, -6}
, {-11, 10, 17, -17, -22, 0, -21, -6, 9, -18, -2, 2, -14, -2, 2, -10, -9, 0, -3, -2, -14, 1, 0, 17, -2, -6, 1, 7, 3, 7, 2, -1}
, {2, 2, -5, 7, 4, 21, 4, 5, 5, -26, 9, 14, -6, 0, 6, -13, 17, 1, 10, -11, -9, -20, 16, 0, 6, 0, 21, -4, 18, -2, 32, 4}
}
, {{-4, -10, 17, -19, -3, 16, -8, 4, -9, 12, -5, -5, -5, -10, -16, 4, 21, 2, -12, -16, -11, -30, 17, -16, -19, -20, -4, 2, 3, 5, -7, -9}
, {6, -5, 7, -4, 6, -3, 4, -12, -16, 7, -3, 0, 5, 2, -1, -2, 14, -14, 1, -20, 0, -7, 13, 0, -4, -1, -2, -7, -7, 1, -18, -15}
, {3, -5, -10, 14, -6, -10, 5, -15, 7, 4, 10, -19, 10, 4, -2, 3, -5, -27, 4, -3, -3, 1, -3, -5, 5, 4, -27, -19, -23, 12, -10, -23}
}
, {{-14, 17, -9, 10, -15, -12, -22, -11, 11, -10, -4, -13, 12, 5, -7, -10, -20, -22, -2, 14, -5, 16, -28, -8, -1, 22, -14, -13, -8, -1, -7, 2}
, {0, -2, 1, -7, -8, -2, -15, 9, 4, -1, -6, -8, 9, 7, -16, 4, 1, -12, -9, -2, 0, 8, -5, -5, -12, -5, -3, 3, 4, 1, -10, 11}
, {17, -7, 3, -14, 5, 0, 7, 8, -4, -1, -2, 22, -5, -10, -20, 16, -1, -9, 5, -2, 0, -5, 15, -3, -14, -13, 14, 9, 14, -9, -7, 10}
}
, {{9, -15, 12, -15, 10, -9, -13, 8, -29, 13, -37, 17, 0, 1, 4, 18, 0, 0, 1, 0, 16, -5, 19, -16, -1, -19, -14, 9, -6, 11, -37, 19}
, {-14, -14, 4, -16, 5, -9, 4, 11, -35, 11, -19, 11, -8, -13, 4, 0, -1, 1, -5, -3, 1, -11, 3, 3, 12, 10, -13, -1, -7, 6, -6, -10}
, {-4, -4, 19, -11, -10, -3, 0, 11, -21, 12, -18, -4, -3, -21, -7, -3, 6, -20, -13, -12, -4, -15, -4, 6, -18, -2, -5, -2, -12, 14, -14, -6}
}
, {{-10, 11, 3, 0, -18, -6, 6, -5, 2, -21, -1, -3, 7, -8, 11, 2, 2, -4, 8, -5, 14, 17, -17, 3, -10, 10, -4, 12, 7, -11, 15, 9}
, {-12, -12, 1, 0, -16, -3, 6, 6, -4, -3, 8, 0, -2, 5, 7, 0, 1, 7, 5, -1, -8, 1, 16, -2, -10, 0, 10, 2, 4, 2, -2, 1}
, {-14, -16, -2, -13, -18, -7, 17, -1, -3, -14, 16, -3, -4, -18, 0, -4, 5, -3, 17, -1, -7, 2, 14, 11, -6, 7, -20, 12, 3, 8, 3, -15}
}
, {{6, -1, -3, 4, 11, -2, 4, 15, 0, 14, 6, 1, 11, -9, 4, 13, -4, -3, -5, -1, -9, 10, 11, -12, 1, -4, -7, 6, 14, -9, -7, -15}
, {4, 1, 12, -6, -6, 3, 5, 10, 3, -3, -6, 1, 3, -5, -3, -6, -11, -3, 5, 1, -5, 10, -9, 1, -4, 3, 1, 5, -4, 12, -3, 2}
, {-19, 9, 7, 14, 8, -5, 0, -13, 5, -11, -2, -12, 5, 7, 6, 2, -12, -15, 14, 6, 20, 3, -10, 13, 12, 13, -5, -16, -2, 4, -4, -10}
}
, {{9, -11, -1, 10, 14, 12, 7, -6, 7, 7, -1, 6, -6, 1, -4, -1, -4, 7, 10, -11, 3, -15, 7, 4, -4, 0, 6, -5, 17, -9, 0, 5}
, {-4, 7, 3, 7, 2, -12, 3, 12, -3, -6, 6, 4, -1, -5, -1, -2, 4, 9, -1, -2, -3, -1, -1, -6, 3, -6, 7, 4, 4, 10, 1, -7}
, {10, 0, 1, 13, -2, -2, 5, 5, -2, -6, 14, 2, 3, -7, -8, 3, 2, 6, 1, 1, -10, 2, -5, -6, -3, -4, 10, 19, -5, 8, -14, -10}
}
, {{5, -7, 5, 3, 7, -2, -8, -4, -11, 21, -14, 1, -11, -1, -3, -10, 7, -7, 0, -14, -1, -20, -4, 2, -25, -4, -2, -8, -3, 6, -25, -11}
, {3, -5, 3, 1, 5, -4, -12, -5, -9, 11, -16, 3, 6, 5, 1, 9, -4, -11, 10, -4, 10, -3, 9, -1, -12, 3, -1, -20, 3, 4, -19, -11}
, {-16, 4, -9, -18, 0, 0, -17, -13, -8, 3, -5, -12, -2, 6, 6, 6, -13, -14, 6, -9, -6, 1, 12, 1, 17, -3, -36, -11, -7, 0, 1, -17}
}
, {{-4, -23, 3, -5, -1, 8, -13, -10, 13, -11, 0, 18, -12, -5, -1, 7, -5, 10, 2, -29, 25, -7, -5, -4, 3, 3, 3, 9, 17, -28, 13, 8}
, {-16, 6, -7, -3, -6, -8, -9, 6, -1, -20, 0, 11, -2, -2, 6, -8, -14, -1, 3, 13, -1, -3, 7, -1, -4, -5, -5, -8, -10, 2, 0, -19}
, {-12, 18, -13, 0, -4, 6, -12, -2, 19, -7, 12, -5, -27, 13, -10, -2, -17, -7, -10, 18, 5, -3, -6, 6, 17, 7, 7, -2, -7, 8, -1, 11}
}
, {{-4, -21, 21, -7, -9, 7, 16, -7, -5, 14, 0, 0, 15, -8, 18, -6, 17, 3, 7, 6, 22, -1, 13, -15, -10, -1, 0, -6, -12, 16, -6, -1}
, {-3, -4, 14, -6, -8, -10, 7, 3, -6, -4, 11, 4, -2, 0, 17, 7, 5, 3, 5, 18, 4, 8, 7, 2, -8, -7, -2, 3, -8, 3, -8, -4}
, {-5, -10, 7, -7, 0, -8, 0, 14, -9, 2, -7, 10, -8, 12, 3, -8, -14, 6, -9, 7, -5, -7, 7, -6, 0, -2, 6, 8, 10, 1, 4, 3}
}
, {{4, -22, -11, 20, 25, -15, 2, 7, 15, 6, -3, 9, -2, -9, 0, 12, -2, 0, 15, 5, 4, 18, 9, -20, 15, -10, 4, 3, 5, -22, -10, 7}
, {-21, -4, -2, -6, -15, 9, -2, -13, 7, -9, -5, -13, -8, -8, -16, -24, -20, -2, 0, 3, -18, -1, -11, 6, 8, -2, -23, -9, 5, -14, -7, -10}
, {-9, 3, -2, -8, -23, 3, -12, -4, -6, -8, -8, -7, -6, 2, -6, -15, -5, -1, -7, 9, 1, -3, -15, -3, -5, 11, -9, -15, 5, 0, 1, -3}
}
, {{-6, -11, -4, 4, -12, -3, 2, -22, 2, 0, 2, -15, 14, -18, 6, 0, 2, -21, -4, -5, 12, 18, -14, -10, -6, 14, -12, -13, -15, -8, 2, 3}
, {-1, 0, 7, 3, -4, -5, -1, 1, 0, 3, 8, 15, 7, -5, -6, -5, -1, 1, -6, -2, 0, -5, -14, 6, -15, -5, -3, -3, -6, 3, -12, 6}
, {35, -8, -11, -4, -1, -4, 0, 3, 10, 6, -17, 11, 13, 3, -12, 4, -13, 2, -7, 2, -6, -1, 7, -12, -3, -6, 10, 12, 18, -9, 10, 10}
}
, {{6, 2, -1, 4, 1, -1, -4, 7, 6, -2, -1, 4, 7, 5, 2, -6, -8, -6, -4, 9, -14, 7, 12, 13, -6, -2, 3, -1, -1, 3, 8, 7}
, {-7, 3, -11, -12, -4, -3, -5, 2, 7, -18, 9, -6, -23, -13, 0, -1, 4, -2, -3, -5, -15, -2, 9, 8, 7, -8, -12, -2, -3, 9, -5, -2}
, {1, 13, 0, -8, -17, -17, 3, -25, 8, -15, 5, -6, -15, -21, -21, -22, 9, -1, -3, -10, -11, 2, 1, 8, -5, -3, -1, -1, -5, 2, -1, -15}
}
, {{12, 6, -9, -13, -1, 9, -16, 8, 17, -7, 1, 7, 18, 15, -8, -3, -2, -1, -18, 24, -7, -17, -4, -4, -8, 13, -11, -5, -12, -5, 0, 7}
, {11, 1, 16, -4, -11, 13, -21, -7, -13, -8, -17, 5, -7, -17, -7, 5, -14, 16, 6, -20, 10, -15, 0, 1, -3, -6, -6, -10, 13, 1, 9, -4}
, {-6, -4, 2, -9, -9, 6, -5, -13, 4, -26, 0, 11, -1, 17, -2, 7, -25, -8, -2, -12, 10, 15, 11, 2, 5, 8, -11, -14, 9, 3, 9, 4}
}
, {{1, 11, -7, -8, -8, -7, -14, -6, 9, 1, -6, 0, 13, 22, -8, -11, -4, -3, -12, 2, -12, 0, -23, -1, -2, 8, -8, 0, -15, -17, -1, 4}
, {7, 8, -3, 1, 5, -10, 3, -5, 9, -1, 6, 7, 2, -2, -3, 9, -8, -7, -9, 14, 3, 2, 0, -16, 6, -11, -4, 5, 1, -16, 13, 1}
, {15, -11, -4, 9, -3, 5, -9, 1, -4, 5, -8, -1, 16, 13, 10, 19, -18, 1, -2, -4, 3, 3, -7, -2, -1, 5, -12, -8, -6, 3, -2, -2}
}
, {{-28, 11, -16, 1, -2, 19, -10, -16, 0, -10, -6, -23, -11, 9, -3, -3, 4, -6, 7, -3, 7, -1, -8, -3, -6, 26, -1, -17, 2, -3, 7, 1}
, {-6, -7, -3, 6, 10, -6, 1, 10, -9, -1, -11, 8, 6, -1, -9, -14, 3, 4, 6, -11, 1, -9, 1, -2, -10, 2, 0, -14, 9, -2, 4, 4}
, {18, -3, 9, 0, -1, 6, -9, 9, 0, 3, -10, 1, 2, -14, -10, 6, 13, -4, -2, -3, -5, 0, 1, -2, -7, -7, 2, 7, -1, -10, -15, 7}
}
, {{-17, -3, -10, -2, 0, 10, -8, -1, -9, 7, -9, -2, 0, -1, 0, -7, 1, -3, 0, -2, 13, -11, 1, 3, -15, 4, 1, -2, -4, -6, 8, 6}
, {3, -5, -2, 9, -1, -4, -3, 7, -2, 4, 7, 3, -19, 10, 0, -3, 7, 5, 4, 1, 8, -11, 3, -1, 5, 11, -1, 7, 9, -16, -11, 5}
, {-27, 1, 11, 7, -16, -3, -16, 0, -11, 1, -3, 1, 2, -13, 5, 4, 31, 6, -2, -6, 10, -14, 11, -10, -22, -13, -1, 0, -4, 19, -20, -5}
}
, {{5, -9, -14, -4, 5, -10, 0, 5, -18, 3, -19, 14, -8, 14, -6, 0, 2, -18, -5, 7, -1, -1, 15, -5, -3, -9, 6, 3, -2, -10, -15, 8}
, {6, -8, -9, -1, 8, -1, -3, -9, -2, -4, -5, -12, -5, 19, 13, 15, -2, -9, 9, 10, 2, 4, 0, 6, 10, 11, 8, -2, -3, -20, 6, 8}
, {-9, -14, -15, 11, -2, -1, 6, -2, 10, 18, -17, 5, -6, 11, 2, -3, 1, 0, 5, -2, 5, -7, -4, -10, -1, 12, 16, 6, 10, -1, 12, 11}
}
, {{-11, 4, 4, -8, -5, -1, -10, -7, -7, 5, -4, -3, -4, -17, 12, 3, 13, -5, -10, 4, 6, -8, -16, 3, 0, 2, -14, -11, -17, 14, -1, 2}
, {-3, 0, -6, -18, -7, -3, -8, -15, -7, 4, 11, 7, 16, -7, -10, -1, 12, 6, 4, -7, 11, -11, 6, 8, -14, -2, -20, -3, 1, 2, 2, -27}
, {32, -17, 10, -8, -1, 14, 23, -2, -23, 13, -17, 3, 18, 4, 7, 7, 5, 8, -24, -15, 8, -1, -5, -28, -24, -17, -5, 11, -12, -3, -3, 8}
}
, {{-3, 1, 0, -9, 2, -2, -13, -14, -4, 11, 2, 2, -4, -10, -2, -1, 9, 7, 12, -9, 0, -20, 5, 0, 5, -14, 13, -12, 0, 11, 9, -9}
, {-8, -12, 11, -6, 7, 1, -7, 2, -9, 9, -6, -4, 0, -5, 0, -4, -9, -7, -8, -12, 7, -7, 7, 4, -10, 0, -10, 10, 8, 5, 8, -2}
, {-1, 8, 7, -5, -9, 5, -7, -2, -14, -2, 3, 2, -9, -6, 9, 0, 26, 9, 11, 2, 16, 5, 3, -1, -16, 1, 13, -5, -6, -18, -18, 0}
}
, {{5, 2, 2, 0, 13, 6, -6, -5, 7, -16, -10, 11, -13, 1, -11, 0, -3, 5, -3, 12, 4, -4, -7, -5, 2, 7, -9, -2, 9, -9, -7, 6}
, {4, -8, -25, 13, -1, 3, -18, 1, 7, -12, -7, 6, 2, 14, -18, 24, -31, -12, -14, 13, 0, 10, 8, -10, 6, 5, -6, -3, -5, -26, -9, 3}
, {-9, 11, -32, 6, -3, -3, -12, 11, 14, -3, -12, 7, -15, 20, -2, -18, -16, -14, -1, 11, 4, -2, -20, -21, 0, 8, 1, -10, 0, -25, -6, 8}
}
, {{-27, 10, 0, 4, -6, -15, -18, -17, 3, -7, 1, 3, 2, 4, -7, -13, 4, -15, 11, -6, -12, -12, 4, 3, -5, 9, -15, -18, -7, 25, 3, -10}
, {1, 13, -13, 8, -3, -2, -13, -4, -1, -16, -11, -2, 0, 3, -8, -18, -8, -9, -7, 4, -6, -12, -14, 6, 7, 6, 6, -8, 6, -4, 4, 11}
, {16, -7, -13, 2, 8, -2, -8, 7, 2, -8, -6, 4, 14, 16, 0, 7, -11, 6, -16, 3, -2, -2, 3, -18, -3, -8, 12, 16, 10, -28, -12, 17}
}
, {{12, -18, 8, -10, 16, 11, -2, 6, 4, -1, -7, 30, -2, 4, 5, 17, -12, 12, 9, 4, 5, -1, 21, -33, -2, -16, -6, -1, 19, -15, -6, -2}
, {-5, 0, -6, -2, 2, -3, -3, -1, 7, -3, 8, -10, 2, 6, 11, 7, 0, -8, -1, 2, 1, -5, 10, -6, 5, 1, 2, 2, -2, -6, -1, 2}
, {-28, -6, 0, -6, -18, -18, -9, -24, 0, 4, 4, -19, -13, 3, 11, -10, 3, -19, 8, -5, 2, -12, -19, 18, 11, 14, -19, -21, -18, 11, -9, 3}
}
, {{-19, 13, 6, -4, -13, -16, -8, -10, -3, 2, -2, -6, -14, 2, 8, -5, -1, -15, -3, 1, 4, -10, -10, -5, -12, 2, -18, -3, -14, 8, -9, -5}
, {-12, -4, 1, -7, -15, -17, -2, 11, -6, 5, 6, 3, -19, -5, 0, 1, 2, -11, -4, -1, 0, -10, 5, -7, -19, 0, -11, 13, -7, 8, -5, -1}
, {27, 4, -9, -12, -10, -5, 8, 7, -10, -5, 4, 2, 16, -7, -2, -2, 22, 8, -8, -17, -16, -10, 13, -6, -17, -12, 12, 16, 7, -6, 17, -15}
}
, {{-4, 6, 0, -6, -26, -21, 5, 13, -9, 1, 9, -24, -3, 3, -7, -6, 7, -4, -22, 9, -37, -13, 4, 8, -9, 11, -8, 28, -12, -5, -8, -1}
, {10, 10, 2, 7, -4, -15, 5, 2, -1, -3, 1, -11, 10, 1, -1, 7, 5, 14, -11, 0, -24, 5, 2, 14, -2, -14, 0, 21, -12, 4, -5, -1}
, {0, 8, 0, 13, 3, -13, 8, 0, -1, 4, 0, -12, 1, 3, 3, 1, -4, 0, -12, 11, -7, 6, -9, -3, 3, -5, -6, 1, -12, 3, -8, -1}
}
, {{-4, -22, -3, 0, 16, 15, 2, -4, -5, 5, -3, 4, 5, 5, 20, 5, -8, -8, 7, 3, 13, -5, 3, 6, 8, 15, 0, -14, 13, 0, 18, 6}
, {-9, 2, -2, 2, 3, -3, -4, 4, 4, -2, -4, 8, -14, 3, 5, -7, 2, -12, 6, 8, -3, -2, -2, 11, -4, -2, -3, -4, 2, -7, -3, 2}
, {-15, 13, 8, -6, -11, -23, 3, 9, -4, 9, 4, 8, 1, -24, -15, -22, 13, -4, 8, 8, -20, -7, -19, 1, -6, -9, -2, 13, -7, 0, -23, -5}
}
, {{2, -3, -11, -10, 12, 18, -5, 4, 4, -1, 0, 9, -2, 15, 3, 15, 0, 7, -2, -2, 3, -10, 9, -16, 0, -20, 7, 1, 7, -26, -7, 10}
, {-2, -8, -1, -7, 4, 7, -16, -9, -6, -3, 2, -8, 2, -2, 4, 1, 2, -9, -7, -16, -2, -2, -3, -6, -3, -6, -5, 4, -7, -6, 2, 10}
, {-2, -10, 24, 5, -1, 4, -13, -9, -11, 14, -16, -5, 14, 8, -4, 5, 7, 0, -9, -8, 6, 6, -15, -3, -5, 3, -7, -13, -11, 13, -7, 2}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_2_H_
#define _BATCH_NORMALIZATION_2_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       8

typedef int8_t batch_normalization_2_output_type[8][64];

#if 0
void batch_normalization_2(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_2_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_2_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_2.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       8
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 6
#define INPUT_SCALE_FACTOR 6
#define OUTPUT_SCALE_FACTOR 6
#define NUMBER_T int8_t
#define LONG_NUMBER_T int16_t


static inline void batch_normalization_2(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_2_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = scale(NUMBER_T, (LONG_NUMBER_T)bias[z], -INPUT_SCALE_FACTOR);
      tmp += (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      tmp = scale(NUMBER_T, tmp, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[x][z] = clamp_to(NUMBER_T, tmp);
#elif defined(ACTIVATION_RELU)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
        tmp = scale(NUMBER_T, tmp, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
        output[x][z] = clamp_to(NUMBER_T, tmp);
      }
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int8_t batch_normalization_2_bias[64] = {-57, -19, -23, -40, -30, -38, -13, -36, -40, -14, 0, -29, -41, -29, -66, -35, -11, -48, -26, -42, -34, -41, -26, -12, -39, -24, -6, -8, -30, -14, -36, -10, 7, -27, -43, -21, -31, -41, -10, -24, -28, -44, -38, -63, -34, -35, -43, -9, -17, -49, -12, -25, -22, -7, -23, -14, -31, -19, -4, -34, -25, -43, -42, -16}
;
const int8_t batch_normalization_2_kernel[64] = {56, 109, 66, 69, 46, 64, 53, 95, 43, 65, 53, 60, 36, 64, 79, 60, 81, 90, 66, 127, 43, 51, 106, 56, 36, 79, 42, 48, 37, 38, 127, 127, 73, 48, 49, 65, 82, 52, 79, 74, 62, 58, 40, 25, 127, 57, 35, 67, 48, 99, 38, 36, 53, 58, 27, 56, 57, 62, 70, 54, 81, 43, 51, 57}
;
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_2_H_
#define _MAX_POOLING1D_2_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   8
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int8_t max_pooling1d_2_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_2(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_2_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_2.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   8
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 6
#define OUTPUT_SCALE_FACTOR 6
#define NUMBER_T int8_t
#define LONG_NUMBER_T int16_t


static inline void max_pooling1d_2(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef WITH_CMSIS_NN
// Not really CMSIS-NN since using arm_relu_q* is not more efficient, but use SSAT anyway
#if ACC_SCALE_FACTOR - OUTPUT_SCALE_FACTOR > 0
      output[pos_x][k] = __SSAT(max[k] >> (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#else
      output[pos_x][k] = __SSAT(max[k] << (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#endif
#else
      max[k] = scale(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, max[k]);
#endif
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_3_H_
#define _CONV1D_3_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       4
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int8_t conv1d_3_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_3(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_3_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_3.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       4
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 6
#define INPUT_SCALE_FACTOR 6
#define OUTPUT_SCALE_FACTOR 6
#define NUMBER_T int8_t
#define LONG_NUMBER_T int16_t


static inline void conv1d_3(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 

      output_acc = scale(NUMBER_T, (LONG_NUMBER_T)bias[k], -INPUT_SCALE_FACTOR);


      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }
      
#ifdef ACTIVATION_LINEAR
      output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
        output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
        output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
      }
#endif
    }
  }

#else



  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 4 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q7_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q7_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q7_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q7_basic_nonsquare(
#endif
#endif
                                      (q7_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q7_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q7_t*)bias, //bias
                                      INPUT_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q7_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q7((q7_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q7((q7_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    64
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  3


const int8_t  conv1d_3_bias[CONV_FILTERS] = {10, -14, 7, -1, 2, 6, 4, 6, 17, -2, 4, 6, 4, 4, 3, 6, -5, 5, -5, 6, 15, 7, 4, 17, 19, 5, 1, 10, 17, 9, 8, 5, 10, 1, 11, 10, 8, 8, 5, 6, 7, -8, 8, 10, 6, 5, 3, 10, 2, 7, 11, -2, 3, 6, 2, 9, 9, 3, 2, 6, 7, 4, -6, 1}
;

const int8_t  conv1d_3_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS] = {{{4, 10, 22, -5, 2, -5, -12, -3, 24, 4, 2, 12, -1, 3, -6, -1, -5, -15, -3, -3, -1, 2, 6, -2, 5, 5, 11, 1, 0, 6, -7, -4, -11, 0, 1, -6, 0, -5, 15, 19, 2, -1, 6, 4, 12, -5, -4, 1, 4, -7, 16, 14, 11, 1, 1, -10, -16, 11, 1, 3, -12, 6, 0, 11}
, {-9, -1, 2, -10, 1, -3, 6, 1, 7, -4, 6, 6, 7, 6, 6, -1, 5, -7, -7, -7, -7, -5, -3, 3, -11, 13, 7, -6, -11, -5, -11, -7, -2, -5, -9, 5, 8, 2, 6, -1, -2, -3, -7, -2, -4, -14, 1, -6, -11, -12, 15, 4, -6, -5, -17, -13, 3, 2, -4, 4, -3, 1, -4, 0}
, {6, -14, 10, 8, 3, -16, -3, 4, 7, -1, -3, 0, 1, -6, -8, -10, -18, -11, 4, -10, -4, 2, 1, 2, -10, 20, 2, 1, -5, -1, -5, -14, -1, -10, -5, 8, 3, -2, -2, -5, -10, -5, -4, 1, -17, 4, 6, 2, -7, -10, 6, -1, 3, 2, -11, -9, 3, 17, -10, -4, -8, 2, 9, -9}
}
, {{-8, 0, -32, 18, 13, -15, 9, -17, -19, 2, -5, -17, -2, 14, -5, 12, -4, -12, -11, 13, 2, -10, 10, 8, 8, -11, -8, 0, -1, -7, -11, 21, 1, -14, -11, -29, -12, -12, 14, -9, 12, -2, -7, 3, 7, -1, 11, -35, -5, 7, -21, -10, -26, -5, 1, 5, 2, -12, 6, 24, 1, -10, -2, 0}
, {1, -11, -2, 4, 7, -6, -1, 13, 7, -14, -10, 11, 10, 0, 11, 3, -9, 13, -10, 6, 3, 2, 6, 1, -1, 12, 5, -15, -14, -11, -2, -3, 1, 4, -1, -3, 15, -7, -10, -10, 4, 3, 5, -6, -1, 8, 2, 9, 3, 11, 0, -5, 8, -6, -11, 12, 4, 13, -1, 8, 3, 7, -1, -10}
, {8, 13, 2, -5, 15, -11, 7, 4, -24, -2, 11, -2, 0, -6, 6, 0, 4, 16, -22, 1, -5, -4, -13, -1, -10, -4, 0, -1, 0, -4, 14, 22, 1, -6, -13, 2, 19, -33, 11, -10, 22, 5, -3, 7, -2, 8, -9, -7, -10, 5, 4, 13, -10, 8, -14, 21, -1, -3, -10, 3, 13, 3, 2, 2}
}
, {{4, 5, 0, -7, -4, 9, 2, -1, -6, 5, 11, 4, 4, 2, -15, 4, 5, -14, -3, -1, -5, 6, -15, 8, -3, 2, 7, -8, 6, -2, -4, 2, -5, -3, 5, 8, -4, 0, 8, 16, 14, 7, 4, -10, -8, 7, 13, -3, 5, 0, 7, 0, 1, 3, -7, 3, 3, -2, 6, -6, -8, 12, 1, -4}
, {-14, 2, 8, 0, 5, -3, 1, -1, 0, 8, -1, 2, 0, 5, 1, 1, 7, -9, -2, -3, 5, -6, 16, 4, 10, -1, 7, 4, 3, 2, -10, 6, 9, 4, -5, -8, 5, 6, 11, -4, 1, 6, 6, 2, 1, 9, 4, 1, -3, -9, 12, 5, -2, 12, -5, 7, -5, 3, 5, 13, -9, 3, 13, 9}
, {-11, 2, 1, -14, -1, 9, -10, 5, -3, -10, 8, -3, -4, -12, 2, 0, 9, 6, 7, -12, -1, -3, -4, -7, 3, -3, -1, -8, 5, 12, -8, 6, 6, 2, 1, 17, 6, -4, -8, 11, -6, -4, -3, 1, -23, -7, 5, -3, 8, -8, 2, 2, 9, 2, 1, -2, 6, 2, -1, -5, -8, 0, 7, -4}
}
, {{-12, -23, 15, 0, -9, -6, 1, 5, 10, 12, -8, 15, -3, -18, 18, -6, -6, -6, -16, -4, -6, -9, 18, -6, -1, 2, 6, -15, -14, 5, 5, -30, 14, 3, -3, -5, 4, 10, -15, 1, -28, -5, 0, -3, -1, 4, -21, 15, 1, 3, 0, -3, 1, -12, -11, -1, -8, 12, -8, 7, 1, -2, -9, -1}
, {4, -3, 7, -4, 4, 4, 3, 9, -8, -9, 3, 9, -6, -18, 10, 3, -2, 0, -6, 7, -17, 9, -8, 1, 3, 9, 0, 4, 0, 1, 7, 5, 7, 3, 5, 14, 12, 6, 0, -7, 7, 3, -5, 0, -7, -4, -2, -4, -2, 23, -3, -8, 0, -5, -8, 5, -1, -3, 9, 4, 22, 4, 4, 0}
, {5, -7, 6, 3, -7, 1, 3, 6, 7, 11, -2, 1, -13, -4, 0, 3, 1, -10, 6, -4, 9, -5, 2, 7, -1, -4, 2, -6, -4, -14, 8, -9, -9, 4, -1, 6, -9, 7, 6, 1, -5, 2, -10, -1, -8, -3, 7, 12, -12, -11, 5, 2, -4, -10, 6, -20, 3, 8, -6, -11, 4, 2, -7, -1}
}
, {{-11, -14, 3, 6, -2, -2, 5, 3, -3, -12, -8, 6, 0, 3, 5, 4, -2, 4, -2, 6, 5, 0, 4, -10, -4, 4, 6, -4, -4, -7, 2, -1, -2, -11, -14, -9, -1, 8, 1, -12, 14, 2, -4, 0, 11, 1, -9, -3, -10, 1, 8, -6, -9, -12, -6, -2, 5, 3, -4, 3, 6, -12, -10, 11}
, {9, -13, 2, -1, -8, 3, 1, -2, 5, 1, -7, 11, 1, 1, -4, -6, -8, 10, -7, 16, -5, 4, 4, -10, 6, 7, -1, -7, 4, -7, 13, -5, 10, -8, -5, -9, -2, -6, -9, 3, -12, -5, -5, -3, 4, -1, -9, -5, 1, 5, -5, 0, -8, -2, -1, -9, -9, -2, -5, -10, 4, 3, -9, -6}
, {6, -8, 5, 5, -5, -6, 3, 2, 2, -5, -9, 0, 10, 0, 0, 1, -11, -4, 0, 7, -4, -3, -3, -7, 4, 11, -1, 1, 4, 1, 6, -8, -1, -1, -1, -2, -10, -9, -2, 2, -11, -2, 7, 3, 1, 6, -2, 4, -7, 0, 3, -2, -1, -15, 0, -8, -4, 8, -8, 7, -13, -9, -3, 4}
}
, {{3, 6, -4, -7, -5, -3, 0, 4, -11, 10, 16, 2, 9, -15, -6, -7, -1, 5, 4, -4, -2, 3, 0, -10, -12, 5, -2, -1, 8, -4, 19, -7, -11, -4, 1, -5, -3, 2, -4, 4, 0, -3, -10, -2, -7, -2, -14, -6, 8, -3, 4, 14, 5, -5, -8, -23, 3, 2, -11, -15, -2, -11, -8, -2}
, {-3, -4, 4, 1, 0, -3, 0, -2, 8, 9, 1, 12, -8, 1, 7, -6, -8, 4, -9, -4, -8, -9, 8, -6, 2, 1, 4, 7, -7, -4, 15, -9, -1, -7, -5, -5, -4, -7, -3, -5, -17, -10, 2, -4, 1, 4, -4, 2, -2, 0, -1, -3, 3, -7, 0, -5, -8, 7, -12, -2, -1, -2, 3, -3}
, {13, -13, 4, 11, 6, 0, 0, 4, -4, -2, -2, 13, -9, -4, 3, -1, 0, 3, -19, 3, -11, -16, 3, 1, -1, -12, -8, 1, -15, 5, 11, -5, 13, 4, -8, -1, 4, 6, -5, -8, -2, 1, -1, 3, 13, 1, -1, 3, 6, 5, -2, -4, -2, 8, -15, 15, -1, -4, 10, 15, 9, 4, 5, -8}
}
, {{-12, 15, 6, -8, -7, 0, 5, 5, -4, 11, 9, 5, -1, -4, 11, 14, 4, 4, -1, -4, 2, -5, -8, 3, -22, 6, 1, -2, -8, 5, 12, 5, 5, -9, -7, 8, 1, -9, -1, -2, -2, -6, -14, -7, 4, 2, -7, -18, -1, -7, -2, -2, -3, 12, -15, -10, 5, 4, 4, -3, 0, -4, -7, 4}
, {-6, 3, 2, -6, -3, 7, -4, 4, -12, 2, -6, 0, 2, 1, 0, 13, 3, 4, 4, -5, -2, 7, -3, -2, -15, -4, 4, -5, -1, -1, 2, 15, 11, 7, -6, 5, 3, 10, 6, 5, -14, -4, -3, -15, 3, -2, -3, -5, 4, -8, 4, 3, -2, 8, -11, 6, 7, -12, 9, 3, 1, 2, 0, -7}
, {-13, 3, -10, -22, -17, -1, -3, 0, -4, -12, -17, -5, -4, 2, -8, 19, 1, -5, 16, -13, 7, 12, -13, -14, 5, -10, -1, -16, -2, 5, -14, 9, -2, -5, 8, 3, -17, 22, -17, 13, -11, 1, -25, -10, -12, -8, 8, -3, 5, -13, 2, -9, -1, -3, 0, -7, 6, -1, 3, -2, -1, -14, -12, -10}
}
, {{-7, -3, 4, 7, -9, -11, 10, -2, 21, 17, 14, 17, 11, -1, -2, 5, 12, 0, -6, 0, 13, -10, -15, -4, -7, 1, 3, -1, 10, 7, -6, 6, -5, -5, -11, -8, 14, 0, -4, 1, -1, 9, 7, 7, -10, 11, 4, 9, -5, -12, 2, 1, -4, 8, 3, 13, -5, 18, -5, -10, 3, 8, 4, -1}
, {-6, 4, 7, -5, 3, 13, -6, 2, 1, 15, 1, 11, 2, 3, 2, 11, 0, 0, -1, -4, -2, -7, 1, 12, -6, -4, 3, -15, -8, -3, -11, 0, 15, -2, -9, -2, 11, 4, 1, -3, 6, -1, -4, -2, -4, 10, 2, 3, -1, -10, 2, 0, -1, -3, -7, 6, 1, 1, 14, 16, -8, 8, 1, 4}
, {-16, 17, 5, -6, -14, 18, -8, 5, 9, 5, -3, 7, 5, -4, 0, 3, 0, -1, 8, -10, 11, 1, -3, 9, -1, -1, 7, -4, 3, 6, -16, 5, -6, 8, 9, 4, -2, 9, 0, 7, 3, -8, 2, -9, 1, 2, -3, 10, 4, -15, 1, 1, 6, -3, 16, -11, 1, 13, 1, -6, -12, -9, -5, -2}
}
, {{14, 5, 1, -4, -6, -8, -7, 3, -5, 9, 6, 6, -6, -3, 4, -10, -7, 0, 2, 9, -11, 3, 1, 13, -4, 6, 2, 1, 1, -1, 4, -4, -5, 1, -1, 6, 6, 4, -4, 12, 5, -18, -2, -11, 0, -20, -4, 4, -4, 4, 10, 2, -8, -12, -4, -1, 1, -4, 16, -1, 5, 19, 10, 2}
, {7, -2, 5, -10, 4, 1, 1, 4, 2, -4, 7, -10, 1, -7, -9, -7, -3, 1, 0, 0, -3, 5, 0, 5, 6, -3, -2, 3, -12, 8, 1, -7, -7, -6, 2, 4, -12, -5, -1, -1, 2, -3, -5, -2, -12, -5, -4, -6, 1, 15, -9, 0, 6, -6, -8, 5, -5, 3, 5, -26, 6, 0, 4, 3}
, {3, -10, 12, 10, 13, 3, 1, 9, 3, 3, 8, 0, -3, 14, 0, -18, 0, 1, -1, 10, -8, -20, 4, -7, -2, 11, -1, 4, -2, -2, 7, -16, 4, 4, -5, -6, -5, -6, 17, -5, 5, -14, 17, -1, 11, 0, -10, -3, -2, 21, -1, 16, -9, 5, -6, 17, -7, 6, 5, -5, 11, 24, 1, 11}
}
, {{5, 0, -3, -7, 4, 9, -6, 6, -16, -5, -2, -7, -8, 11, -3, -16, 2, -2, 5, 1, 3, -3, -3, 7, -19, 3, -1, 6, -13, -3, 9, 11, -11, -5, -6, -4, 8, -4, 18, 3, 12, 1, -11, 7, 9, -11, -4, -4, -7, 1, -4, 1, -11, -1, -6, 1, -3, -2, -1, -11, -6, -12, 0, -10}
, {16, -5, 0, 2, 3, 6, -8, 12, 13, -6, -2, -4, -5, -10, 15, -2, 4, -4, -7, -2, 8, -3, 3, -5, -2, -3, 1, -1, -8, -3, 5, 1, -9, 4, 0, -6, -5, 5, -3, 10, 2, 1, -4, -5, 10, 5, -6, 8, 1, 4, -3, -1, 2, -6, -10, 3, 3, 12, -3, -10, 6, -3, -9, -9}
, {0, -1, -12, -19, -6, 3, -5, 3, -11, -7, -5, -10, -1, 0, 16, -2, 4, 7, -9, 3, -5, 1, -9, -5, -3, 15, -11, -13, 6, -10, 7, 1, -1, -8, -17, 13, 8, 0, -13, 13, 6, -8, -4, 2, -17, -5, 1, -6, 6, -2, -8, -5, -13, 8, -12, 17, -3, -6, -7, -15, 2, -9, -12, -7}
}
, {{-6, -15, 4, 5, -5, -9, 1, -4, 5, 5, -6, 1, 3, 29, -7, 0, -1, 2, 8, -2, -1, -5, 17, -12, 4, -2, 3, -8, -11, -8, 3, -8, 24, -13, -13, -4, -2, -5, 6, -15, 3, 3, 0, 4, -5, -1, -3, 0, 2, 5, 1, -2, -6, 4, 3, -4, -7, 0, -6, 14, -14, -5, -10, 12}
, {4, -11, 6, -9, -4, -6, 2, 6, 10, -8, 7, -2, -4, -10, 2, -6, -2, -2, -13, 7, -3, -9, -3, -7, -3, 2, 0, 3, 0, -3, -17, -9, 1, 4, 2, 1, 6, -7, -11, 1, 6, -1, 14, -2, -1, 0, 1, -5, -5, -2, -13, 1, 8, -3, -3, 3, -7, 2, -5, 2, 3, 3, 8, -7}
, {6, 0, 3, 8, -13, -7, 2, 16, 14, -10, 11, -7, -1, -6, -2, 2, -15, -8, 0, 11, -4, -3, 9, 6, -15, 12, -4, 9, -19, -14, 5, -3, 11, 0, 5, 4, -1, -4, 0, -3, 11, -4, 0, -11, -3, 0, -14, 3, -19, 10, 1, -7, -2, 9, -12, -16, -2, 15, -9, -8, -9, 0, -2, -2}
}
, {{-2, -11, -8, -2, 1, -15, -7, -16, 2, -2, -7, 3, -4, 5, 8, 0, -7, 17, -5, -11, -7, -10, 12, -1, -1, -9, 0, -10, 1, -2, 7, -13, 15, 4, -11, -10, 2, 0, 1, -4, -1, 0, -13, -7, 3, -9, -13, 6, -7, 0, -1, -5, -16, -5, 4, -5, -1, 1, -11, -7, -5, -16, -21, -2}
, {3, -6, -2, -3, 7, 3, 1, -10, 3, -4, -3, 6, 1, -3, 2, 11, -1, 4, 2, 6, 8, 1, 1, 2, -7, 15, -6, -10, -2, -3, 9, 1, -5, 0, -4, 0, 8, 0, 12, -3, 8, -3, 0, -2, 3, -1, 8, -9, -2, 2, -5, -3, -9, 10, -2, 1, -9, -2, -2, -1, -6, 4, -17, 1}
, {2, -2, -10, -8, 1, 10, -6, -4, 6, -1, -2, 4, -3, -1, -4, 5, -15, 5, -4, 0, 3, 0, -8, 0, 5, 8, -9, -14, 1, 0, 4, -3, 10, 1, 7, 1, 5, -3, -9, 0, 18, -7, -12, -14, 4, 4, -4, -6, 3, 1, 1, -3, -4, 5, -3, 7, 1, 2, 0, -7, 3, 3, -18, 4}
}
, {{-2, -10, 6, 17, 4, -8, 4, -6, 7, 6, -9, 5, -8, 15, -1, -8, -1, -6, -2, -8, -13, -20, 11, 1, 22, -10, -9, 8, -21, 0, -3, -11, 12, -10, -20, -19, 2, -5, 14, -11, -15, 4, 9, 0, 12, 11, -7, 9, 2, -9, 0, -9, -9, 1, 6, -13, -13, -4, 0, 22, -16, -7, 10, 5}
, {-2, 4, -10, -3, 4, -5, -1, 4, -4, -1, 1, 13, -7, -2, 0, -2, 0, -1, 8, 4, -10, -7, -9, 4, -12, 12, -13, 0, -2, -6, -7, 4, -9, -3, -7, -8, 1, -1, -8, -3, -2, -5, 5, -3, 2, -8, 2, -1, -8, 7, 5, 0, -2, -7, -4, -4, -4, 7, -2, 9, -13, 12, -1, -3}
, {11, -7, -3, 5, -8, -14, -9, 18, 2, 16, 11, 26, -8, -1, 0, -8, -5, -4, 3, 0, 0, -1, 2, 7, -23, 3, -7, 7, -22, -17, 8, -2, 0, -8, -12, -13, 0, -19, -7, -7, -14, 4, -9, -6, 7, 3, -4, 2, -4, -8, 15, 9, -5, -9, -2, -1, -7, 6, -10, 6, -6, 14, -14, -2}
}
, {{-5, -2, 18, 4, 7, -4, -4, 11, 14, 3, -10, -1, 11, -4, 4, 1, -16, -10, -16, -7, 1, 3, 8, -19, 4, 23, 9, 3, -18, 6, -9, -25, 10, 11, -10, 14, -5, -2, 2, 0, -40, -6, -2, -1, 21, 3, -25, 17, -18, -10, 3, -6, 12, -9, 3, -13, -24, 27, 5, 3, -9, -10, 0, 9}
, {9, 7, 4, 0, -8, 9, 2, 9, -11, 4, 0, -5, -12, -5, 5, -7, 2, -6, -2, 5, 2, 6, -6, 5, -8, -4, -3, -4, 0, 1, -9, -6, 3, -7, 4, 2, 10, 6, -22, -1, -13, -6, 5, -10, 1, -8, 1, 6, 11, -10, 5, -3, -1, -4, -2, -15, -6, 2, 7, 8, -2, -4, -6, 1}
, {0, -9, -6, -14, -4, -5, -4, 18, 8, -17, 2, 2, 0, 11, -3, -4, -10, 1, 7, -13, -12, 7, -6, 7, -1, -4, -4, -5, 5, -4, 0, -5, -18, -12, 2, 14, -14, 22, 9, 11, -23, 7, -15, 14, 3, -11, 24, -4, 9, -12, 1, -4, -10, -5, -5, -15, 11, -5, 2, -16, -6, -2, -7, 2}
}
, {{9, -10, -5, -8, -2, -7, 1, -2, 2, -4, 1, -3, -18, -11, 3, 0, -2, 6, -9, 4, -4, -3, -10, -6, -8, -3, -13, -16, 5, -15, 14, -3, -9, -2, 13, 0, -2, 4, -11, 0, 3, 6, -6, -13, -20, -5, 5, 2, 8, -6, -14, 4, 0, -5, -3, -3, -6, -1, -5, -17, -8, 10, -11, -8}
, {-2, 1, 2, 4, 11, 0, 0, 2, -3, -7, 9, -3, -14, 14, -14, -13, -7, 13, -7, -2, 6, -3, -1, 4, 3, -10, -11, -2, -4, -3, 0, -3, 10, -9, -7, -6, 5, -3, -2, 1, 11, 0, 2, -5, -3, -5, -2, -4, -3, 3, -13, -2, -8, 3, 10, -1, -4, -4, -2, -11, -7, -5, -4, 2}
, {7, 2, -2, 17, 2, -9, 4, 3, 0, 3, 5, -2, -13, 6, 3, -16, -6, -6, 11, 5, -7, -11, 13, 11, 0, 2, -2, 18, -8, -12, -3, 0, -3, 10, -2, -7, 17, -23, 14, -3, 9, 11, 11, -3, -3, -18, -5, -1, -12, 0, -1, -3, -5, -16, -5, -9, -18, -8, 2, 2, -3, 0, 13, -4}
}
, {{10, 8, 12, -2, -3, 14, 1, 3, 2, 3, 1, 3, 6, -4, 0, -4, 3, 2, -11, -9, -2, 4, -4, 6, 2, 7, 10, -3, 3, 7, 0, 8, -6, 3, 0, 1, -4, 8, 7, 5, 7, 0, 1, -3, 1, -3, 5, -13, 10, -5, 3, 1, 3, 8, 15, 4, 3, 2, 0, -4, -8, -6, -7, 3}
, {-8, 7, 1, 4, -1, 4, 10, 9, 1, 11, 4, -6, -5, 1, 1, 5, 14, -4, -6, -2, -3, 2, 11, 13, -6, 4, 3, 3, 9, -5, -12, 2, 0, -1, -1, -7, -4, 7, -2, 2, 7, 2, -2, -1, -9, 3, 9, -6, 3, -10, 12, 0, 9, 7, 0, 0, 2, 4, 5, 9, -9, -11, 12, -5}
, {2, -1, -11, -6, 4, -5, 1, -1, -1, 7, 0, 10, 3, 5, -1, -4, 12, 2, 4, 3, -11, 7, -5, 4, 0, -5, 5, 5, 8, -8, -10, 1, 9, 7, 4, 3, 5, 5, 11, -2, -1, 9, 3, -5, -3, -3, 9, -6, 5, 5, 9, -4, -5, 1, 3, 11, 13, -6, 6, 10, 1, 1, 10, 11}
}
, {{-4, -10, -13, 0, -4, -9, 1, -11, -18, 3, -11, 13, 1, 20, 1, 19, 3, 1, 5, 7, 2, -17, 8, 0, -2, 9, -9, -4, -8, -11, -1, 3, 2, -12, -11, -5, 3, 1, -5, -25, -2, -5, -17, -6, -4, 1, 2, -18, -9, 1, -23, -15, -16, -12, -8, -4, 2, -1, -10, 3, 1, -4, -8, 2}
, {4, -11, -4, 3, -10, 11, -3, -10, 1, -8, -5, 0, 4, -3, 5, -2, -5, 6, -7, 5, 10, -2, 4, -12, -7, 14, 2, -1, -2, -1, 0, -13, 3, -5, 3, -3, 11, -2, 1, 2, -3, -5, 1, 4, 2, 0, -9, 2, 2, 0, 5, -3, -4, -12, -1, 2, -10, 7, -7, 12, 16, -5, -8, 4}
, {-4, -1, -2, 9, -8, -14, -2, 6, 6, -7, 10, 2, 0, -8, -5, 4, 0, -1, -11, 13, -10, -6, -1, -2, -12, 12, 6, -5, 1, -8, 10, 14, 8, -8, 1, -3, 7, -19, -18, -4, -9, 14, 6, -7, -2, 5, -6, -2, -5, -4, 1, -2, 9, -9, -5, 6, -18, 11, -10, 9, 17, 2, -2, -5}
}
, {{-8, 11, -20, -4, 4, 5, -10, -10, -18, -6, 10, 14, -21, -2, -11, -4, -6, -3, 1, 11, -14, -9, 16, 8, -6, -5, -3, -3, -6, -12, 4, 5, 8, -10, -10, -1, -3, -2, 16, -14, 9, -7, -8, -13, 3, -10, 5, -18, -4, -4, -7, -17, -18, -9, -1, -1, -1, -21, 12, 12, 1, 1, -1, 11}
, {-2, 4, -11, 4, 7, -3, -12, 7, -16, 2, -6, 8, -10, -8, 5, -4, -13, -1, 1, 1, -4, 4, 4, -10, 2, -8, 2, 0, -8, -5, 4, 4, 1, 4, -6, -2, 3, -8, 4, -9, -13, -3, -4, -10, -4, -4, 0, -3, -13, 7, 8, -1, -8, -4, 4, -4, -1, 1, 5, 5, 3, 8, -1, 4}
, {-3, 5, 6, -4, -10, -7, -12, 8, 12, 15, -3, 10, 2, -6, 0, -1, -14, -2, 12, -1, 8, 7, -5, 7, -3, -3, 12, 1, -11, -5, 0, -20, -16, 13, 11, -5, -18, -13, -3, 4, -30, -6, -5, -14, 1, -5, -1, -1, 4, -5, 8, 7, 5, -22, 10, -23, -6, 11, -10, -1, -19, 11, -8, -12}
}
, {{6, 0, -2, -11, 3, -3, -8, -1, -12, 4, -3, 1, -10, -10, -1, -9, 9, 3, 5, 7, -4, -8, -4, -8, -4, 1, 0, -9, -3, -10, 1, 15, -4, -2, 9, 0, 6, 0, -10, -5, 0, -9, -3, -9, -1, -13, -9, -17, 4, 1, -4, -3, -5, 0, 1, 6, -5, -2, -6, -11, 8, -4, -9, 0}
, {4, 0, 3, 8, -3, -1, -12, 2, 0, -1, 5, -1, -5, -11, 4, 6, -3, -9, -3, 2, -1, -4, -4, -3, 1, 1, -8, -9, 1, -1, -1, -4, -4, -7, 2, 1, -13, 6, 4, 6, -6, 2, 0, 1, -10, -7, 0, -1, -1, -2, 8, -6, 2, 2, -2, -2, -9, -1, 3, -11, 0, -2, -5, -5}
, {5, 1, -10, -5, 2, -4, -8, 10, 6, 5, 0, 2, 5, 7, -9, 8, -1, -1, -2, 8, -2, 3, -5, 2, -1, 1, 2, -7, 1, -3, -4, 3, 17, 5, 1, -6, -3, 12, -3, -6, 13, 11, -4, 6, -7, -1, -1, -9, 10, 23, -4, 10, -9, 7, 0, 7, 3, 3, 11, -1, 25, -4, -2, -5}
}
, {{1, 11, -2, -1, -3, 12, 10, 1, -2, 5, 4, -9, 2, 9, -4, 7, 9, -6, 3, -1, 15, 1, 6, 0, 3, 1, -1, 6, 12, 4, -4, 8, -3, 5, 8, -7, -10, 2, 16, 8, 3, 4, 7, 6, -2, 2, -5, -6, 3, 2, 2, 4, 5, -1, 6, -6, 7, 1, 6, 2, -7, 2, 12, 5}
, {-4, 14, 5, 3, -2, 2, 1, 5, -2, 7, 6, 3, 0, 5, 2, -3, -2, -15, 7, -1, -7, -6, 7, 17, 3, -5, 4, 6, -2, -4, 0, 4, -3, 4, 0, -2, -7, 1, 16, 4, 12, 9, 1, -1, 1, 0, 4, -5, 4, 1, -1, -3, -6, 4, 9, 2, 7, 4, 8, 0, -10, 1, 6, -7}
, {2, 7, 2, 11, -1, 3, 5, 10, -2, 11, 5, 6, -11, 3, -8, -6, 2, -12, 0, 3, 0, -10, 6, 11, 0, -7, 4, 8, -13, -13, 11, 7, 0, -6, -11, -11, -4, -5, 12, -7, 5, 16, -1, 8, 8, -9, 1, -1, -9, -4, 7, -6, -3, 5, -3, 6, -1, 1, 6, 17, -5, 6, 13, -1}
}
, {{13, -3, 4, -2, -5, -11, -5, 0, 5, -7, 11, 18, -11, -17, 6, 8, 5, 0, -3, 1, -8, 5, -12, -1, -10, 1, 4, -3, 6, -15, 15, -8, -5, 9, -3, 2, 4, -12, -19, 4, -2, 1, -7, -16, -4, -3, -1, 8, 10, -1, 5, 5, 8, -1, -17, -1, 1, 9, 2, -17, 11, 40, -5, -22}
, {-10, -8, -1, -9, 7, 4, -11, -4, -7, 0, -6, 12, -4, 3, 1, -4, -10, 3, -1, 12, -1, -1, -2, -18, 6, 6, -3, -11, -4, 8, 2, -3, 3, -7, -8, 4, 8, 4, -3, -4, 1, -16, -6, 2, -1, -2, -9, -10, 2, -5, 1, 1, -15, 5, -10, 5, 2, 0, -3, -9, 2, -6, -13, 3}
, {-7, -8, 0, -2, -2, 13, -6, 0, -6, 5, 7, 19, -3, -10, -15, -4, -3, 10, -1, 0, 4, 9, -5, 9, 5, 0, 4, -1, -1, 13, 2, 2, 8, 3, 5, 10, -3, 8, 1, 0, 9, -6, 2, -3, 2, -8, 2, 0, -6, -8, 8, -17, 3, 13, 8, -3, 0, 2, 6, 16, -6, -4, 3, 2}
}
, {{0, -2, 1, 3, 4, 5, 1, -10, -7, 14, -5, 10, 10, 8, -7, -7, 3, -4, -3, 1, -1, 2, 7, 13, -15, 10, -5, 4, -3, -10, 3, 3, 0, -8, -9, -9, -4, 5, 5, 0, 12, -6, -12, 6, -15, -4, -2, -8, -8, -9, -9, 7, -4, 2, 0, -15, -6, 6, -1, 1, 3, -1, -10, -5}
, {-10, -1, 8, -7, -5, 9, 0, 17, -7, 5, -7, -1, -1, 5, 1, -8, 1, -6, 3, -2, 0, -3, 3, -1, -1, 3, 4, -1, -8, 5, -2, 12, 11, 0, 7, 20, 3, -4, 2, 15, -13, 5, -1, -11, -13, 0, 1, -3, 11, 6, 10, -1, -2, 3, -4, 6, -9, -5, 4, -13, -3, 9, 0, -5}
, {-4, -15, -22, 2, -6, 2, 0, -11, -21, -14, -9, -8, -5, 8, -10, -4, 5, -7, -3, -7, -8, 6, -11, -2, -3, -5, -2, 2, 3, 5, 0, 6, 2, -5, -2, 3, -4, 5, -9, -6, 9, 14, -11, -2, 2, -13, 0, 2, 2, 8, -5, -16, -4, -3, -11, 15, 12, -19, 4, -4, 7, -7, -8, -1}
}
, {{-3, 9, -14, -11, 8, 9, 4, -20, -16, -7, 5, -8, 5, -1, -12, -15, 6, -4, 15, 7, 0, 1, 1, 16, -13, 4, 4, 7, 4, -2, -2, 9, -3, 2, 9, -7, -5, 4, 8, 3, 17, -10, -2, -7, -4, -21, 10, -16, 2, -1, -4, 4, 7, -2, 9, 0, 6, -2, 13, -1, -4, -1, -8, -12}
, {-5, 0, -3, -10, -5, -2, 6, 1, -2, 0, 9, -1, -4, -7, 1, 0, -4, 3, -2, 7, 0, 0, -2, -1, 2, -1, 2, -11, -12, 0, 2, -22, -8, -1, -1, -2, -9, -3, -9, 6, -5, -11, -10, -1, 6, -6, -7, -7, -2, -10, -10, -9, -3, -3, 3, -11, -14, 3, -10, -18, -4, -13, -8, -8}
, {4, -13, -1, 25, -13, -13, 4, 0, 4, 7, -13, 4, -1, -15, 0, 0, -10, 17, -25, 11, -4, -8, 10, 2, -22, -4, 10, 3, -21, -6, -3, -9, 7, -7, -13, -7, 10, -26, -18, -19, -11, -3, 7, -12, 1, 10, -15, -3, -26, 9, -1, -5, -3, 0, -12, -3, -21, 4, -16, 1, -7, -8, -3, -7}
}
, {{-7, -9, -1, 2, 1, 1, 8, -11, -6, 2, -3, 0, -2, -5, 9, -5, 5, -6, 13, 2, 0, 11, -9, 10, 2, 0, 5, 8, -3, -5, -13, 9, 3, -5, -14, -16, 16, 10, 4, -2, -2, -6, 9, 4, 1, 10, -2, -8, 6, -14, 2, -2, 2, 3, 7, -4, 7, -7, 13, 14, 4, 5, 17, 10}
, {-12, 8, 2, -7, -8, 4, 4, 5, -1, 0, -3, 2, -2, -2, 5, 3, 1, 0, -1, -7, -11, -3, -3, 4, 3, -1, 2, 11, -3, 3, -6, 2, -15, -5, -3, -7, -8, -1, -13, 5, -7, 6, -8, -1, -10, -9, 7, 8, -6, -4, 3, 6, -5, -1, -2, -10, 0, 4, 3, -14, -5, 0, 6, 5}
, {6, -4, 2, 12, 15, -4, 18, 0, 0, 2, -1, 3, -12, -10, -1, -14, -5, 6, -15, 15, -8, 8, 12, 9, -8, 7, 5, 22, -11, -12, 2, -4, 6, -4, -14, -7, 10, -15, -4, -6, 16, -11, 9, -1, 3, 3, -7, 13, -16, 0, 3, 5, -1, 7, 4, -7, 2, 2, -7, 9, -4, -12, 17, -2}
}
, {{-3, -18, -2, -2, -4, -7, 7, -17, 0, -19, 2, -4, -1, 0, -1, -5, -10, -4, -3, 4, 5, 1, 3, 12, -10, -13, 14, 1, 1, -5, 1, -3, -7, -11, -16, -5, 14, -5, -2, -11, 22, -5, -8, -13, 8, -4, 3, -3, -15, 4, 1, -16, -10, -1, 9, -5, 18, -12, 4, 19, -5, 5, -13, -2}
, {1, -6, 2, -1, 2, -2, 11, 5, 4, 2, -2, 6, -9, -6, -6, 3, -4, 6, -3, -2, 0, 4, 12, 2, -1, -14, -7, 4, -15, 5, 4, 6, 3, -10, 5, -7, -6, -11, 17, -4, 2, -2, -4, -12, 4, -3, -9, 0, -1, -2, 9, -3, 10, 1, -2, -8, -6, -10, -7, -2, -10, 15, 8, 5}
, {-1, 8, -13, -7, 4, 24, -5, -2, -5, -7, -5, 0, -5, -3, -14, 3, 10, 10, 3, 2, 7, -3, -1, 3, 13, -6, 0, -9, -3, 16, -1, -1, 0, -5, 8, -3, 1, 5, 6, 5, 12, -20, 2, -13, -6, -5, -2, -1, 1, 6, -10, 2, 3, 7, 7, 6, 1, -2, 16, 5, -9, -12, 2, 3}
}
, {{1, 4, 1, 1, 1, 4, 1, 10, -4, 11, 8, 13, 8, 0, -15, 16, -1, -2, 9, -5, 4, 14, 2, 6, 3, -9, 10, 5, 1, 4, -2, 5, -3, 11, 8, -8, -5, -2, 4, -15, 7, 12, 2, 5, 2, 15, 15, 6, -5, -5, 12, -5, 8, 2, -7, 1, 9, -6, -4, 3, -4, 15, 8, 3}
, {-4, 4, 4, 0, 4, 4, 4, -4, 6, 1, 7, 0, 2, 2, -9, 2, -1, -4, 3, -10, 4, 8, 5, 3, -3, 7, 1, 0, 3, 8, -6, -3, 0, 2, 1, 3, 3, -2, 5, -2, 6, 1, -1, -6, 9, 8, -2, 2, 3, -7, 1, -7, -2, 3, 7, -1, 3, -3, 1, 12, -5, -11, 4, 0}
, {7, 4, 9, 5, 5, -1, 3, -5, 6, 19, 8, 17, 7, -6, -3, 2, -1, -7, 2, 3, 1, 4, 8, 11, -3, 11, 0, 9, 6, 7, -3, 1, -4, 5, 5, -3, -2, -6, 3, 2, 8, -2, -1, -4, 10, 15, 5, 4, 2, -10, 16, 8, 5, 3, -3, 3, -4, 13, 1, 17, -4, 4, 11, 3}
}
, {{-12, -7, -8, -1, 4, -4, 6, 8, -7, 3, 5, -5, -3, 1, 4, -1, 6, -1, 0, 8, 5, -2, -7, 1, 2, -8, 5, 0, 8, 6, 1, 5, 7, -8, -9, 7, 9, 3, -4, 1, 8, 10, 2, -7, -12, 3, -5, 5, 6, 1, 5, 5, 1, -3, 0, 13, 6, 4, -9, -8, 15, -6, 6, -2}
, {-10, 5, -4, 0, 7, -1, -4, -10, -5, 4, -1, 0, 7, 7, -9, -5, 2, -3, 6, 0, 0, -6, 3, 9, -3, -1, 1, 5, -3, 3, -1, 14, 4, -4, 0, 0, 10, -2, 17, -2, 7, 3, 9, -2, -2, 4, -5, -9, 9, 10, 3, 5, 3, 6, 9, 5, 8, -12, 14, 1, 13, -1, 3, 9}
, {-1, -13, -4, 5, 8, 0, 2, -5, -7, -12, -3, -12, 7, -2, -14, -6, -7, -12, -7, 8, 2, 4, -1, 6, -1, -12, 7, 10, 15, 9, -13, -3, 10, 10, -1, -6, -7, -5, -6, -2, 3, 7, -6, -1, 0, 12, 7, -1, 11, 2, 2, 1, -5, 7, 3, 9, -3, 3, 8, 4, 19, 0, -8, 10}
}
, {{-6, 4, -4, -4, 10, 1, -2, -15, -4, -2, -2, -21, 5, 17, -13, -4, 4, -13, 3, -16, 4, -11, 2, 9, -3, -6, 9, -4, 1, -4, -2, 14, 12, -6, -12, 2, -2, 13, 9, -8, 16, -7, -3, -3, -3, -7, -2, -7, 3, -17, -4, 6, -6, 3, 7, 10, -1, -11, 6, 9, 3, -1, 3, 12}
, {0, 0, 5, -3, -5, 7, 11, 3, 2, 5, -2, 0, 1, 10, 5, -3, 4, -12, 1, -1, 1, 2, -5, -1, -3, 7, 1, 3, 3, 7, -5, 9, 1, 6, 3, 5, 1, 3, 6, 1, 2, 5, 6, -2, 2, -8, 16, 4, 8, -9, 6, 5, 4, 13, 7, 6, 4, -1, -5, -3, -12, 1, -5, -7}
, {0, 9, 13, -2, 12, -5, 3, 8, 0, 7, 10, 0, 5, 8, 0, -6, 8, -10, 5, -1, -6, -4, -7, 7, -1, 13, 2, 2, 8, 4, 7, 5, 1, -6, -2, 6, 3, 0, 3, 2, 4, 13, 2, -1, 3, -2, 6, -9, 3, -9, 12, 12, 3, 13, -9, 8, 4, 2, 3, -6, 6, 2, -13, -2}
}
, {{25, 2, 0, -9, 10, -7, -6, 5, 2, 1, -11, 5, -15, 8, -16, 2, 0, 0, 9, 2, -5, 4, 6, 0, 5, 4, -2, 11, 1, -9, 7, -8, -2, 5, -9, -10, -5, 2, 9, -10, 13, -11, -3, 1, -3, -17, -3, -1, -7, 3, 1, -1, 3, 8, 2, -8, 11, 2, -1, 1, 1, 34, 2, 5}
, {-9, -3, 8, 0, 4, -6, 1, 4, 3, 0, -11, 5, 7, 8, 6, 12, -2, -6, 6, -3, 2, -17, 7, -5, 3, 11, -2, 1, 1, -1, 3, -17, -4, -10, 2, -4, -2, 9, 7, -5, -8, -8, 13, -5, -10, -10, -5, 14, -9, -20, 4, 0, -2, -12, -4, -8, -3, 8, -16, -1, -12, 9, -9, 7}
, {-2, 6, 14, 15, 2, -3, -4, 5, -3, 8, 10, -4, -6, 2, 0, -7, 2, -9, -2, -9, -9, -17, -13, 14, -11, 12, 1, 11, -15, -8, 4, -1, 8, -5, -16, -9, 27, 2, 6, -4, 2, 13, 16, -6, 12, -9, -3, 6, -11, 10, 6, 2, 6, 1, 2, -2, -1, 6, -2, 13, -3, 9, 1, 1}
}
, {{19, 3, 8, -1, -11, 4, -20, 19, 3, 4, 16, 15, -6, -15, -2, 8, 3, 4, 8, 8, 8, 15, -8, 9, -11, -1, 5, 4, 11, -13, 6, 0, -7, 6, 7, -4, 6, -9, 11, 16, 0, -3, -9, -3, 4, -12, -5, 13, -2, 10, 6, 10, 13, 7, -4, -19, -5, 17, 10, -11, 2, 16, 7, -13}
, {-3, -3, 2, -1, -4, -5, -4, 9, 6, 1, 11, 4, 0, 4, 8, 2, -2, -5, 0, -4, 6, -12, -8, 7, -5, -3, 2, -5, 2, -6, -9, -9, 5, -7, 2, 4, 4, 9, -3, 14, -25, 4, -7, -10, 0, 0, -7, 9, 0, -4, -3, 2, 2, 3, -2, 2, -12, -1, 4, 0, 4, 10, -13, 5}
, {-3, 2, -6, -3, -15, 8, -6, -1, 11, 4, -7, -9, -3, -19, 1, 3, -3, 0, 8, -7, -5, -1, -6, -6, 3, 4, 5, 0, 5, 15, -24, -5, -2, 4, 12, 13, 1, 6, -4, 17, -26, -2, -6, -7, -11, -6, 3, 2, 17, 5, 4, 0, 9, -17, 4, -1, -6, 3, 0, -3, 0, -4, -6, 1}
}
, {{-6, 2, 2, 2, -4, -7, 1, -4, 0, 9, -4, -6, 12, 7, 16, 9, -1, 3, 6, -14, 5, -9, -1, 0, 7, 7, -4, 4, 1, 4, 0, -20, 12, -5, -6, 0, 7, 5, -2, 4, -23, -7, 6, 2, -12, 6, -11, -1, 2, -16, -2, -4, 3, -1, 6, -6, -1, 5, -3, -1, -12, -20, -5, 8}
, {5, 1, 10, 8, -15, 3, -8, 7, -2, 5, 13, 1, 2, -5, -2, 6, 8, -5, 1, -3, 0, 2, -2, -6, 1, 5, 10, 10, 9, 9, -4, -2, -9, 7, 13, 5, 7, 16, -3, 16, -1, 7, -5, 6, 3, -1, 0, 10, 14, -5, 6, -1, 4, 8, 6, -4, -8, 11, 8, -5, -14, 6, 2, 4}
, {4, -3, -9, 3, -2, 3, -1, 6, 1, 1, 15, 1, 13, -2, 0, 3, 0, 1, -5, -3, 0, -4, -1, 8, 5, 3, 8, 5, 1, -9, -1, 5, 9, 3, -10, 14, 18, -1, -4, -5, 6, -8, 8, -10, -8, 2, -3, 3, 4, -4, 0, -2, -1, 11, -2, 17, 4, 6, 2, -1, 11, -6, -5, 11}
}
, {{-7, -14, -2, -8, -6, 10, 8, 7, -10, -2, 4, -8, -5, 7, 4, 0, -7, 2, 1, -2, 2, -1, -13, -8, 10, -8, -7, -7, -4, 17, -2, 5, 4, 7, 2, 6, 5, 2, -4, 6, 0, -9, 6, 1, -1, -4, -6, 8, 9, -6, 2, -7, 0, 1, 3, 7, 2, -2, -6, 2, 6, -12, 2, 4}
, {5, 14, -8, 2, 10, -4, 5, 10, 8, 4, 7, -15, 0, 4, -10, -4, 4, -9, 5, -1, -4, -7, -4, -6, -3, 4, -7, 17, 7, 9, 3, 11, 13, 11, -1, -3, 4, 2, 3, 7, -8, 10, 2, -6, -4, 2, -4, 1, -2, -2, -1, 6, -3, 12, -6, 0, -6, 5, 6, -8, -1, 6, 11, 1}
, {0, -5, -19, -6, 2, 1, 9, -7, 2, 7, 7, -5, 15, -2, -6, -5, 8, 2, 2, -13, 4, 3, -16, 5, 4, 0, -2, 1, 3, 10, -13, -4, -2, 9, 3, 0, 5, -3, 3, 9, 2, 1, 1, -1, 0, 0, 0, -4, 17, 1, 3, -2, 2, 4, 14, 8, -3, 7, 0, 0, 8, -11, 8, 9}
}
, {{0, 11, -20, 1, -6, 9, -11, 5, -22, -9, -6, 0, -14, -10, 2, -5, 9, -2, -2, 17, -15, 2, -7, -6, -4, -1, -9, -7, -1, 10, -5, 12, -6, -19, 18, 7, 10, -2, -7, -5, 12, -5, -12, -6, 0, -9, 7, -13, 2, -7, -2, -9, 0, 11, -11, 9, 6, -1, 10, -1, 10, -5, -7, -9}
, {-15, -2, -3, -20, -9, 9, 0, -3, -14, -4, 11, -10, 1, -1, -8, -3, 0, -5, -5, 1, 5, 2, -17, 12, -3, -8, 3, 1, -3, 11, 2, 3, 0, -9, -1, -6, 3, -1, 1, -7, -1, -5, -11, -9, -3, -4, 3, -9, -6, -8, -1, -1, 1, 7, 3, 9, 12, 5, 10, 6, 2, -8, 1, 3}
, {-7, 10, 4, 0, 3, -2, 3, 12, 8, 5, 12, 1, -7, 5, -6, 0, 6, 3, 1, -9, 8, -3, 8, 13, 10, -6, 4, 8, -5, -1, -1, 10, 0, 3, 3, 6, -10, -1, 6, 2, 22, 2, 0, -5, 5, -3, -9, -9, -1, -4, -11, -8, -3, 9, 7, -7, -1, 9, 5, -3, -12, 1, 11, -4}
}
, {{5, -11, -4, -6, 11, 4, 7, 16, 10, 0, 0, 4, 2, -5, -7, 16, -4, -4, -6, 3, -1, 5, 2, 5, -11, 9, -4, -9, -7, -10, -6, 0, -2, -13, -3, -1, -6, 1, -6, -7, 12, -4, -6, -14, 1, -4, 11, -10, -10, 0, -3, -8, 0, -4, -21, -7, -5, 12, 9, -8, -7, 11, -16, -9}
, {0, 5, -1, -2, -5, -3, -2, 3, -2, -14, -2, 2, -4, -15, -10, -3, -12, -2, -3, -2, -14, -4, -5, 1, 4, 20, 4, -9, 1, -4, 10, 1, 2, -2, -5, -1, 6, -6, 4, -5, 7, -12, -11, -6, -2, -6, 12, -11, -8, 6, 3, -9, -7, 1, -13, -2, -8, -5, -3, 5, -2, 2, 3, 4}
, {5, 6, 4, -4, -11, -8, -2, 2, 4, -4, -4, -1, -10, 9, 3, -2, -9, -10, 7, 2, -6, -9, -5, -12, -2, 16, -12, 3, -2, -4, -10, 5, -3, -3, 16, 4, -20, 3, -5, 5, -9, 0, -15, 3, -14, -7, -4, -3, 12, 4, 6, -5, -12, -9, 14, 1, -4, 9, -13, -13, -1, 2, 0, -9}
}
, {{2, 6, -3, 3, 13, 15, 7, -1, -5, -3, 5, 0, -1, 2, -4, -3, -9, -10, 0, 3, 4, 2, -3, 6, 1, 3, 1, 0, 7, -3, 0, 5, 12, 8, -6, 1, 0, -1, 8, 11, 17, -4, -2, -1, 6, 0, 9, -1, 3, 6, 12, 4, 7, 2, -1, 4, 12, 6, 7, 14, 7, 5, 4, 9}
, {-7, 3, 2, 3, -3, 1, 1, 4, -5, 1, 9, 0, 5, 4, -14, 9, 9, -2, 6, 0, 0, 7, 3, 4, -2, 12, -1, 1, 6, -6, 5, 6, -9, -8, 6, 9, 9, -2, 6, 4, -4, 7, -1, -8, -13, 1, 9, 2, 7, -4, 9, 13, 5, 9, 5, 0, -1, 5, -1, 13, 0, 15, 5, 7}
, {5, 1, -2, -4, -1, 1, 4, -1, 3, 1, 3, 2, 1, 10, -15, 13, 12, 5, 12, 2, -3, 10, -4, 11, 6, 2, 1, -8, -1, 13, -5, 23, -7, -14, 2, 1, 12, 6, -1, 6, 10, 8, 0, -5, -3, 2, 13, -3, 5, -22, 10, 7, 0, 9, 1, 14, 1, 10, 11, 6, 6, 4, -2, 7}
}
, {{-5, 6, 0, 16, 6, -11, 1, -2, 9, 11, 8, 0, -5, -2, 8, -3, -1, -8, -4, -10, 0, -14, -7, -5, 7, -5, 4, 3, -3, 0, -5, -16, 12, 0, -8, -9, 13, -3, -8, 3, -33, 4, 1, 4, -5, 7, 0, 10, 2, -1, 11, 3, 2, 1, -1, -11, -9, 5, -5, 4, -5, -7, 10, 3}
, {-2, -3, -4, 11, 1, -2, -9, 2, 1, 8, 7, -6, -3, 5, 3, 3, 4, -15, 0, -6, -1, 0, 5, -3, 2, -7, 9, 3, -8, 1, -7, 1, 13, 1, -5, -6, -3, 6, -7, 10, -4, 7, 11, 3, -7, 1, 0, 16, 5, -5, -2, 2, 3, 4, 0, -1, 6, -4, -6, 7, 7, -5, 7, 3}
, {0, -4, -3, 11, 2, -1, -4, -7, 8, 13, 9, 3, -14, -7, 0, -4, 3, -1, -8, -11, -3, -2, 6, 7, -7, -10, -5, 4, -5, 4, 5, 4, 23, 2, -6, 15, 8, -4, -8, 9, 10, 9, 1, -3, -4, 8, 0, 14, 2, 3, 4, 0, 0, 13, 4, 11, 4, -3, -1, 14, 18, 15, 27, 1}
}
, {{-13, 7, 8, -3, 8, 0, -9, 5, -4, 0, 0, 6, 13, 0, -15, -3, 1, -10, 8, -7, 1, 13, -6, 6, -11, 5, 8, 10, 8, -1, -9, 14, 11, 5, 3, 9, 7, 1, -4, 7, 6, 12, 19, -5, -13, 4, 5, 5, 3, -1, 1, 9, 10, 2, -10, 0, 1, 6, -1, 8, 14, 20, 12, -12}
, {-19, 4, -3, -3, -4, 4, 2, -17, 1, 2, 2, 4, 9, -11, -11, -1, 7, -5, 4, -7, 9, 11, -13, 1, -4, 0, 5, -3, 5, 7, -13, 4, 4, -1, -3, 0, 11, -2, 5, -5, 6, 5, 11, 6, -5, -5, -4, 10, -3, -11, 5, -2, -4, 2, 8, 8, 6, -2, -3, 9, 10, -8, -2, -1}
, {-18, 4, 0, 3, 3, 1, 0, -14, -9, 8, -6, -1, -7, 0, -7, -6, 17, -4, 6, -12, 3, 9, -3, -3, 3, 3, 5, 2, 11, -9, 5, 13, -7, 10, -14, -1, 8, -7, -10, -3, -6, 12, -1, -8, 20, -3, 9, -1, 9, 0, 2, 1, 7, -1, -9, 6, 3, 2, 6, 4, 16, 3, -4, -6}
}
, {{5, -6, -8, 1, -13, -12, 1, -4, -3, 9, 5, 11, -2, -11, 5, 5, -7, 6, -3, -1, -2, -1, -7, -3, -21, 1, 0, -11, -8, -9, 10, 3, 0, -4, -3, 11, -2, -4, -14, -3, -4, 5, -12, -14, 5, 11, -1, -6, -10, -2, 2, -1, 5, 2, -15, -3, -1, -1, 7, 1, 5, 15, -10, -2}
, {2, 6, -7, -1, -6, 8, -5, -1, -5, 3, 4, 12, 1, -11, -9, 2, 1, -2, 2, -2, 1, 1, -13, -6, -3, -1, 0, 4, -7, -7, 2, -3, -5, -10, -5, 7, -1, -19, -1, 8, -4, -2, -6, -10, 0, 3, 4, 3, 5, -3, 15, -5, 0, 4, -13, -3, 1, 7, 7, 1, -2, 2, -5, -12}
, {3, 1, -7, 8, 5, 1, -1, -6, -11, 2, 2, 22, -12, 5, -13, 4, -6, 1, 4, 0, -11, 2, 12, 5, -12, -7, -8, -7, -3, -15, -1, 0, -8, -2, -7, -14, 2, -23, 8, -5, -4, 7, 0, -3, -5, 8, -6, -14, -13, 19, 2, -9, -7, -6, -15, 10, -11, -11, -2, 2, -6, 11, -26, -9}
}
, {{-4, -12, 19, 1, -10, -10, -9, -4, 15, -7, 2, 4, -5, -15, 21, -2, 9, 5, -6, -28, -18, -18, -2, -6, 5, -14, -1, -3, -7, -5, -10, -16, 2, 15, -4, 9, 3, 5, -18, 4, -29, 7, 1, 2, -21, 9, -15, 40, 12, -9, 9, 8, 4, 4, -16, 0, -8, -12, 4, -4, -6, -4, 2, 4}
, {13, -1, -20, -6, 2, 1, 4, 1, -5, -8, 6, -3, -10, 4, -2, -2, 2, 10, -1, -5, -11, 2, -19, -5, -8, 7, -12, -21, 0, 2, -11, 3, 2, -16, -11, 14, 9, 4, -7, -10, -1, -10, -1, 11, -7, 3, -3, 18, -3, 0, 0, -3, -4, -8, 5, 4, 2, -5, 2, -12, -2, -5, -4, 5}
, {4, 7, 0, 4, 9, -7, 26, 7, 11, -11, 2, -9, 1, 3, 5, 2, -4, -8, -1, -1, -2, -7, -3, -2, -4, 5, 4, 11, -4, -3, 0, 0, -5, -7, -10, 3, 1, -9, 5, -1, -5, -13, 0, -2, 4, 4, -8, -13, -7, -4, 6, 2, -14, -13, 3, -5, -11, 27, -2, -8, -4, -5, -2, -1}
}
, {{-9, 11, 6, 8, 3, 6, -7, 1, -1, -2, 5, 0, 17, -10, 1, 0, 4, 3, -9, -7, 0, 5, 5, -4, -4, 7, 27, -4, -4, -3, 2, 5, 12, 15, 13, 13, 5, 6, -5, 20, 1, -1, -8, -15, -4, 16, -12, 10, 9, -5, 23, 10, 4, 3, -2, 4, 1, 11, 4, 7, 3, 0, 5, -2}
, {-5, 5, -10, 1, -5, 9, 8, 0, -9, 3, 0, 10, 4, 4, 4, 9, 5, 1, -3, -3, -5, 9, -8, -4, -11, 2, 14, -3, 8, 3, 2, 7, 5, -1, 4, 5, 13, 6, -3, 11, 0, 8, -2, 2, -2, 15, -4, -13, -1, -7, 12, 7, 1, 10, 6, 3, 16, -6, 0, -3, 8, -8, 4, 6}
, {-6, 8, -10, -3, -2, -7, 13, -1, 5, 3, 3, -2, -13, 10, -10, 10, 0, -4, -1, 2, -2, 5, -5, 6, 1, -9, 3, -2, 3, 5, -5, 4, 5, 8, -3, 10, 7, 6, -2, 1, -2, 3, 7, 6, -1, 0, 14, -3, -1, -9, 1, -4, 0, 10, -7, 4, 5, 0, -7, 5, 5, -2, -1, 12}
}
, {{10, -3, -7, -5, -11, 12, -6, 9, -14, -3, 12, 14, 2, -14, -14, 6, 5, -8, 6, -2, -4, 14, -12, 6, -9, -3, 0, 11, -2, 3, -5, 11, 0, -1, 6, 15, 8, -5, 3, 5, 1, 3, -9, -10, 0, 4, 15, -1, 14, 7, 14, -13, 10, 5, 4, 5, -4, -11, 6, 9, 9, 6, 19, -15}
, {-6, -1, -4, -11, -2, 2, 0, -15, 5, -4, 2, -1, 1, -6, -3, 4, 2, -1, 6, -2, 1, 5, -9, 7, -6, 1, -1, 1, 7, 5, -9, 13, 4, -4, 4, 5, -7, 0, 5, -3, -1, 3, -1, -6, -6, 9, 5, -9, 11, 1, -9, -5, 8, -2, -5, 11, 4, 1, 3, -9, 4, 7, -7, -1}
, {-9, 4, 4, -2, -3, -5, 2, -5, 10, -4, -5, 6, 12, -4, -3, 4, 6, -10, 9, -4, 12, -6, -2, 11, -4, 2, -1, 2, -1, 2, -10, 1, -4, 5, 5, 1, -8, 9, 8, 10, 0, -3, 0, -10, 3, 1, -4, -9, 13, -2, 0, 9, -9, 10, 3, -13, -3, 4, 9, 5, -7, 7, 2, -1}
}
, {{0, -9, 19, -7, -14, 3, -1, 12, 14, 5, 3, 6, -7, -6, -1, 0, -5, 8, -7, -7, 8, -8, -2, 1, -13, -8, 3, 2, -10, 0, -4, -21, -14, -2, 5, 7, 1, -9, 12, 16, -24, 12, 2, 0, 5, 0, -13, 5, -1, -6, 4, 2, 9, 7, -6, -12, -11, 5, -13, -11, -13, 0, -2, -6}
, {-1, 7, 5, -4, -1, 0, 1, -4, 6, 6, 0, 3, 3, -3, 11, 7, -7, -6, -9, -5, -7, 6, 6, -2, -5, -9, 3, -7, -7, -7, 2, -2, -3, -4, 10, -15, -12, 4, 4, 1, 9, 6, -16, -4, 11, 0, 1, 2, -6, 7, -9, -13, 1, -1, -5, -20, 2, -7, 8, -3, -5, 7, 1, -10}
, {4, 2, 8, -3, -7, 2, -15, 0, 8, 10, 1, 5, 10, -5, 11, 1, -1, 0, -3, -7, 7, -7, 0, -3, -9, 1, 4, -7, 2, 0, 18, -14, -13, 0, 12, 4, -7, 7, -7, 5, -4, 3, -13, 9, 4, -4, 2, 1, -5, -11, 8, -13, 4, 1, -2, -43, 3, 0, 0, -12, -21, 11, -16, -4}
}
, {{-4, 3, 11, -16, -1, -9, 12, 8, 7, -8, 23, 1, -3, -1, 2, 10, 3, 11, -4, 11, 5, -1, -8, 5, -13, -6, -7, -10, -9, -13, 9, -1, -9, -18, -8, 3, 1, 1, -8, -9, -3, -4, -3, 0, -8, 11, 3, 3, -8, 8, 18, 6, 9, 0, -23, 4, -2, 9, -18, -12, 9, -5, -1, -13}
, {1, 0, -11, -2, 0, -2, 2, -12, -5, -2, -1, -13, -1, 1, -10, 0, -5, -1, -15, 10, -12, -7, -10, -4, -12, -1, -4, -1, -4, -4, 4, -9, -1, -13, -8, 2, 19, 6, -12, -3, 7, -3, -6, -4, -3, 2, -8, -1, -10, 0, 0, -2, -4, -2, -6, 1, 10, 1, -9, 5, 3, 2, -2, 0}
, {-14, 5, 15, 5, 2, 3, -12, 1, 18, 5, -12, 9, 4, -4, -14, -7, -33, -11, 19, 3, -1, 19, 22, 14, -2, 26, 9, 8, -11, 0, -9, -11, -9, 6, 2, 11, -23, 14, 4, 5, -22, -12, -1, 6, -20, 1, 4, 15, 8, 4, 21, 3, -5, -27, 10, -25, -16, 26, -2, 1, -16, -6, -9, 2}
}
, {{2, 4, -1, 3, 1, 1, 6, -1, 3, 7, 9, -9, 10, -2, -11, -15, 0, -9, 10, -4, -1, 0, -11, 8, 2, -1, 3, 14, 3, 7, -4, -2, 4, 1, -6, 4, 1, 0, 13, -4, 1, 0, 7, 8, 9, 13, 2, -7, 5, -6, 2, 6, 10, 7, 7, 2, 7, 1, -4, -8, -11, -3, 17, -5}
, {8, 1, 1, 3, 0, -9, 2, 1, -16, 7, 3, 3, -10, 6, -7, 0, 7, -7, -3, -2, 1, -4, 3, 0, 5, 5, -1, 8, 2, 1, 12, 9, 8, 2, 4, -13, 5, 0, 2, 1, 7, 7, 1, 3, 3, 0, -3, -5, 2, 11, 8, -6, 3, 4, 1, -10, -4, -17, -1, -2, -1, 5, 11, -2}
, {1, 7, 3, -7, 4, 11, -9, 3, 2, 13, 13, 10, 7, -6, 4, 1, 10, 12, -3, -19, 2, -1, 8, 1, -5, 9, 5, 2, 18, 2, -4, -7, -1, 0, -6, -1, 5, -8, 20, 1, 1, -7, 10, -3, 6, 2, 6, -9, 7, -6, 4, 6, 4, 6, 10, -10, 21, -6, 5, -6, -2, -1, 11, 18}
}
, {{-1, 17, 6, -10, 4, 11, 4, 8, -12, -3, 9, 4, -5, -8, -17, 4, 10, -13, 5, 13, 2, 13, -13, 9, -19, 6, -6, 10, 4, 3, 0, 8, -14, -5, 7, 1, -8, -7, -2, -4, 16, 1, -8, -7, -3, -7, 14, -10, -9, -2, 2, 3, 14, 14, -2, 7, 3, 10, 18, -6, -3, 11, 3, -12}
, {-7, -7, 13, -7, -1, 1, -3, 8, 5, -3, 4, 4, 11, -8, 6, 11, 2, 1, 2, 1, 10, 10, 3, -1, 1, 1, 4, 1, 4, 7, 1, -8, -10, 4, 4, -1, -15, -3, 0, 0, 1, -3, -6, -13, 10, 2, 6, -4, -2, 7, 3, -2, -4, -9, -2, -1, 0, 14, -4, -4, -6, -6, -1, -4}
, {-10, -3, -2, -7, -8, -2, -1, -7, -14, -13, -2, 4, -10, -4, 6, -8, 12, 0, -21, -6, -12, -12, -15, -11, -8, -3, 4, -10, -1, -1, 4, 15, 24, -9, -8, 8, 0, -8, -18, -4, 0, 2, -12, -3, -11, 3, -4, -1, -5, 17, -7, -20, -4, 6, -18, 19, -1, 1, -10, -2, 18, 6, 8, 0}
}
, {{21, 10, -10, -6, 3, 6, -13, 0, -5, -9, 6, 5, -12, -2, -7, -6, -9, 3, 6, 3, 2, -8, -2, 3, -2, -5, -2, 4, -8, -11, 8, 8, 3, -4, 1, -8, 2, -2, 1, 3, -6, -4, -2, -2, 10, -20, -15, -10, -4, 15, -5, 3, -5, -1, 2, -8, -7, 4, 0, -8, -6, -4, -4, -5}
, {-1, 0, 9, 1, 4, -3, -9, -3, 8, 4, 3, 0, -2, -11, 7, 4, 0, 11, 2, -2, 1, -2, -1, 3, -4, -11, 0, -5, 1, 1, 2, -13, -1, 1, 5, 7, 0, 6, -10, 0, -9, -4, 2, -3, 7, 4, -10, 8, -1, -10, -5, 2, 3, 4, 0, -6, 0, 6, -7, -9, 4, 0, -2, -10}
, {3, -9, -2, 4, -1, 0, 5, -8, -7, -7, -6, -1, -10, -6, 0, 8, 4, 2, -4, 0, -14, -3, -5, -2, -8, -15, 1, 1, -11, 3, 1, 7, 4, 0, -6, 1, 16, -7, -1, 3, -2, -2, -6, -5, -9, 2, -13, -1, -3, 9, 5, -9, 3, 7, -5, 6, -12, -12, 9, 1, 8, -1, 8, -12}
}
, {{-11, -12, -1, 8, 8, 1, 0, 0, 3, 0, 5, -4, -4, 7, -11, -5, -6, -11, 2, -15, 8, 1, 0, -9, -8, -2, 0, 8, 2, 2, -7, -10, 5, 1, 0, -1, -3, 0, -1, -1, -8, -2, 4, -4, 9, 9, 7, 1, 6, -16, -10, -7, -3, 12, 4, -7, 6, -1, 1, -7, -9, -6, -5, 5}
, {-5, 1, -7, -3, 2, 7, -10, -8, 8, 3, 8, 12, 11, 4, -2, 2, 12, -2, 7, -9, -4, 5, -9, 11, 2, 5, 6, 7, -1, 8, -2, 11, 4, 4, 13, 4, -1, 3, -1, -3, -3, 8, -9, -10, -6, 1, 1, 7, 0, -4, 4, -3, -2, 5, 4, 13, 6, 2, 11, 0, 12, 6, 5, 3}
, {-3, 11, 3, 0, -4, 14, -2, 1, 8, 9, 12, -3, 1, -2, -8, -4, 2, -4, -2, -8, 0, -10, -9, 5, -3, -3, 5, 7, 8, 2, -7, -1, 2, 5, -1, 5, 2, -6, 15, -12, -5, 5, 1, 5, -3, -1, 7, 8, -4, -7, 3, 5, 6, 13, 13, 8, 8, 1, 4, 0, -1, -5, 5, 8}
}
, {{-8, -4, 13, -13, -6, 14, -6, 14, 2, -9, 10, 21, -4, -11, -5, -11, 12, -3, -9, -7, 1, 14, -19, -7, -15, -7, 16, -5, -4, 8, -3, 7, -5, 10, 8, 14, -10, -11, -13, 7, -3, 3, -6, -11, -10, 16, 1, 13, 16, 9, 6, 15, 23, 5, -16, 10, 7, 4, -8, -11, 0, 19, 3, -13}
, {-6, -7, -2, -3, 5, 0, 6, -6, -3, -10, 1, 19, 6, 9, -10, -14, -10, 8, -13, -1, 1, -5, -6, -5, -12, 5, 2, -14, -10, 15, -6, 3, 8, -10, -4, -8, 7, -14, 0, -2, 6, -8, -3, -10, 0, 1, -8, -6, -7, 3, 4, 7, -5, 4, 2, 10, 11, 6, -9, 5, 5, 9, -17, 1}
, {17, -6, -8, 17, 2, -9, -3, 9, -1, 16, -1, 2, -9, 6, -2, -16, -9, -4, -5, 0, -6, -13, 6, 8, -9, 1, 4, 6, -25, -18, 17, 6, 1, -2, -22, -22, 9, -16, 5, -9, 6, 10, 6, -20, 3, -2, -5, 0, -20, 0, 8, -3, -13, -3, -14, 4, -17, -3, -9, 14, -2, 14, 8, 1}
}
, {{-1, 9, -6, -1, -4, -8, 3, -1, -23, -1, -8, 12, -1, -8, -13, 4, 1, 6, -11, 7, -3, -3, 2, -3, 1, 2, -2, 0, 7, 1, 18, 8, -1, -2, 3, 0, 1, -4, -8, -1, -1, -7, -16, -16, -4, -13, -9, -17, -8, -6, -1, 0, 1, -3, -16, 0, 3, 7, 4, 2, 15, 6, -4, -11}
, {3, -1, -1, -3, -4, 9, -3, -10, 0, 4, -10, -2, 2, 5, -2, -4, -3, 1, 10, -9, 8, -1, 5, 0, 1, -1, -7, -1, 0, -1, -10, -5, -13, -3, 5, -2, -3, 6, 1, -3, -10, -5, -2, -10, -3, -10, -3, -1, 4, 0, -2, 0, 1, -3, -4, -11, -9, 7, 5, 0, -7, -8, -4, -10}
, {-15, -7, -7, -8, -6, -4, 0, -2, -10, 7, -12, 11, -8, -4, -6, -12, -7, -2, 1, 16, 6, 12, 5, 3, -1, -19, -8, -7, -1, -3, 5, 10, 5, 0, -14, -17, 5, -16, 1, 3, -9, 2, -2, 1, -6, -3, -2, -4, -13, 6, 9, 3, -3, 5, 1, -9, -9, 3, -10, 14, 9, 8, -16, -6}
}
, {{-10, -6, -2, 0, -7, 3, 5, 2, 3, 12, 2, -1, 1, 0, 3, 4, 1, -8, -5, -1, 8, -11, 1, 1, -6, 3, 6, 2, 3, 9, 5, -13, 4, -5, -12, -14, 5, -2, -9, -2, 3, 2, 2, 3, 2, 3, 2, 2, -5, -4, 0, -4, -2, 7, -5, 15, -6, 2, 0, 5, 3, -10, 4, 9}
, {1, 2, -2, 3, -1, 3, 3, 8, -5, 8, 4, -1, 4, -2, -5, 4, -9, -6, 2, 3, 11, 1, 8, 9, -3, -3, 3, -5, 2, -12, 7, 8, 9, -13, -8, 3, 17, -4, -1, -5, 9, 8, 4, -10, 14, 6, 17, 0, 7, -3, 0, 2, -7, 0, -5, -3, 11, -11, -4, 9, 0, -2, 5, 1}
, {-10, -9, -17, -6, 3, 6, -13, -8, -14, -8, 5, -1, -2, 1, -14, -5, 7, 8, -15, -5, -13, -3, 13, -11, 4, -17, 0, 3, 3, 3, -9, 6, 15, 3, 4, 17, 11, -17, 4, 4, 13, -6, -1, -3, -11, 0, 4, 7, -1, -4, 2, -8, 2, 13, -3, 4, 14, -20, 5, 9, 5, 2, -3, 0}
}
, {{3, 10, 13, -1, -10, 0, -9, 10, 3, 0, -2, 5, 4, -16, 0, -5, -4, -7, -5, 2, -1, 3, 2, 11, 3, -2, 15, -7, -4, -1, -6, -9, 9, 17, 13, 4, 7, -3, -8, 26, -11, 0, -5, -3, -9, 7, -1, 3, 0, -1, 20, -2, 3, 3, -2, 1, -11, -7, 8, 7, -7, -3, -1, 4}
, {-8, 4, 1, -7, -2, 5, -9, 6, 0, -8, 6, 10, 0, 1, -3, -3, 5, -1, -4, 0, 3, 1, -9, 6, -6, 0, 12, -8, 0, 4, -7, 1, 4, -1, 4, 4, 2, -3, -3, 4, -2, -5, -3, -13, -8, -7, -10, -2, 2, -4, 3, 4, 3, 2, 1, 0, -1, 3, -3, -19, 1, -2, -9, -4}
, {-4, 4, 15, 1, 3, 12, 0, -3, -3, 6, -7, -2, 3, 12, 3, 9, 12, 4, -1, 1, 2, -7, -9, 7, -9, 2, 6, -1, -6, -4, -7, -1, 11, -5, -2, 10, 3, 4, 4, -2, -6, -6, 1, -6, -16, 5, -10, 2, 10, -2, 14, 6, 4, 2, 5, 12, 2, 9, 2, 0, 12, -6, 0, 14}
}
, {{-1, -1, -5, 5, -12, -13, 3, 6, -1, 5, 8, -6, 8, 3, 6, 13, 4, 7, 3, 14, 6, -8, -4, -7, 5, -4, -1, -2, 10, 4, 5, -1, -8, -5, -5, -8, 0, 0, -7, -2, -1, 1, 2, 2, -7, 8, -2, -3, -7, 3, -24, -5, -4, 7, 5, 1, -4, 5, -12, -13, -2, -4, -5, -10}
, {2, -13, 4, 3, 0, -8, 3, 8, -9, 0, 0, 0, 11, -5, 15, 9, -11, 4, 11, 11, 15, -8, 4, 9, 3, -17, 5, 5, -9, 5, -6, -3, 6, 2, 4, 0, 0, -5, -5, -17, -17, 2, -3, -17, 2, 7, -13, 1, 0, -2, 0, 7, -6, -6, -2, 3, -7, 0, 2, 0, 2, -1, 9, -5}
, {-3, -5, -31, -10, 17, 20, -14, -17, -39, -26, -9, -4, -4, 0, -20, -20, -11, 10, -19, 8, -11, 2, 18, -21, 10, -43, -18, -3, 10, 17, -2, 14, 11, 5, -3, 10, 10, -6, -8, 8, 24, -2, -6, -18, -12, -26, 2, -10, 6, 5, -10, -15, 13, 10, -2, 31, 4, -41, 21, 17, 12, -11, 7, 21}
}
, {{12, 5, -5, -3, -4, -12, -2, 5, -1, -5, 5, 12, 9, -13, -10, 7, -16, -3, 3, 5, 1, 4, 6, 12, -11, 5, -2, -14, -6, -7, 6, -2, 2, -5, 6, -8, -5, -6, -4, 8, 1, 6, -1, -8, -13, -4, 2, -21, 7, 2, 2, -5, 0, -7, -13, -14, -7, 2, 4, -3, 2, 3, -6, -11}
, {-1, -4, 1, 1, 2, -5, -1, 2, 1, -1, -1, -3, 3, -3, -7, -2, -3, 2, 3, 1, -8, 9, 8, 5, -7, 1, 1, -5, -7, -8, -15, -6, -9, 1, 4, 1, 6, -9, 3, 1, 6, 7, -7, -7, 3, 5, 0, -3, 0, 5, -4, -10, 8, 1, -1, -4, 3, 9, -2, 9, -8, 5, -2, -2}
, {-2, -3, -3, 3, -9, -5, 3, 1, 8, -9, -3, -5, 0, -5, -6, 4, -12, -11, 10, 10, -7, 10, 5, -3, 0, -7, -8, 1, -1, -4, 5, 1, -21, -10, 5, 2, -1, -1, 0, -1, 4, -9, -11, -11, 3, -3, 6, -9, -2, -5, 0, -12, -9, -12, -14, -8, -2, -6, 0, 3, 1, -1, -18, -7}
}
, {{-6, 3, -6, -2, 1, 20, -5, -4, -7, 4, 0, 2, 2, 0, -9, -11, 2, 10, 7, -13, 12, -1, -1, 10, 7, -7, -1, 10, -2, 21, -3, -2, -12, -11, 10, -8, 2, 1, 11, 5, 5, -3, 5, 6, 6, -13, 15, 5, -1, -15, -1, -9, -5, 4, 14, 3, -1, -1, 22, -6, -1, -4, 7, 2}
, {-3, 2, 11, -8, -6, 19, 6, 7, 4, 5, -1, 7, 18, 0, -9, 15, 0, -10, 5, -11, 14, 10, -6, 15, -8, -6, 7, -10, -11, 3, -7, -4, -8, 5, 7, 12, -1, 7, 3, 10, 2, 17, -2, 0, -3, 13, 4, 8, 5, -5, -5, -12, 1, 1, 2, -11, -1, 11, 8, 4, -5, -5, -4, -1}
, {-8, -1, 3, 4, -6, 6, 4, -1, -4, -2, 11, 2, 6, 5, -3, 18, 9, 9, -2, -17, 10, -3, -22, 4, -8, 4, 0, -7, -4, 0, -6, 5, -4, 3, -4, -5, 8, 1, -2, -4, -14, 1, -9, -12, -2, 10, -6, 1, -6, -4, -12, -13, 4, 8, 8, 4, -2, 7, -3, -11, -11, 2, -4, -4}
}
, {{3, 4, 2, 2, 7, 12, -3, -4, -5, 2, 4, 6, -4, 5, -4, 14, 11, 2, -2, 1, -3, -3, 0, -1, 0, -3, 2, 0, 8, 1, -3, 1, 5, 4, 6, 8, 6, -1, -2, 0, -2, 9, -7, -14, -1, -5, 3, 2, 10, -2, -5, -7, -2, 10, -1, 8, 14, -1, 11, 4, 7, 2, 0, -4}
, {-1, 3, -5, -5, -8, 10, 7, -6, -9, 10, 8, -4, 5, 4, 5, 5, 9, -1, 1, -5, -9, 3, -1, -7, -11, -2, -6, 11, 3, 13, 6, 1, 3, 7, 16, 13, -4, 7, 1, 10, -6, 3, -8, 7, -7, 4, -4, 6, 7, -5, 5, 2, -3, 8, 8, 3, 1, 4, 2, 1, 8, -3, 5, 7}
, {-2, 2, -1, 2, 5, 2, 7, -1, -1, 12, 4, 9, -2, 10, -2, 6, 4, 3, -7, -12, -1, 5, -6, 2, -7, -1, -4, 3, 0, 5, 0, 7, 6, 2, 4, 12, -6, -2, -6, 1, -3, 8, 1, 6, -9, 7, 1, 4, 4, -7, 10, -5, -3, 12, -2, 16, 3, 7, -6, -2, 3, 2, 6, 4}
}
, {{5, 11, -2, 1, 4, -4, 7, 13, 12, 4, 10, 6, 4, 12, -14, 8, 1, -7, 4, -9, -6, 1, -6, 2, 2, 6, -1, 9, 2, -5, -6, 8, 7, 6, -6, -7, -2, -1, 26, 4, 6, 6, 5, 0, -9, -3, -1, -11, -3, -15, 12, 19, 3, 6, 8, -1, 0, 8, 18, -10, -5, -1, -6, 10}
, {5, 1, 5, 9, 9, 9, 4, 10, 6, 11, 10, 0, -9, 0, -1, -3, 1, -9, -5, -5, 6, 4, -7, 10, -4, 4, 14, -6, 7, 7, -7, 5, 7, 2, 0, 12, -1, 6, 2, 10, -8, 7, 14, -1, 2, 10, 9, 21, -1, -21, 10, 2, 2, 12, 4, -5, 4, 3, 11, 2, -5, 5, 7, 1}
, {-7, 3, 12, 3, -7, -4, 9, 5, 1, -8, -5, 3, 9, 7, -2, 4, -2, -2, 0, -10, 8, -7, -3, -2, 6, 8, 5, 4, -3, 2, -12, -9, 13, 1, 0, -5, 5, 4, -4, 9, -2, 11, -9, -2, -11, 0, 18, 5, -2, -16, 1, 6, 1, 4, -7, -3, 6, 13, -3, -1, -18, -13, -14, -1}
}
, {{-8, 4, -13, 0, 5, 1, 4, -10, -11, -8, -3, -10, 8, 8, 5, 8, 3, -13, -9, -7, 3, -4, -13, 0, 4, 0, 0, -10, 2, 3, -4, 14, 1, 2, -2, 4, 1, 7, -4, -7, -4, 0, 12, -10, -5, 7, 2, 1, 7, -4, 5, 1, 4, 7, 6, -1, -8, -2, 3, -3, 0, -11, -8, 11}
, {-12, 7, 1, -9, 5, 1, 15, 4, 3, 4, 3, -12, -7, 0, 4, 2, 14, -7, 1, -1, -9, 1, -10, 11, -3, -8, -1, 6, 3, -2, -6, 8, 7, -2, 0, 0, 2, -5, 7, -1, 10, -6, -5, -6, -12, 2, -4, -13, 2, -12, 2, 11, 9, 10, 0, 7, -7, -2, 3, 10, -5, 1, 0, 5}
, {-7, 10, -7, -12, -2, 3, 9, -5, -7, -7, 5, -13, 4, -5, -7, 5, 3, -10, -2, -7, -5, 8, -9, -6, 2, 4, -3, -12, 13, 8, -4, 3, 2, 0, 7, 9, 8, 7, 2, 14, 5, -2, 7, -10, -4, -9, 1, -12, 6, 3, 2, 4, -4, 3, 4, 9, -5, -9, 2, -2, 5, -7, -10, 5}
}
, {{-7, -6, -13, -3, -5, 26, -11, 14, -18, -2, 0, 3, -13, 3, -4, 5, 2, 22, 14, 3, -5, -6, -5, -3, 0, -14, -3, 0, -7, 1, -10, 5, 0, -16, 15, 26, 12, -9, -2, 1, -5, 8, 0, -6, -13, -4, 7, 10, 3, 3, -2, -22, 16, 9, -11, 17, -6, -18, -5, -21, 5, 16, -8, -2}
, {1, -3, -6, 9, -9, 11, -7, -7, -8, 5, -4, 2, 1, -3, -2, 2, -3, 7, -7, 2, 10, 2, 11, -16, -2, 2, -12, 7, -8, 2, 4, -5, -5, 0, 4, 6, 5, -2, 3, -4, 2, -8, -12, -4, -4, -18, -5, 3, 1, -2, 8, -11, -9, 6, 0, 0, -3, -11, 5, -7, 3, 0, -3, -5}
, {3, -1, -9, 0, -20, 11, -6, -12, 4, 8, -2, 15, 8, -6, 2, 7, -12, 9, 1, 5, 5, 0, -4, -6, -2, -13, -6, -23, -4, 6, -1, -7, -21, -1, 20, 0, -15, 9, -16, -6, -8, -14, -16, -3, 1, -3, -1, 3, -7, -4, 3, -9, -5, -5, -1, -5, -7, 7, -5, -17, -1, 8, -19, -6}
}
, {{-2, 1, 12, 5, 5, 3, 2, 3, -2, 3, 3, 9, 2, 11, -7, 14, 4, -7, 3, 1, 3, -9, 5, 9, 5, 14, 9, -9, 3, -5, 3, 18, 9, 10, 6, 0, 7, -3, 2, 6, -1, 6, 7, -7, -8, 8, -1, 8, 7, 1, 8, 7, 0, 3, -9, 0, -17, 10, 6, 12, 8, 12, -4, 8}
, {-2, 7, -1, 6, 5, 4, 0, 3, 12, -3, 5, -6, -1, 12, 1, -7, -2, -10, -3, -5, 0, -9, 4, -2, 4, 7, 2, 3, 8, 7, -9, 3, 17, 3, -4, -1, -2, 9, 7, 11, -10, -2, 5, -1, -5, 6, -3, 0, 5, -5, 5, 3, 2, -3, 8, 8, -6, 11, 2, 8, 10, 4, 2, 11}
, {-10, 6, 9, 11, 6, 10, 1, 4, 18, 4, 2, -2, 15, -5, -1, 2, 1, -9, 4, -2, 15, -6, 12, 9, 7, 1, 0, 0, 4, 8, -10, 17, 17, 3, 11, 10, -11, 17, 2, 16, -12, -3, 2, -10, -3, 7, 3, -1, 4, -8, 10, 1, 6, 4, 10, -3, -13, 18, 10, 10, 8, -2, 2, -2}
}
, {{7, -7, -9, 3, 4, -10, -4, 12, -2, -6, 8, 1, -4, -3, -12, 5, -1, -11, 7, 5, 3, -8, 19, -4, -3, 2, -5, -14, 1, -6, 2, 0, 0, -1, -5, 1, 2, 2, -1, 2, -15, 10, -5, 2, -13, -3, -4, -6, 10, 5, -11, -4, -3, -2, 1, -1, -6, 2, -12, -15, -13, 17, -10, -5}
, {3, -7, -8, -1, 0, -20, 9, -10, 3, -4, 3, 0, 3, -5, 2, -3, -1, -2, 4, 10, -2, -6, 7, 5, -1, -3, -5, -10, 1, 1, 5, -1, 8, -5, -2, 0, 11, 3, 2, -8, -9, -12, 5, -3, -5, 4, -8, 0, -5, 1, -2, -11, -3, -8, 15, -10, -4, 7, -12, 4, -9, -17, -1, -8}
, {6, -6, -2, 3, 9, -7, -5, -6, 0, 11, 9, 10, -1, -3, 7, -9, -13, 16, -7, -4, -4, -8, 11, 2, -2, -4, -2, 8, -13, -8, 2, -2, 12, -2, -12, -1, 3, -5, -11, -10, -14, -9, 0, -3, -2, -1, -15, 17, -19, 3, -11, -7, -8, -4, 7, -6, -3, 9, -12, 4, -6, 0, 5, 6}
}
, {{-11, -1, -5, 1, 0, 4, -2, -7, -13, -4, 9, -1, -8, 5, -13, 5, 6, 0, 2, 1, -4, 2, 0, 9, 2, -1, -5, -6, -7, -16, -9, 5, -8, -8, -2, -3, 5, 10, 8, -6, -9, -3, -2, -6, 0, -8, -1, 6, 0, -2, -9, 2, -7, -7, -6, 0, -1, -15, 6, -11, 1, 2, -9, -2}
, {-8, 3, 1, -12, 4, 4, 4, -9, -7, 2, 8, 0, 7, 3, -5, -12, 4, -6, -3, -10, -1, -2, -3, 7, 2, -7, -3, -4, 4, -2, -1, 12, 0, -5, 1, 3, -3, -1, 7, 6, -3, 2, -3, -5, 2, -1, -7, -6, 1, -5, 13, 7, 9, -3, 2, -1, -6, -3, 3, 0, -8, -8, 1, -9}
, {-4, 12, 15, 5, 6, -2, 14, -7, 3, 11, 11, 1, 5, 7, -2, -8, 7, -8, -7, 1, -1, 0, 3, 11, -13, 18, 16, 16, 2, -3, -6, 4, 4, 3, 3, -1, 8, 0, 3, 13, -3, 11, 24, -10, 7, 13, 1, -2, 3, -16, 19, 1, 10, -1, 7, -5, 3, 21, 3, 14, 1, -4, -2, 6}
}
, {{-7, 6, 9, 3, -5, 1, -4, -3, 9, -10, -10, -13, 3, 8, 19, -5, -4, 2, -5, -15, -17, -15, -1, -9, -9, 0, -12, 2, -5, 1, 1, 1, 7, 0, -7, 0, -7, 0, -3, 0, 1, 0, -5, -4, 9, 5, -13, 9, -8, -1, -9, -8, -8, -9, -11, 2, -3, -8, -1, 0, -9, -20, -7, -2}
, {1, -5, -2, -6, -8, 8, -3, -2, -10, 4, 5, -9, 0, -9, -6, -4, -1, 10, 2, -4, -14, -9, 4, 3, -5, 8, 0, -4, -4, -8, 3, 0, 4, 3, -3, -4, -2, -3, 6, 5, 13, 1, -11, -6, -7, -4, -6, -2, 1, 12, 9, -7, -10, -8, -6, 8, -1, -6, 3, -1, 13, -6, -8, -4}
, {-3, -4, 5, -1, 2, -6, 4, 3, 4, -5, -4, -8, -5, -6, 6, -11, -11, -1, -1, 1, 2, -4, 6, -10, -2, 15, -2, -9, -9, -4, 4, -1, -2, -8, -9, 3, -7, -6, 11, -3, 5, -6, -6, -13, 4, 2, -11, -7, 4, 2, -12, 4, -4, 5, 0, -5, -2, 12, -3, 1, -1, -6, -1, 0}
}
, {{4, -7, -36, -2, 1, 35, -14, -16, -30, -25, -17, -1, -1, -6, -16, 6, -1, 9, 1, 0, 1, 6, 3, -2, 14, -13, -22, 3, -6, 15, 5, 5, 11, -12, 31, -12, -5, 2, 5, -8, 13, -21, 12, -19, 0, -23, -1, -33, -3, 17, -5, -13, -11, -4, 9, 15, -1, -19, -6, -1, 11, 5, -6, 6}
, {7, -10, -7, 0, 0, -9, 9, 11, -2, -14, -1, -2, 5, 2, -7, 0, -4, -8, -7, 6, 7, -10, -4, 4, 0, 8, 0, -8, 6, -5, -12, 6, -13, 10, 7, -12, -6, 2, -10, 0, 5, 1, 0, -4, -2, -1, -4, 0, -1, 3, -9, -2, 4, 4, 4, 0, 1, -3, -17, -8, -4, 12, -5, -7}
, {3, 2, -16, -9, -6, -2, -2, -4, -5, 4, 3, 12, -4, 1, -13, 10, -4, 18, -5, 7, -5, 0, -8, -2, 2, -17, -17, -6, 0, 5, 5, 10, 7, -7, -5, -7, 14, 0, -15, -4, 14, 6, -12, -2, -7, 4, -9, -23, -16, 10, 1, -4, 5, 13, 1, 8, -15, -9, -6, -2, 10, 7, 8, -6}
}
, {{3, -9, -14, -13, -8, 12, -7, 12, -11, -7, 8, 11, -5, -11, 4, -17, 8, 14, -7, -1, -1, 7, 3, 6, -12, 6, -7, -2, -2, -10, 13, 7, -6, -3, -1, 9, -1, -13, 1, 5, 7, -2, -1, -7, -3, 10, 5, 2, 8, 7, -2, -6, 10, -3, -6, 4, -9, -5, 0, -7, -1, 19, -11, 2}
, {-5, 7, -9, 3, -1, 3, -3, -3, -4, 9, 7, 0, -6, 5, 1, -6, -3, 0, -8, 7, 3, 10, 8, -3, -2, 1, -2, -1, -7, -6, 11, 2, -2, 1, -9, -11, 2, -11, 2, -2, 5, -1, -5, 5, 3, -3, -9, -9, -12, 3, -6, -4, -12, -12, 4, 7, -6, 1, -1, -1, -6, -12, -3, 3}
, {0, -2, 3, 7, -3, 3, -5, 0, -5, 15, -9, 1, -5, -1, 2, -4, -10, -10, 2, 11, -3, -5, 15, 4, -5, -3, 12, 6, -14, -18, 6, 5, -11, 1, 0, -12, -9, 8, 2, -3, -7, -9, 11, -5, 18, -6, -1, 0, -9, 0, -3, -3, -3, -18, -8, -18, -8, -4, 9, 5, -3, -7, -2, 8}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_3_H_
#define _BATCH_NORMALIZATION_3_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       2

typedef int8_t batch_normalization_3_output_type[2][64];

#if 0
void batch_normalization_3(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_3_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_3_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_3.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       2
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 6
#define INPUT_SCALE_FACTOR 6
#define OUTPUT_SCALE_FACTOR 6
#define NUMBER_T int8_t
#define LONG_NUMBER_T int16_t


static inline void batch_normalization_3(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_3_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = scale(NUMBER_T, (LONG_NUMBER_T)bias[z], -INPUT_SCALE_FACTOR);
      tmp += (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      tmp = scale(NUMBER_T, tmp, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[x][z] = clamp_to(NUMBER_T, tmp);
#elif defined(ACTIVATION_RELU)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
        tmp = scale(NUMBER_T, tmp, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
        output[x][z] = clamp_to(NUMBER_T, tmp);
      }
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int8_t batch_normalization_3_bias[64] = {-36, -17, -67, -25, -10, -22, -56, -58, -22, -14, -27, -25, -37, -28, -10, -71, -14, -45, -8, -79, 2, -65, -17, -38, -45, -73, -57, -60, -37, -60, -78, -57, -50, -13, -97, -72, -63, -19, -29, -61, -66, -29, -8, -44, -23, -21, -66, -45, 1, -45, -68, -19, -9, -62, -70, -68, -63, -21, -64, -14, -65, -26, 3, -23}
;
const int8_t batch_normalization_3_kernel[64] = {22, 25, 15, 30, 54, 51, 24, 13, 22, 40, 28, 48, 32, 18, 54, 15, 71, 47, 36, 17, 28, 31, 60, 24, 20, 13, 16, 13, 24, 22, 15, 13, 23, 58, 15, 18, 16, 32, 25, 11, 20, 50, 40, 18, 26, 51, 14, 27, 70, 14, 17, 32, 78, 14, 12, 13, 13, 46, 9, 29, 15, 47, 65, 60}
;
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_3_H_
#define _MAX_POOLING1D_3_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   2
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int8_t max_pooling1d_3_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_3(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_3_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_3.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   2
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 6
#define OUTPUT_SCALE_FACTOR 6
#define NUMBER_T int8_t
#define LONG_NUMBER_T int16_t


static inline void max_pooling1d_3(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef WITH_CMSIS_NN
// Not really CMSIS-NN since using arm_relu_q* is not more efficient, but use SSAT anyway
#if ACC_SCALE_FACTOR - OUTPUT_SCALE_FACTOR > 0
      output[pos_x][k] = __SSAT(max[k] >> (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#else
      output[pos_x][k] = __SSAT(max[k] << (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#endif
#else
      max[k] = scale(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, max[k]);
#endif
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    flatten.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _FLATTEN_H_
#define _FLATTEN_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define OUTPUT_DIM 64

typedef int8_t flatten_output_type[OUTPUT_DIM];

#if 0
void flatten(
  const number_t input[1][64], 			      // IN
	number_t output[OUTPUT_DIM]); 			                // OUT
#endif

#undef OUTPUT_DIM

#endif//_FLATTEN_H_
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0.0
  * @date    26 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "flatten.h"
#include "number.h"
#endif

#define OUTPUT_DIM 64

#define NUMBER_T int8_t
#define LONG_NUMBER_T int16_t

static inline void flatten(
  const NUMBER_T input[1][64], 			      // IN
	NUMBER_T output[OUTPUT_DIM]) {			                // OUT

  NUMBER_T *input_flat = (NUMBER_T *)input;

  // Copy data from input to output only if input and output don't point to the same memory address already
  if (input_flat != output) {
    for (size_t i = 0; i < OUTPUT_DIM; i++) {
      output[i] = input_flat[i];
    }
  }
}

#undef OUTPUT_DIM
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _DENSE_H_
#define _DENSE_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 64
#define FC_UNITS 16

typedef int8_t dense_output_type[FC_UNITS];

#if 0
void dense(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 64
#define FC_UNITS 16
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 6
#define INPUT_SCALE_FACTOR 6
#define OUTPUT_SCALE_FACTOR 6
#define NUMBER_T int8_t
#define LONG_NUMBER_T int16_t


static inline void dense(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const NUMBER_T bias[FC_UNITS],			              // IN

	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 

    output_acc = scale(NUMBER_T, (LONG_NUMBER_T)bias[k], -INPUT_SCALE_FACTOR);

    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
    output[k] = clamp_to(NUMBER_T, output_acc);
#elif defined(ACTIVATION_RELU)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
      output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[k] = clamp_to(NUMBER_T, output_acc);
    }
#endif
  }
#else


  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q7(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q7(
#endif
                             (q7_t*)input,
                             (q7_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q7_t*)bias,
                             (q7_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q7((q7_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q7((q7_t*)output, FC_UNITS);
#endif
#endif


#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 64
#define FC_UNITS 16


const int8_t dense_bias[FC_UNITS] = {-24, 2, 0, -20, -9, -10, 0, 0, 1, -19, -3, -18, -20, -13, -9, -9}
;

const int8_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{-6, -4, -15, -6, -9, 10, -2, 7, 6, -8, -8, 3, 5, -11, -9, -3, -3, 3, 13, -4, -8, -4, -1, -11, -1, 9, 1, -18, -5, -18, 1, 0, 10, -10, 2, -7, -10, -1, 9, 12, 3, -6, -21, -16, -11, -5, 13, 5, -16, -18, -5, -6, 8, -20, 8, 9, 19, 2, -2, -11, 12, 1, -13, 12}
, {10, -17, 22, 1, -1, -5, 8, 12, -8, 0, -16, 1, -19, 0, -6, 3, 0, 2, -1, -12, -5, -7, -15, -8, -7, -8, -5, -14, -5, 20, -10, 18, -16, -2, -6, -26, -2, -12, -9, -3, -4, -5, 1, 9, 5, 0, 13, -7, -3, -2, -10, -45, -12, -4, -19, 17, 10, -7, -10, -8, 5, 5, -27, -2}
, {0, -10, -2, -11, -4, -16, 17, -11, -24, 8, 13, 1, -27, 11, 2, 2, -9, 4, -4, 6, -18, 17, 4, -30, -12, -5, -12, 18, -2, -7, -5, -11, -23, 12, 6, -20, -16, -29, -4, 10, -13, -3, -7, -15, 1, -1, -18, -7, -7, -17, -4, -31, 5, -7, 1, 29, 10, -23, 13, -31, -13, -2, -32, -11}
, {1, 4, -6, -17, 8, 8, -18, -14, 4, 9, 8, 5, 10, -10, -15, -18, 4, 25, -7, -6, -8, -6, 12, 9, 9, 8, -25, 11, -7, -10, 10, -2, -8, -21, 0, 21, -17, -10, 4, -1, -4, -16, -11, -16, 5, 19, 4, -6, -7, -15, 17, -10, -21, 16, 18, 0, -2, -18, -3, -14, 15, 2, -4, -15}
, {8, -24, 9, -19, -19, 18, 20, -7, -11, 3, -21, 10, -28, 10, 2, 6, -13, 9, -12, 8, -15, 0, 18, -14, 9, 3, -18, 0, 2, 23, 13, -19, -26, 6, -3, 10, -21, -28, -6, 5, -18, -19, -9, -4, 7, -7, -25, 13, -16, 21, 5, 30, -14, 5, -4, 2, 4, -7, -9, 4, 7, 4, -28, -13}
, {-18, -9, 19, 5, -10, 4, 15, 9, -11, 21, -27, 7, -21, -15, -12, 1, -1, -34, -4, -25, 7, 6, -3, -7, -1, -17, 15, -21, -5, 1, 0, 19, -2, -8, 5, -1, 9, -20, -10, -22, 4, -26, -21, 2, -3, 2, 3, -8, -2, 18, -28, -3, -21, 3, -1, 6, 12, 20, -23, 0, -23, -6, 0, -7}
, {9, 9, -11, -15, 4, 0, -18, -18, 15, 0, -2, -4, 2, -55, -11, 21, -5, 0, 1, -5, -7, 2, 1, 13, 12, -30, -14, 14, 9, 0, 17, -1, -6, -8, 24, 1, -12, 0, -40, -15, 3, -27, -8, 6, 8, 10, -7, -6, -1, 2, 11, -3, -8, 15, -23, 6, 3, -20, -22, -10, 6, 8, 0, 0}
, {2, -18, -3, 8, -9, 0, -2, -18, -1, 4, 3, 3, 13, -1, 1, 3, -3, 2, -19, 1, 9, -6, -3, -12, -3, -16, -17, -21, 0, -16, 22, -11, 25, -7, -14, 19, -16, 10, 3, 3, 24, 12, 2, 25, -12, -4, 8, 6, -13, 21, 4, -5, -1, 21, 16, -38, 1, 1, -27, -2, 5, 8, 2, 5}
, {-1, -30, 3, 4, 4, -4, 16, 27, -4, 7, -14, -18, -11, -7, 1, -1, -7, 5, -18, 0, 2, -12, -16, 1, -12, 13, 23, 5, 2, 20, -24, 4, 2, 7, -26, -24, 0, -19, 4, -3, -8, 0, 3, 17, -4, -4, -17, 2, -13, -5, -4, -32, -11, -14, -19, 7, 13, 6, 14, 6, -11, -8, -19, 8}
, {-6, -8, -6, 7, -7, -9, 3, -2, -19, 8, -7, -4, 10, 4, -4, -15, -6, 2, -4, -2, -33, 12, 3, -3, 9, -13, -12, -8, -10, -1, 8, 10, 7, 11, -8, -3, 6, 17, 3, 3, 7, 12, -12, -24, -1, 11, 5, 0, 6, -10, 11, 0, -9, 1, 4, -19, 12, 9, 8, 6, 9, 0, -3, 15}
, {-4, -11, -27, 7, -2, -9, -25, -3, 8, -6, 9, 2, 19, 7, -13, -2, -7, -20, -16, -7, -3, -1, -7, 17, -25, -8, 3, 0, -5, -33, 30, 4, -4, -6, 5, 15, -12, -2, 6, 7, 3, -1, -6, 4, -22, 0, 21, 0, -7, -13, 22, -5, 1, 4, -22, -7, -4, -3, -19, -24, 0, 13, -25, -4}
, {-2, -8, 10, -12, -3, 9, 13, 8, -9, -2, -8, 12, 3, -1, -4, 4, -8, 10, -8, 2, -21, 16, 7, -13, 17, 2, 8, -10, -8, 14, -8, 9, 5, -9, 0, 10, -9, 8, 4, -8, 3, 0, -23, -16, -2, 5, -5, 7, -13, 0, 10, 3, 4, -2, 6, -18, 6, 6, 5, 3, 10, 12, -3, -4}
, {-13, -31, -17, 9, 1, -11, -4, -9, -7, -10, -12, 14, -4, -14, 7, -4, -16, 13, -11, 16, 0, -28, -8, -1, 15, -15, 11, -19, -33, -12, -2, 7, 23, 6, 1, 15, -6, -15, 24, 8, 15, -12, 5, -10, -28, -13, -18, 26, -15, 2, 9, -7, -8, -6, 6, -18, 13, -7, 9, -7, 11, 8, -20, 19}
, {-41, -2, -11, 14, -10, -9, 0, 13, -18, 0, -4, 6, -5, -13, -11, -21, -17, -21, -2, -6, -20, -5, 4, -33, 1, -5, -3, 0, -29, 2, -4, 12, 13, -19, 11, -2, -27, -10, -1, -3, 4, -15, -9, -15, -14, 2, -7, 2, 2, 0, 8, -12, -17, 17, 6, -4, -2, 8, 7, -5, -30, 5, 3, -4}
, {-32, 2, -8, -26, -14, -10, -17, 7, -18, -6, 7, 15, 6, -46, 9, -4, -10, -6, -7, 19, -22, -4, -8, -8, -10, -4, 6, 11, -20, -13, 10, -2, 3, -10, -11, -8, -21, -16, -21, -19, -20, 6, -28, 4, -8, -12, 18, -15, 0, 6, -11, 5, -16, 6, -1, -3, -6, -14, -6, -2, -4, 15, -1, -4}
, {-12, 20, -5, -6, 5, 3, -11, -7, -8, -10, 1, 5, 0, 3, -1, -18, 19, -36, -7, -12, -15, 12, 17, -2, -19, -6, -2, 16, -9, -18, 17, 3, -21, -12, -6, 5, 8, -17, 1, -1, -17, 3, 4, -14, -9, -9, 1, -24, -10, -13, -8, -18, -22, 0, 5, 3, 3, -21, 23, 2, 11, 6, -29, -19}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _DENSE_1_H_
#define _DENSE_1_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 16
#define FC_UNITS 1

typedef int8_t dense_1_output_type[FC_UNITS];

#if 0
void dense_1(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_1_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense_1.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 16
#define FC_UNITS 1
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 6
#define INPUT_SCALE_FACTOR 6
#define OUTPUT_SCALE_FACTOR 6
#define NUMBER_T int8_t
#define LONG_NUMBER_T int16_t


static inline void dense_1(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const NUMBER_T bias[FC_UNITS],			              // IN

	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 

    output_acc = scale(NUMBER_T, (LONG_NUMBER_T)bias[k], -INPUT_SCALE_FACTOR);

    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
    output[k] = clamp_to(NUMBER_T, output_acc);
#elif defined(ACTIVATION_RELU)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
      output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[k] = clamp_to(NUMBER_T, output_acc);
    }
#endif
  }
#else


  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q7(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q7(
#endif
                             (q7_t*)input,
                             (q7_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q7_t*)bias,
                             (q7_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q7((q7_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q7((q7_t*)output, FC_UNITS);
#endif
#endif


#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 16
#define FC_UNITS 1


const int8_t dense_1_bias[FC_UNITS] = {0}
;

const int8_t dense_1_kernel[FC_UNITS][INPUT_SAMPLES] = {{-3, -4, -7, 3, 4, 4, 2, -2, -3, 1, -3, -3, -5, 4, 4, -6}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"

 // InputLayer is excluded
#include "conv1d.h" // InputLayer is excluded
#include "batch_normalization.h" // InputLayer is excluded
#include "max_pooling1d.h" // InputLayer is excluded
#include "conv1d_1.h" // InputLayer is excluded
#include "batch_normalization_1.h" // InputLayer is excluded
#include "max_pooling1d_1.h" // InputLayer is excluded
#include "conv1d_2.h" // InputLayer is excluded
#include "batch_normalization_2.h" // InputLayer is excluded
#include "max_pooling1d_2.h" // InputLayer is excluded
#include "conv1d_3.h" // InputLayer is excluded
#include "batch_normalization_3.h" // InputLayer is excluded
#include "max_pooling1d_3.h" // InputLayer is excluded
#include "flatten.h" // InputLayer is excluded
#include "dense.h" // InputLayer is excluded
#include "dense_1.h"
#endif


#define MODEL_INPUT_DIM_0 48
#define MODEL_INPUT_DIM_1 1
#define MODEL_INPUT_DIMS 48 * 1

#define MODEL_OUTPUT_SAMPLES 1

#define MODEL_INPUT_SCALE_FACTOR 6 // scale factor of InputLayer
#define MODEL_INPUT_NUMBER_T int8_t
#define MODEL_INPUT_LONG_NUMBER_T int16_t

// node 0 is InputLayer so use its output shape as input shape of the model
// typedef  input_t[48][1];
typedef int8_t input_t[48][1];
typedef dense_1_output_type output_t;


void cnn1(
  const input_t input,
  output_t output);

void reset(void);

#endif//__MODEL_H__


#ifdef __cplusplus
} // extern "C"
#endif
/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"
// #include <chrono>

 // InputLayer is excluded
#include "conv1d.c"
#include "weights/conv1d.c" // InputLayer is excluded
#include "batch_normalization.c"
#include "weights/batch_normalization.c" // InputLayer is excluded
#include "max_pooling1d.c" // InputLayer is excluded
#include "conv1d_1.c"
#include "weights/conv1d_1.c" // InputLayer is excluded
#include "batch_normalization_1.c"
#include "weights/batch_normalization_1.c" // InputLayer is excluded
#include "max_pooling1d_1.c" // InputLayer is excluded
#include "conv1d_2.c"
#include "weights/conv1d_2.c" // InputLayer is excluded
#include "batch_normalization_2.c"
#include "weights/batch_normalization_2.c" // InputLayer is excluded
#include "max_pooling1d_2.c" // InputLayer is excluded
#include "conv1d_3.c"
#include "weights/conv1d_3.c" // InputLayer is excluded
#include "batch_normalization_3.c"
#include "weights/batch_normalization_3.c" // InputLayer is excluded
#include "max_pooling1d_3.c" // InputLayer is excluded
#include "flatten.c" // InputLayer is excluded
#include "dense.c"
#include "weights/dense.c" // InputLayer is excluded
#include "dense_1.c"
#include "weights/dense_1.c"
#endif


void cnn1(
  const input_t input,
  dense_1_output_type dense_1_output) {
  
  // Output array allocation
  static union {
    conv1d_output_type conv1d_output;
    max_pooling1d_output_type max_pooling1d_output;
    batch_normalization_1_output_type batch_normalization_1_output;
    conv1d_2_output_type conv1d_2_output;
    max_pooling1d_2_output_type max_pooling1d_2_output;
    batch_normalization_3_output_type batch_normalization_3_output;
    dense_output_type dense_output;
  } activations1;

  static union {
    batch_normalization_output_type batch_normalization_output;
    conv1d_1_output_type conv1d_1_output;
    max_pooling1d_1_output_type max_pooling1d_1_output;
    batch_normalization_2_output_type batch_normalization_2_output;
    conv1d_3_output_type conv1d_3_output;
    max_pooling1d_3_output_type max_pooling1d_3_output;
    flatten_output_type flatten_output;
  } activations2;


// Model layers call chain 
  
  
  conv1d( // First layer uses input passed as model parameter
    input,
    conv1d_kernel,
    conv1d_bias,
    activations1.conv1d_output
    );
  
  
  batch_normalization(
    activations1.conv1d_output,
    batch_normalization_kernel,
    batch_normalization_bias,
    activations2.batch_normalization_output
    );
  
  
  max_pooling1d(
    activations2.batch_normalization_output,
    activations1.max_pooling1d_output
    );
  
  
  conv1d_1(
    activations1.max_pooling1d_output,
    conv1d_1_kernel,
    conv1d_1_bias,
    activations2.conv1d_1_output
    );
  
  
  batch_normalization_1(
    activations2.conv1d_1_output,
    batch_normalization_1_kernel,
    batch_normalization_1_bias,
    activations1.batch_normalization_1_output
    );
  
  
  max_pooling1d_1(
    activations1.batch_normalization_1_output,
    activations2.max_pooling1d_1_output
    );
  
  
  conv1d_2(
    activations2.max_pooling1d_1_output,
    conv1d_2_kernel,
    conv1d_2_bias,
    activations1.conv1d_2_output
    );
  
  
  batch_normalization_2(
    activations1.conv1d_2_output,
    batch_normalization_2_kernel,
    batch_normalization_2_bias,
    activations2.batch_normalization_2_output
    );
  
  
  max_pooling1d_2(
    activations2.batch_normalization_2_output,
    activations1.max_pooling1d_2_output
    );
  
  
  conv1d_3(
    activations1.max_pooling1d_2_output,
    conv1d_3_kernel,
    conv1d_3_bias,
    activations2.conv1d_3_output
    );
  
  
  batch_normalization_3(
    activations2.conv1d_3_output,
    batch_normalization_3_kernel,
    batch_normalization_3_bias,
    activations1.batch_normalization_3_output
    );
  
  
  max_pooling1d_3(
    activations1.batch_normalization_3_output,
    activations2.max_pooling1d_3_output
    );
  
  
  flatten(
    activations2.max_pooling1d_3_output,
    activations2.flatten_output
    );
  
  
  dense(
    activations2.flatten_output,
    dense_kernel,
    dense_bias,
    activations1.dense_output
    );
  
  
  dense_1(
    activations1.dense_output,
    dense_1_kernel,
    dense_1_bias,// Last layer uses output passed as model parameter
    dense_1_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif
