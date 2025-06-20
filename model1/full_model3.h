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

#define NUMBER_MIN_FLOAT -2147483648
#define NUMBER_MAX_FLOAT 2147483647

static inline float min_float(
    float a,
    float b) {
	if (a <= b)
		return a;
	return b;
}

static inline float max_float(
    float a,
    float b) {
	if (a >= b)
		return a;
	return b;
}

static inline float scale_number_t_float(
  float number, int scale_factor) {
	return number;
}
static inline float clamp_to_number_t_float(
  float number) {
	return (float) number;
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

typedef float conv1d_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

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
#define WEIGHTS_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define NUMBER_T float
#define LONG_NUMBER_T float


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


#error "Data type unsupported by CMSIS-NN"

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


const float  conv1d_bias[CONV_FILTERS] = {-0x1.8450dc0000000p-4, -0x1.7cacea0000000p-3, -0x1.2d1ab20000000p-5, 0x1.0ea2e40000000p-1, -0x1.f5f2d40000000p-4, 0x1.14d8bc0000000p-2, 0x1.3a403a0000000p-2, 0x1.9193240000000p-3, -0x1.3e8d9e0000000p-2, 0x1.593d860000000p-3, 0x1.2ee0be0000000p-1, -0x1.0da75c0000000p-3, 0x1.4c867a0000000p-5, 0x1.50f4880000000p-2, 0x1.6dae860000000p-2, 0x1.2e3c180000000p-6}
;

const float  conv1d_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS] = {{{0x1.1081ec0000000p-3}
, {-0x1.328f920000000p-6}
, {0x1.43567a0000000p-2}
}
, {{-0x1.4d568c0000000p-2}
, {-0x1.b18bb80000000p-3}
, {-0x1.22aade0000000p-2}
}
, {{-0x1.1d3f7c0000000p-3}
, {0x1.9d94c80000000p-3}
, {0x1.e002c20000000p-4}
}
, {{0x1.a285040000000p-3}
, {0x1.0a1f980000000p-3}
, {-0x1.7230900000000p-4}
}
, {{0x1.c4ba480000000p-4}
, {0x1.fe4b820000000p-3}
, {-0x1.b8fbcc0000000p-3}
}
, {{-0x1.4c2ab40000000p-2}
, {0x1.2f0c3c0000000p-2}
, {-0x1.96d4ec0000000p-5}
}
, {{0x1.31498a0000000p-2}
, {-0x1.496e940000000p-3}
, {0x1.9dcd2e0000000p-4}
}
, {{-0x1.620af40000000p-7}
, {0x1.136f060000000p-2}
, {-0x1.783dba0000000p-2}
}
, {{0x1.65c1040000000p-2}
, {0x1.ff2f4a0000000p-3}
, {0x1.20fff80000000p-3}
}
, {{-0x1.552f8a0000000p-3}
, {-0x1.a744f00000000p-3}
, {0x1.0961aa0000000p-5}
}
, {{0x1.7847f60000000p-3}
, {0x1.99c6d40000000p-3}
, {0x1.04d1260000000p-4}
}
, {{-0x1.257e6e0000000p-2}
, {0x1.5d36d40000000p-4}
, {0x1.4bc2ca0000000p-2}
}
, {{0x1.ae26460000000p-2}
, {-0x1.3cd5020000000p-9}
, {-0x1.3ba4ac0000000p-2}
}
, {{-0x1.edd1780000000p-3}
, {-0x1.54094c0000000p-3}
, {0x1.3aa2940000000p-2}
}
, {{0x1.22182a0000000p-2}
, {0x1.5795420000000p-3}
, {0x1.c6d7480000000p-2}
}
, {{0x1.f415d80000000p-4}
, {-0x1.6eedfe0000000p-2}
, {0x1.2fbcfc0000000p-2}
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

typedef float batch_normalization_output_type[46][16];

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
#define WEIGHTS_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define NUMBER_T float
#define LONG_NUMBER_T float


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

const float batch_normalization_bias[16] = {-0x1.348f880000000p-2, -0x1.f606ee0000000p-2, -0x1.623ab80000000p-2, -0x1.13ce480000000p-1, -0x1.429bbc0000000p-3, -0x1.0e190a0000000p-1, -0x1.7a54b40000000p-3, -0x1.41f8900000000p-1, -0x1.0486820000000p-2, -0x1.af2c420000000p-2, -0x1.815a5c0000000p-2, -0x1.122e300000000p-2, -0x1.121afa0000000p-2, -0x1.b13ad60000000p-2, -0x1.790a7c0000000p-2, -0x1.2aa4a80000000p-4}
;
const float batch_normalization_kernel[16] = {0x1.39fe180000000p-1, 0x1.2b21000000000p-2, 0x1.d73f200000000p-1, 0x1.8642fc0000000p-1, 0x1.1689900000000p+0, 0x1.adaabc0000000p-1, 0x1.793a980000000p-1, 0x1.3e5d9c0000000p+0, 0x1.58c5860000000p-2, 0x1.51cd800000000p-1, 0x1.028e360000000p-1, 0x1.04ebd80000000p+0, 0x1.b415c60000000p-1, 0x1.c210120000000p-1, 0x1.6517460000000p-2, 0x1.b3c5de0000000p-1}
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

typedef float max_pooling1d_output_type[POOL_LENGTH][INPUT_CHANNELS];

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
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define NUMBER_T float
#define LONG_NUMBER_T float


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

typedef float conv1d_1_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

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
#define WEIGHTS_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define NUMBER_T float
#define LONG_NUMBER_T float


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


#error "Data type unsupported by CMSIS-NN"

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


const float  conv1d_1_bias[CONV_FILTERS] = {0x1.4eccda0000000p-3, 0x1.0203e20000000p-3, 0x1.3d45540000000p-5, -0x1.b76e520000000p-3, 0x1.1883ca0000000p-4, -0x1.484e940000000p-3, 0x1.34687c0000000p-3, -0x1.6702ba0000000p-6, -0x1.2ef4b20000000p-4, -0x1.d367680000000p-4, -0x1.93a2580000000p-7, -0x1.5e0b5a0000000p-4, 0x1.0fafae0000000p-6, 0x1.5fa5220000000p-5, 0x1.417fc00000000p-4, -0x1.da1ec60000000p-6, -0x1.1dabc00000000p-3, 0x1.5b84860000000p-2, 0x1.15a98a0000000p-5, -0x1.48c5f40000000p-2, 0x1.78eaba0000000p-2, -0x1.19f1240000000p-3, 0x1.ec1fea0000000p-6, -0x1.9fdba20000000p-3, 0x1.6dbca20000000p-4, -0x1.0976300000000p-3, 0x1.b963420000000p-4, -0x1.3ad15c0000000p-4, 0x1.64916e0000000p-8, -0x1.acd3820000000p-3, -0x1.cc62960000000p-4, 0x1.26991a0000000p-5}
;

const float  conv1d_1_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS] = {{{0x1.36003e0000000p-5, -0x1.a79d160000000p-3, -0x1.ca1b7a0000000p-5, 0x1.31dbc80000000p-4, -0x1.4cb3dc0000000p-4, 0x1.3607260000000p-5, 0x1.5319280000000p-3, 0x1.a7ba1c0000000p-5, -0x1.0938e20000000p-7, -0x1.e27c0e0000000p-4, 0x1.03b1c20000000p-3, 0x1.286e660000000p-5, 0x1.258e4a0000000p-3, 0x1.01d8ca0000000p-3, -0x1.920eee0000000p-3, -0x1.dd27540000000p-11}
, {0x1.7367b00000000p-6, -0x1.764e0e0000000p-3, -0x1.727bae0000000p-4, 0x1.23d7000000000p-3, 0x1.18cdd00000000p-4, -0x1.38a4120000000p-4, 0x1.50fba40000000p-3, 0x1.ea89c80000000p-3, 0x1.1bea020000000p-3, -0x1.d48d580000000p-2, -0x1.dd7c500000000p-8, 0x1.9fab060000000p-8, -0x1.1d880a0000000p-6, -0x1.57feea0000000p-2, 0x1.3c0f9a0000000p-3, -0x1.1d82240000000p-2}
, {-0x1.3cc6200000000p-2, 0x1.448e1a0000000p-2, -0x1.8fe2280000000p-3, 0x1.0315980000000p-4, 0x1.12158a0000000p-2, -0x1.8402800000000p-3, -0x1.9cfdfe0000000p-4, 0x1.be082e0000000p-2, -0x1.ee17c00000000p-3, 0x1.f96d5e0000000p-4, -0x1.068d200000000p-4, 0x1.938c720000000p-2, 0x1.1ca35a0000000p-1, 0x1.98ccda0000000p-3, -0x1.ec789a0000000p-3, -0x1.74ee2c0000000p-4}
}
, {{-0x1.9ecc060000000p-4, -0x1.31fe940000000p-2, 0x1.d537ba0000000p-3, -0x1.6903c20000000p-4, -0x1.3edf660000000p-3, -0x1.7363c20000000p-5, 0x1.084c620000000p-8, -0x1.369de60000000p-2, 0x1.1ca30c0000000p-4, -0x1.3382440000000p-2, 0x1.37b7ba0000000p-3, -0x1.0ba8460000000p-2, -0x1.7ca0500000000p-3, -0x1.16e6f00000000p-1, -0x1.731ef20000000p-5, -0x1.3db8720000000p-2}
, {0x1.58d8f20000000p-5, -0x1.8c3a220000000p-2, -0x1.11eb520000000p-2, -0x1.6e00ee0000000p-2, -0x1.74fca40000000p-2, -0x1.608b4a0000000p-3, -0x1.33b9860000000p-4, 0x1.16d2460000000p-2, 0x1.efc8860000000p-7, -0x1.5bbb200000000p-2, -0x1.8799540000000p-3, -0x1.7c9c0a0000000p-2, -0x1.1a43ca0000000p-3, -0x1.5ffc460000000p-5, 0x1.45e1900000000p-3, 0x1.4807f40000000p-2}
, {0x1.e19e2e0000000p-3, -0x1.7795d20000000p-4, 0x1.53570a0000000p-3, -0x1.b6b7040000000p-4, -0x1.efc98e0000000p-4, 0x1.4568000000000p-3, 0x1.eb23b80000000p-2, 0x1.9c5d180000000p-3, 0x1.13d0880000000p-6, -0x1.2377200000000p-2, -0x1.d0c6320000000p-5, 0x1.da0cf40000000p-3, 0x1.b925600000000p-5, 0x1.67a4820000000p-6, 0x1.34f7340000000p-3, 0x1.bf4f160000000p-3}
}
, {{0x1.0a664c0000000p-3, -0x1.94bfe60000000p-5, -0x1.4283b00000000p-4, 0x1.c1baa80000000p-9, 0x1.42c6dc0000000p-6, -0x1.6997820000000p-4, -0x1.e281fa0000000p-3, 0x1.4e2ba60000000p-4, 0x1.cd0a7e0000000p-3, 0x1.35def00000000p-4, 0x1.614e0c0000000p-3, 0x1.27cf5e0000000p-2, 0x1.2086b20000000p-3, 0x1.98b9700000000p-5, 0x1.3e601c0000000p-5, 0x1.ef5af40000000p-5}
, {0x1.066b480000000p-2, -0x1.c30bc60000000p-5, 0x1.321a340000000p-6, -0x1.75e55a0000000p-5, 0x1.d798a80000000p-3, 0x1.e3753a0000000p-4, -0x1.4707b20000000p-3, 0x1.a0521a0000000p-4, -0x1.b4583c0000000p-3, 0x1.3bde660000000p-2, -0x1.88bc800000000p-5, -0x1.95b3ae0000000p-2, 0x1.0b52760000000p-3, 0x1.760ff80000000p-2, 0x1.621a600000000p-4, 0x1.acd34a0000000p-3}
, {0x1.80be680000000p-2, -0x1.d069340000000p-4, -0x1.f668940000000p-2, -0x1.a1c12e0000000p-2, -0x1.569b9c0000000p-2, -0x1.2360aa0000000p-7, 0x1.cf54ca0000000p-5, -0x1.8fd7c20000000p-1, -0x1.6df2f20000000p-3, -0x1.4d12a80000000p-2, -0x1.6bb27e0000000p-2, -0x1.e8aab60000000p-2, -0x1.2b2bdc0000000p-1, -0x1.3b0ad60000000p-5, -0x1.5ac4680000000p-5, 0x1.9c11140000000p-7}
}
, {{-0x1.6745a80000000p-2, -0x1.ad612e0000000p-5, -0x1.2bcc120000000p-6, -0x1.1271e80000000p-2, -0x1.f430e60000000p-3, -0x1.c61d500000000p-4, -0x1.cfb0f00000000p-2, 0x1.5681620000000p-4, -0x1.2d6d300000000p-3, 0x1.7682220000000p-7, -0x1.3643c80000000p-3, 0x1.4f7abe0000000p-6, -0x1.5f6e420000000p-2, 0x1.a35a300000000p-5, -0x1.d0a14a0000000p-3, 0x1.c647640000000p-8}
, {0x1.2133260000000p-3, 0x1.a9d31c0000000p-4, 0x1.1f2e3c0000000p-2, 0x1.fc45a00000000p-6, -0x1.f323d40000000p-3, -0x1.392df00000000p-6, -0x1.f889480000000p-7, -0x1.74b0d60000000p-2, 0x1.377cfc0000000p-2, -0x1.f5e7080000000p-4, 0x1.226f600000000p-2, 0x1.9a87900000000p-3, -0x1.173aea0000000p-3, -0x1.d509580000000p-5, 0x1.106e9c0000000p-2, -0x1.3e96020000000p-3}
, {0x1.c9db760000000p-3, -0x1.7abca20000000p-6, -0x1.6aa8f80000000p-5, -0x1.4713ce0000000p-4, 0x1.2c80220000000p-4, -0x1.cf5d7e0000000p-3, -0x1.c391180000000p-3, -0x1.1458a80000000p-2, -0x1.e1c3b80000000p-4, 0x1.0ccc0e0000000p-3, -0x1.e802c20000000p-3, 0x1.0fe44e0000000p-3, 0x1.7248560000000p-5, 0x1.3b77a80000000p-2, 0x1.b1b5f80000000p-5, 0x1.efb2600000000p-3}
}
, {{-0x1.d89c660000000p-5, -0x1.99fc960000000p-5, -0x1.9524e60000000p-7, -0x1.8cb56a0000000p-3, 0x1.74241a0000000p-3, 0x1.9d3ac00000000p-3, -0x1.a5b2de0000000p-3, 0x1.25246e0000000p-7, -0x1.1010ee0000000p-2, 0x1.1abbbe0000000p-2, -0x1.e3179a0000000p-3, 0x1.8bff420000000p-2, 0x1.d8bb4c0000000p-4, 0x1.7919880000000p-2, -0x1.0acfaa0000000p-2, 0x1.8164060000000p-8}
, {0x1.311aa80000000p-2, -0x1.15de9a0000000p-3, 0x1.60a4ec0000000p-3, 0x1.6533b40000000p-6, 0x1.3be43a0000000p-6, 0x1.9c8f7e0000000p-4, 0x1.0c1bd40000000p-4, -0x1.7f81780000000p-3, -0x1.43c1560000000p-3, -0x1.05fe580000000p-4, 0x1.463e4c0000000p-5, 0x1.302bfc0000000p-3, -0x1.ac8a860000000p-3, 0x1.2a6b480000000p-2, 0x1.e96a800000000p-3, 0x1.6f408c0000000p-3}
, {0x1.e9ca380000000p-4, -0x1.3e45d00000000p-7, 0x1.64b1aa0000000p-4, 0x1.761fb80000000p-3, 0x1.3072b40000000p-7, -0x1.5d7c260000000p-2, -0x1.f0523a0000000p-7, 0x1.24ae200000000p-5, -0x1.7309240000000p-3, -0x1.8618ac0000000p-3, -0x1.7c81260000000p-5, 0x1.aa57ec0000000p-3, 0x1.15f1240000000p-9, -0x1.5e775a0000000p-2, -0x1.0ca7660000000p-3, 0x1.b595560000000p-4}
}
, {{-0x1.484d340000000p-3, -0x1.1f6b840000000p-3, -0x1.0140760000000p-4, 0x1.2eabd00000000p-5, -0x1.c8b6100000000p-4, 0x1.1e65200000000p-3, -0x1.1eb77c0000000p-3, 0x1.f74db60000000p-3, -0x1.0bba6c0000000p-1, -0x1.a0f24a0000000p-2, -0x1.a44a580000000p-3, 0x1.1d543a0000000p-2, -0x1.76b71e0000000p-4, -0x1.213de40000000p-4, -0x1.da45920000000p-9, 0x1.68766a0000000p-3}
, {0x1.a558f60000000p-3, -0x1.acf5a20000000p-3, -0x1.af0c820000000p-4, 0x1.cc71da0000000p-4, 0x1.4843a40000000p-4, 0x1.b6b4a80000000p-4, 0x1.6eda2e0000000p-3, -0x1.81d7ea0000000p-5, -0x1.b8124a0000000p-5, -0x1.293b5c0000000p-3, -0x1.1da2620000000p-3, -0x1.41c68c0000000p-3, -0x1.60e64c0000000p-8, -0x1.d11ea60000000p-3, 0x1.be83da0000000p-6, 0x1.5225c80000000p-3}
, {-0x1.456d480000000p-2, 0x1.3d2cfc0000000p-2, 0x1.667df20000000p-5, -0x1.d168620000000p-3, -0x1.90a95a0000000p-4, -0x1.3368a80000000p-4, -0x1.4a18ec0000000p-2, -0x1.095dfe0000000p-3, -0x1.4d3f500000000p-4, 0x1.22b6580000000p-1, -0x1.6d17420000000p-2, 0x1.d46f260000000p-7, 0x1.7f03d20000000p-5, -0x1.5f9cd40000000p-3, -0x1.c4ea3a0000000p-3, 0x1.0584bc0000000p-2}
}
, {{0x1.74dc060000000p-3, -0x1.1159b00000000p-2, -0x1.e2505c0000000p-7, 0x1.fc92980000000p-3, 0x1.539dd20000000p-4, 0x1.4a70b00000000p-2, 0x1.d284e40000000p-4, 0x1.1235000000000p-2, 0x1.a203ee0000000p-3, -0x1.65330c0000000p-7, 0x1.1c28f60000000p-5, -0x1.90444c0000000p-6, 0x1.641b580000000p-3, 0x1.8b1bec0000000p-4, -0x1.a2c3dc0000000p-7, 0x1.3c98a80000000p-2}
, {0x1.0740700000000p-2, 0x1.009a320000000p-3, -0x1.76f7860000000p-3, -0x1.7c8f740000000p-4, -0x1.2c7eea0000000p-3, 0x1.0643c80000000p-3, 0x1.35f5560000000p-5, -0x1.4cef6c0000000p-2, 0x1.5ad0960000000p-3, -0x1.77653e0000000p-5, 0x1.a8f2120000000p-3, 0x1.a89db80000000p-4, -0x1.cdd2420000000p-3, -0x1.4bbfd60000000p-7, 0x1.6ca9f00000000p-3, -0x1.8c5f3a0000000p-6}
, {0x1.3ad9600000000p-5, -0x1.c5af980000000p-2, -0x1.c34e0c0000000p-5, 0x1.95e7b00000000p-4, 0x1.6ad52c0000000p-6, -0x1.161ef00000000p-1, 0x1.3041e60000000p-5, 0x1.f1b0840000000p-5, -0x1.ee88dc0000000p-6, 0x1.d2f6b20000000p-5, -0x1.bd18420000000p-4, -0x1.5a627a0000000p-3, 0x1.3aa5ba0000000p-3, 0x1.492e840000000p-3, 0x1.4160980000000p-2, 0x1.a12e700000000p-4}
}
, {{0x1.c77d680000000p-5, -0x1.3581fe0000000p-3, 0x1.e1d4f40000000p-4, 0x1.d22bb00000000p-6, 0x1.9ea1160000000p-3, 0x1.6363400000000p-5, 0x1.0b68f00000000p-3, -0x1.2df85a0000000p-3, 0x1.bdae120000000p-3, 0x1.2249740000000p-2, 0x1.7c50640000000p-3, -0x1.a1d2b20000000p-6, 0x1.40834a0000000p-3, 0x1.1a93a00000000p-1, 0x1.48f8680000000p-4, 0x1.ab4eb00000000p-4}
, {-0x1.fb00820000000p-3, -0x1.1415e20000000p-2, -0x1.36be440000000p-2, 0x1.b227f40000000p-5, 0x1.c0588e0000000p-4, -0x1.4fadc40000000p-4, 0x1.3cecf00000000p-2, -0x1.98bbd20000000p-4, -0x1.45f6f80000000p-4, 0x1.416fe80000000p-4, 0x1.b32c7c0000000p-8, -0x1.4be1080000000p-2, 0x1.5ed8c20000000p-2, 0x1.1086e80000000p-3, -0x1.99b58a0000000p-4, 0x1.d3e7b20000000p-5}
, {-0x1.f52dae0000000p-8, -0x1.4915e00000000p-1, 0x1.4cf6540000000p-4, -0x1.c43cdc0000000p-5, 0x1.ceabe80000000p-3, 0x1.b47f8a0000000p-3, 0x1.9590800000000p-6, -0x1.4e9f0e0000000p-4, -0x1.dcfe440000000p-2, -0x1.fae9660000000p-3, -0x1.00593e0000000p-2, 0x1.40169a0000000p-7, -0x1.4741480000000p-3, 0x1.9d9db00000000p-3, -0x1.7da96c0000000p-2, 0x1.d096260000000p-4}
}
, {{0x1.be83da0000000p-4, -0x1.53bf640000000p-4, 0x1.a92c240000000p-4, -0x1.86ea6c0000000p-5, 0x1.23adda0000000p-2, 0x1.b51b860000000p-2, 0x1.d614b60000000p-4, -0x1.13f0740000000p-4, -0x1.aebf960000000p-3, -0x1.ff40040000000p-4, 0x1.1ba5780000000p-3, 0x1.73063e0000000p-2, 0x1.49a2440000000p-3, 0x1.5849060000000p-4, 0x1.1447bc0000000p-6, 0x1.a5c6ec0000000p-4}
, {0x1.bac8f80000000p-6, 0x1.0f05bc0000000p-2, -0x1.281b760000000p-3, 0x1.2901800000000p-3, -0x1.a29b8c0000000p-3, -0x1.ce76ca0000000p-2, -0x1.23c1fe0000000p-4, -0x1.04185a0000000p-2, 0x1.332d8c0000000p-7, -0x1.3e73bc0000000p-4, 0x1.0196e20000000p-3, 0x1.fa64840000000p-5, -0x1.3570600000000p-2, -0x1.c6063e0000000p-1, -0x1.25c2ca0000000p-5, -0x1.f9ff080000000p-1}
, {-0x1.c98bce0000000p-3, 0x1.1a74480000000p-3, 0x1.a7e30e0000000p-2, -0x1.6fea880000000p-4, -0x1.3aaf680000000p-7, -0x1.561cdc0000000p-2, -0x1.2065fc0000000p-3, -0x1.101c560000000p-3, 0x1.522f2c0000000p-3, -0x1.d6645c0000000p-6, 0x1.6a47c00000000p-5, 0x1.f776200000000p-4, -0x1.1bbe6e0000000p-3, -0x1.59f3cc0000000p-3, -0x1.5d5f140000000p-3, 0x1.fbf4a60000000p-2}
}
, {{0x1.6ff1b80000000p-2, 0x1.7a64260000000p-2, -0x1.a2db9a0000000p-6, -0x1.f0eed20000000p-2, -0x1.2128ec0000000p-1, 0x1.1f34b80000000p-4, -0x1.226ab60000000p-5, -0x1.9d7a7a0000000p-1, 0x1.0759e00000000p-6, 0x1.3820b00000000p-3, 0x1.2129060000000p-5, -0x1.034ab60000000p-3, -0x1.c569040000000p-2, -0x1.fb83a20000000p-4, 0x1.f5d07c0000000p-3, -0x1.142f780000000p-2}
, {-0x1.ab45e80000000p-6, -0x1.9eade40000000p-3, 0x1.f3f9f80000000p-3, -0x1.428dce0000000p-5, 0x1.7bdd5c0000000p-3, 0x1.6d8b900000000p-3, -0x1.b390280000000p-4, 0x1.4415c80000000p-4, 0x1.afa9da0000000p-4, 0x1.33af400000000p-4, 0x1.41bad20000000p-2, 0x1.31d7ee0000000p-3, -0x1.b7275e0000000p-7, -0x1.a7dfa40000000p-5, -0x1.28c2a40000000p-3, 0x1.ab3a800000000p-5}
, {-0x1.fc56640000000p-5, -0x1.27b35a0000000p-2, -0x1.5a24ae0000000p-3, -0x1.26ee9a0000000p-3, 0x1.0c923e0000000p-2, 0x1.022c640000000p-4, 0x1.dc922e0000000p-3, -0x1.bd68e80000000p-3, -0x1.a0eeee0000000p-3, -0x1.28dc720000000p-1, -0x1.201d0c0000000p-2, -0x1.12c1080000000p-2, 0x1.7ed99e0000000p-3, -0x1.6c61c00000000p-5, 0x1.25b1780000000p-4, 0x1.2520780000000p-3}
}
, {{0x1.a38fb60000000p-3, -0x1.30fd160000000p-3, 0x1.d173a40000000p-7, -0x1.48676e0000000p-8, 0x1.3512ca0000000p-3, 0x1.895f880000000p-3, 0x1.ff074a0000000p-6, 0x1.5f70900000000p-5, 0x1.2b2fd60000000p-4, -0x1.402a940000000p-3, 0x1.ae90d40000000p-3, 0x1.4e89a60000000p-3, 0x1.06b84a0000000p-8, 0x1.d6910c0000000p-6, 0x1.8ed2f60000000p-3, -0x1.1b23f00000000p-2}
, {-0x1.513c480000000p-5, -0x1.529f420000000p-2, -0x1.49f1400000000p-5, 0x1.6e16ce0000000p-5, -0x1.a750b60000000p-4, 0x1.a122ba0000000p-5, 0x1.3da1920000000p-3, 0x1.0fef560000000p-5, -0x1.4a0e520000000p-4, 0x1.9525d40000000p-4, -0x1.f9cd720000000p-7, 0x1.0533d20000000p-5, 0x1.dc70cc0000000p-5, 0x1.8f148e0000000p-4, -0x1.81af060000000p-2, 0x1.61140a0000000p-1}
, {0x1.3c7d340000000p-5, 0x1.2935540000000p-3, -0x1.03123a0000000p-2, -0x1.0bc7800000000p-3, -0x1.4f8c3e0000000p-2, 0x1.378bc80000000p-2, 0x1.39bb0c0000000p-2, -0x1.43ac560000000p-3, 0x1.6b8d3a0000000p-4, -0x1.9a6f9c0000000p-2, 0x1.14f54c0000000p-3, -0x1.878f740000000p-3, -0x1.d700940000000p-2, -0x1.40e7e00000000p-2, 0x1.38fe2e0000000p-2, -0x1.d92d040000000p-2}
}
, {{-0x1.6cdbae0000000p-2, 0x1.aa74080000000p-4, 0x1.4a63ee0000000p-5, 0x1.8c82860000000p-3, 0x1.7edc4c0000000p-3, 0x1.d30f760000000p-3, 0x1.69af960000000p-4, -0x1.b355ce0000000p-4, -0x1.933a2a0000000p-5, 0x1.b354140000000p-3, -0x1.815cbc0000000p-4, -0x1.02444c0000000p-4, 0x1.48ed1c0000000p-4, 0x1.ec46820000000p-2, -0x1.5cb5220000000p-2, -0x1.80b7e00000000p-3}
, {-0x1.7952f40000000p-6, -0x1.13036a0000000p-4, -0x1.aee8780000000p-4, -0x1.617ba20000000p-2, 0x1.98e3ca0000000p-4, 0x1.428b960000000p-2, 0x1.f2918e0000000p-8, -0x1.9e97400000000p-3, -0x1.acc4c60000000p-4, 0x1.67a5620000000p-2, -0x1.2aacca0000000p-2, -0x1.094eaa0000000p-2, -0x1.e344ac0000000p-8, 0x1.6e6c0e0000000p-3, 0x1.ff7d340000000p-4, 0x1.7aeb1e0000000p-6}
, {-0x1.4617d40000000p-2, -0x1.4e3bee0000000p-2, -0x1.99cd8a0000000p-6, 0x1.5f91f20000000p-2, 0x1.c23dd00000000p-3, -0x1.4e95920000000p-6, -0x1.da75620000000p-4, -0x1.83b1bc0000000p-6, -0x1.39deea0000000p-2, -0x1.18e6860000000p-2, 0x1.40c2fc0000000p-3, -0x1.fb7a240000000p-5, 0x1.2de9440000000p-2, 0x1.90919c0000000p-4, -0x1.fbd93c0000000p-3, 0x1.03c08e0000000p-9}
}
, {{-0x1.4cbf200000000p-3, -0x1.31aef00000000p-3, 0x1.33071c0000000p-3, 0x1.78ba1a0000000p-3, 0x1.bbc17a0000000p-4, -0x1.3459a40000000p-2, -0x1.d1797e0000000p-6, -0x1.77b24a0000000p-3, -0x1.8c682c0000000p-4, 0x1.2656f00000000p-4, 0x1.a804c60000000p-4, -0x1.ad8e4c0000000p-3, 0x1.5eda780000000p-2, -0x1.4d317e0000000p-2, 0x1.c820700000000p-3, 0x1.275fba0000000p-3}
, {0x1.5373d40000000p-3, 0x1.426f740000000p-3, -0x1.4320c60000000p-3, -0x1.34c4740000000p-2, -0x1.4ecb600000000p-2, 0x1.4bec840000000p-4, 0x1.88fba00000000p-3, -0x1.3c7cf40000000p-2, 0x1.5abb280000000p-4, -0x1.6e3d1c0000000p-2, 0x1.1fc4e40000000p-3, -0x1.4809600000000p-3, 0x1.a75bc80000000p-4, -0x1.42e0ac0000000p-2, -0x1.410b080000000p-3, -0x1.55d6400000000p-8}
, {-0x1.b239120000000p-6, -0x1.084e800000000p-3, -0x1.5d05540000000p-4, -0x1.e131a80000000p-4, 0x1.78b5760000000p-3, 0x1.3b394e0000000p-5, -0x1.3dcbe20000000p-5, 0x1.cfe1fe0000000p-2, -0x1.a5767a0000000p-3, 0x1.030fc40000000p-3, -0x1.2b05c00000000p-2, 0x1.e074f80000000p-3, 0x1.458bb00000000p-2, -0x1.33d74e0000000p-3, -0x1.1adafc0000000p-2, 0x1.eafcd60000000p-5}
}
, {{0x1.47a1380000000p-2, 0x1.82d9d80000000p-5, -0x1.b46e3e0000000p-6, 0x1.73ed5a0000000p-3, 0x1.0e6c120000000p-2, -0x1.3a54b80000000p-2, 0x1.7b18700000000p-3, 0x1.27383c0000000p-3, -0x1.4b32b60000000p-3, -0x1.0d4aec0000000p-3, 0x1.c2bef20000000p-4, -0x1.5c12b80000000p-4, 0x1.5d7ca20000000p-6, 0x1.a2dfd40000000p-3, 0x1.22beec0000000p-2, 0x1.ca69280000000p-8}
, {0x1.a617240000000p-4, -0x1.68f6e80000000p-4, 0x1.0248ea0000000p-5, -0x1.88cdd40000000p-3, -0x1.01ab140000000p-4, 0x1.38536e0000000p-7, 0x1.dab8020000000p-4, -0x1.1e33b80000000p-3, 0x1.5b0d1c0000000p-2, -0x1.6f59920000000p-3, -0x1.2282fa0000000p-5, -0x1.03db960000000p-2, -0x1.2e14820000000p-2, -0x1.4fa5460000000p-1, -0x1.6eeeb60000000p-4, -0x1.1422640000000p-1}
, {-0x1.33050c0000000p-1, 0x1.8c0c180000000p-4, 0x1.59502e0000000p-4, -0x1.38acd00000000p-2, -0x1.285c0c0000000p-1, -0x1.9bb1e00000000p-3, -0x1.dc1aae0000000p-2, 0x1.0a45ea0000000p-1, -0x1.a565d80000000p-3, 0x1.7691a40000000p-2, -0x1.8da0f40000000p-2, 0x1.55c9cc0000000p-3, -0x1.e948180000000p-3, 0x1.77e0660000000p-7, -0x1.523f720000000p-2, 0x1.160fba0000000p-5}
}
, {{0x1.3768740000000p-2, 0x1.5cd2960000000p-2, -0x1.8784c60000000p-3, -0x1.4c23060000000p-3, 0x1.696d920000000p-4, -0x1.7fb1e60000000p-2, -0x1.7f3a9e0000000p-2, 0x1.302bca0000000p-4, 0x1.1292ec0000000p-2, 0x1.56d5500000000p-3, 0x1.f5bc1a0000000p-3, -0x1.d121040000000p-2, -0x1.5367800000000p-3, -0x1.774a300000000p-2, 0x1.4270580000000p-2, -0x1.361e4a0000000p-2}
, {-0x1.0971000000000p-2, -0x1.2fc5100000000p-3, -0x1.91256a0000000p-3, -0x1.d021300000000p-5, -0x1.2f9cc60000000p-7, -0x1.564c300000000p-3, 0x1.5397160000000p-3, -0x1.1ea70e0000000p-6, -0x1.aa81600000000p-4, -0x1.d550c80000000p-3, -0x1.150d000000000p-5, -0x1.d775920000000p-6, 0x1.4fe4ae0000000p-3, 0x1.3f0fd40000000p-4, -0x1.1b4f800000000p-2, 0x1.841eac0000000p-2}
, {0x1.5ea3ac0000000p-2, 0x1.e557720000000p-2, 0x1.ecd6520000000p-6, -0x1.716e200000000p-2, -0x1.666fd60000000p-3, -0x1.c25ac00000000p-7, 0x1.6157bc0000000p-8, 0x1.e599980000000p-5, 0x1.0a79d40000000p-3, 0x1.380a520000000p-3, -0x1.877bb60000000p-3, 0x1.2db6d80000000p-4, -0x1.5b82540000000p-3, 0x1.57709e0000000p-5, 0x1.def5640000000p-12, -0x1.66b87e0000000p-3}
}
, {{-0x1.e757c40000000p-7, -0x1.5e1a760000000p-3, -0x1.dfa4c00000000p-7, 0x1.f895320000000p-3, 0x1.a1cc500000000p-3, -0x1.06c0c00000000p-2, 0x1.96c9220000000p-3, -0x1.3c6d340000000p-3, -0x1.18c0000000000p-2, -0x1.eba5000000000p-2, 0x1.53f8be0000000p-3, 0x1.2bd3c40000000p-8, 0x1.381c9c0000000p-2, -0x1.d861c20000000p-2, -0x1.8d8da60000000p-3, -0x1.a6d5d40000000p-3}
, {-0x1.8f597e0000000p-2, -0x1.5a45600000000p-3, 0x1.a377ea0000000p-3, -0x1.6faf740000000p-3, 0x1.2527a20000000p-3, -0x1.9102da0000000p-3, -0x1.657e740000000p-2, 0x1.302b260000000p-6, -0x1.5b8bf00000000p-5, 0x1.0d50f20000000p-2, -0x1.93339a0000000p-3, 0x1.8efefe0000000p-2, 0x1.77b6e80000000p-4, 0x1.5eaace0000000p-4, -0x1.e5f2f60000000p-3, -0x1.cf02100000000p-6}
, {-0x1.516e880000000p-2, 0x1.699f0e0000000p-3, 0x1.1a429a0000000p-2, -0x1.fe0e1a0000000p-3, -0x1.3519460000000p-6, -0x1.c84b9e0000000p-4, 0x1.740eca0000000p-5, -0x1.0aa3400000000p-3, -0x1.525ebc0000000p-3, 0x1.40c7360000000p-3, -0x1.6bf0a20000000p-3, 0x1.50fdb00000000p-5, 0x1.d931b00000000p-5, 0x1.9928320000000p-4, -0x1.f7c1700000000p-6, -0x1.8090a80000000p-4}
}
, {{0x1.df2a8a0000000p-4, -0x1.d5ce4c0000000p-3, 0x1.294b000000000p-4, 0x1.4290e20000000p-2, 0x1.9e82920000000p-3, 0x1.ecb6600000000p-3, -0x1.4baaac0000000p-3, -0x1.245fc20000000p-5, -0x1.f425fc0000000p-4, -0x1.90b1980000000p-2, -0x1.17cfa00000000p-3, 0x1.10fc580000000p-4, 0x1.4f187c0000000p-3, 0x1.3b3cc40000000p-3, -0x1.62ecd00000000p-5, 0x1.01deae0000000p-2}
, {-0x1.5615a40000000p-3, 0x1.9b78660000000p-4, -0x1.89284c0000000p-3, -0x1.1b406c0000000p-2, -0x1.5e09460000000p-2, 0x1.c744180000000p-5, 0x1.bf231e0000000p-3, -0x1.22d7de0000000p+0, 0x1.3609f40000000p-9, -0x1.9c47840000000p-3, -0x1.f8bad80000000p-5, -0x1.11b4280000000p-2, -0x1.25bc340000000p-2, 0x1.d3e3960000000p-4, 0x1.bb70c40000000p-6, -0x1.91ef580000000p-3}
, {0x1.590aae0000000p-2, 0x1.7ea1480000000p-3, 0x1.0f0d160000000p-3, -0x1.23eef80000000p-9, 0x1.f88c500000000p-3, -0x1.6fa58a0000000p-2, -0x1.c083800000000p-2, -0x1.c98a260000000p-3, 0x1.0ac1120000000p-5, 0x1.c99f900000000p-4, 0x1.0f7e9a0000000p-2, 0x1.f002080000000p-3, -0x1.038a360000000p-2, -0x1.8dccf40000000p-4, 0x1.c9a5300000000p-3, 0x1.9c93a20000000p-4}
}
, {{-0x1.0246b60000000p-2, 0x1.cd3c8e0000000p-11, 0x1.0b84ae0000000p-4, -0x1.27de680000000p-2, -0x1.430cbc0000000p-2, -0x1.1fbe1e0000000p-5, 0x1.973f3c0000000p-6, 0x1.c404ea0000000p-5, -0x1.03e9ce0000000p-5, -0x1.abd9ec0000000p-8, -0x1.d4dc6e0000000p-3, 0x1.17be320000000p-7, -0x1.5f4fee0000000p-4, 0x1.9590ce0000000p-4, -0x1.32e4de0000000p-2, -0x1.635bee0000000p-5}
, {-0x1.1f74e80000000p-4, 0x1.4ff44a0000000p-7, 0x1.b598780000000p-4, -0x1.e7eab80000000p-10, 0x1.561b7a0000000p-3, 0x1.ea98920000000p-3, 0x1.3432b00000000p-5, 0x1.0698480000000p-1, 0x1.ae81da0000000p-7, -0x1.c2daba0000000p-3, -0x1.2333f20000000p-4, 0x1.6751800000000p-4, -0x1.fa71b60000000p-7, -0x1.df67720000000p-3, -0x1.076d160000000p-2, -0x1.29f6ec0000000p-10}
, {0x1.2ed9720000000p-2, 0x1.213cec0000000p-3, 0x1.1279860000000p-3, 0x1.a275640000000p-3, -0x1.5134960000000p-3, -0x1.1100f20000000p-4, 0x1.1f39420000000p-3, -0x1.d6e94c0000000p-3, 0x1.bcf9100000000p-4, 0x1.a7d2540000000p-2, 0x1.2d495c0000000p-3, 0x1.80f1ae0000000p-3, -0x1.f2d9a60000000p-7, 0x1.b860800000000p-2, 0x1.e4c40e0000000p-3, 0x1.a01a3a0000000p-3}
}
, {{-0x1.70ce140000000p-2, 0x1.b466b40000000p-5, -0x1.021bda0000000p-3, -0x1.b43a600000000p-3, 0x1.b003200000000p-9, 0x1.2883ba0000000p-2, -0x1.8aa1b20000000p-4, -0x1.3e93940000000p-2, -0x1.f689160000000p-12, 0x1.56cd6e0000000p-2, -0x1.7d05c80000000p-4, 0x1.ae03f40000000p-3, 0x1.69ee6a0000000p-5, 0x1.f5e9c40000000p-3, -0x1.514c2a0000000p-2, 0x1.a6bd1a0000000p-3}
, {-0x1.cf67440000000p-5, -0x1.21c0da0000000p-3, -0x1.18db400000000p-2, -0x1.0ee4880000000p-2, -0x1.063b120000000p-3, 0x1.58d80a0000000p-6, 0x1.f5b67c0000000p-3, 0x1.3c86b00000000p-8, -0x1.2151ea0000000p-3, -0x1.7b13c00000000p-3, -0x1.7105920000000p-7, -0x1.1010700000000p-2, -0x1.8173800000000p-2, -0x1.0e626c0000000p-6, 0x1.6fbc100000000p-6, 0x1.1323540000000p-3}
, {0x1.de4f440000000p-5, 0x1.35cd260000000p-6, 0x1.3242cc0000000p-3, 0x1.5cff0c0000000p-4, -0x1.681fac0000000p-2, -0x1.051a4c0000000p-4, 0x1.2a1ca60000000p-5, -0x1.0c43c60000000p-3, 0x1.c209de0000000p-6, 0x1.b905800000000p-5, -0x1.6a6b980000000p-3, 0x1.4cba220000000p-6, -0x1.0970600000000p-6, 0x1.8335080000000p-3, -0x1.263d580000000p-2, 0x1.5f7de60000000p-2}
}
, {{-0x1.c178da0000000p-4, 0x1.c4fe440000000p-6, 0x1.c8e87c0000000p-5, 0x1.e8b6340000000p-3, 0x1.6a25460000000p-3, 0x1.e4fbb40000000p-4, 0x1.f5f77e0000000p-4, -0x1.09e0540000000p-3, 0x1.1228c20000000p-3, 0x1.16c5a40000000p-3, 0x1.b9bd1c0000000p-5, -0x1.120e0a0000000p-2, 0x1.9438a60000000p-2, -0x1.866c340000000p-3, 0x1.6de9c40000000p-6, 0x1.6c61e60000000p-3}
, {0x1.0184c40000000p-3, -0x1.ab278e0000000p-5, -0x1.0c2a0e0000000p-3, -0x1.ad23620000000p-5, -0x1.88c1c20000000p-2, -0x1.b0ea220000000p-2, 0x1.067e060000000p-3, 0x1.06361c0000000p-2, 0x1.40615c0000000p-2, -0x1.39e0500000000p-3, -0x1.c432a00000000p-3, -0x1.1cc82a0000000p-1, -0x1.d15d3a0000000p-3, -0x1.676ae40000000p-1, 0x1.0f33e20000000p-6, -0x1.3533120000000p-3}
, {-0x1.66c0a60000000p-3, -0x1.15ce8e0000000p-2, -0x1.0ffb580000000p-2, -0x1.cb6cb60000000p-4, -0x1.3502140000000p-2, -0x1.533f120000000p-3, 0x1.7e54c80000000p-4, 0x1.0da93e0000000p-5, -0x1.abf2b00000000p-3, -0x1.b2db880000000p-3, 0x1.27e35a0000000p-5, -0x1.bdbe1e0000000p-6, -0x1.eef4680000000p-3, -0x1.367dde0000000p-3, 0x1.0893760000000p-2, 0x1.0e634c0000000p-2}
}
, {{0x1.bc58a40000000p-4, 0x1.713b460000000p-6, -0x1.5fee740000000p-2, -0x1.23b1500000000p-2, 0x1.e8b19a0000000p-4, -0x1.575df40000000p-2, -0x1.e0995e0000000p-3, 0x1.ed54b00000000p-4, -0x1.4f63380000000p-2, 0x1.eb49280000000p-4, -0x1.38221c0000000p-4, -0x1.e681540000000p-3, 0x1.49a9140000000p-4, 0x1.39d9a40000000p-4, 0x1.823aba0000000p-6, 0x1.036df20000000p-2}
, {-0x1.5b4a4a0000000p-5, 0x1.a136b80000000p-2, -0x1.6b2e820000000p-4, -0x1.1584280000000p-2, -0x1.3b5b7a0000000p-4, -0x1.4321700000000p-2, -0x1.6af4b60000000p-8, -0x1.82ef820000000p-2, -0x1.a9ca380000000p-3, 0x1.6a15460000000p-2, -0x1.f1dcd80000000p-3, -0x1.6dfd440000000p-5, 0x1.cd89b00000000p-3, 0x1.743a480000000p-3, 0x1.cc598a0000000p-4, 0x1.4bbaf60000000p-6}
, {0x1.84d1fc0000000p-4, -0x1.3915160000000p-3, -0x1.f77d140000000p-5, -0x1.4539320000000p-4, 0x1.1235d00000000p-8, -0x1.4bdad80000000p-5, 0x1.107a3e0000000p-5, 0x1.527dda0000000p-5, 0x1.0468d20000000p-3, 0x1.dfcca40000000p-5, 0x1.47640c0000000p-3, -0x1.18e6e60000000p-3, -0x1.2b31a20000000p-3, -0x1.50b74c0000000p-5, -0x1.247cce0000000p-9, 0x1.0dbeb80000000p-8}
}
, {{0x1.6ea1ba0000000p-3, 0x1.0693840000000p-2, 0x1.3513f20000000p-5, 0x1.4128420000000p-2, 0x1.2b5b0c0000000p-4, -0x1.1154420000000p-2, 0x1.45fa180000000p-3, -0x1.1a125c0000000p-2, 0x1.6afe620000000p-2, -0x1.c6d2100000000p-4, -0x1.3e4abc0000000p-5, -0x1.2beeaa0000000p-2, 0x1.9244f40000000p-5, -0x1.f2fdde0000000p-2, 0x1.ab1d520000000p-3, -0x1.451d3a0000000p-1}
, {0x1.6637ee0000000p-4, 0x1.00b6300000000p-3, -0x1.68549a0000000p-5, -0x1.9567120000000p-4, 0x1.72d89c0000000p-5, -0x1.eb6cea0000000p-3, -0x1.49db1a0000000p-2, 0x1.edf36c0000000p-3, 0x1.4c68c60000000p-3, 0x1.b8bbce0000000p-5, 0x1.1b140a0000000p-2, -0x1.fb470e0000000p-4, -0x1.c909b00000000p-10, 0x1.6372c60000000p-6, -0x1.d0f1020000000p-5, 0x1.1586320000000p-2}
, {-0x1.1ee4c60000000p-2, -0x1.0be83a0000000p-2, 0x1.3401a80000000p-4, 0x1.9635480000000p-4, 0x1.7a68860000000p-3, 0x1.2800880000000p-2, 0x1.7775b60000000p-3, 0x1.5f706a0000000p-4, -0x1.6c218c0000000p-3, -0x1.2512c80000000p-2, -0x1.6a098c0000000p-9, -0x1.4706d60000000p-5, 0x1.3e14ce0000000p-5, 0x1.909b660000000p-4, 0x1.49794e0000000p-4, -0x1.b8aed60000000p-6}
}
, {{0x1.ebe76a0000000p-4, -0x1.42e86e0000000p-5, -0x1.de65540000000p-10, 0x1.e374bc0000000p-3, 0x1.a5cc0e0000000p-2, 0x1.0be0fa0000000p-2, 0x1.46d2e00000000p-7, 0x1.3c90ac0000000p-4, -0x1.c50b5c0000000p-2, -0x1.3b139a0000000p-4, -0x1.fbb84c0000000p-4, -0x1.fa2f340000000p-7, 0x1.8741200000000p-2, 0x1.7396c40000000p-3, 0x1.ea74e80000000p-4, 0x1.5319860000000p-3}
, {-0x1.11eb8a0000000p-2, 0x1.2b4d2a0000000p-6, 0x1.008b300000000p-3, -0x1.a07d860000000p-3, -0x1.04eb140000000p-2, 0x1.3983aa0000000p-5, 0x1.25dca00000000p-3, 0x1.7aacea0000000p-3, -0x1.8d40840000000p-3, 0x1.4415fc0000000p-3, -0x1.1323f20000000p-2, 0x1.b456ea0000000p-5, -0x1.36ddfa0000000p-3, 0x1.782ac60000000p-5, -0x1.94cc9c0000000p-2, -0x1.916c2c0000000p-3}
, {0x1.7770720000000p-4, -0x1.9ba4a60000000p-6, -0x1.bd92380000000p-4, 0x1.f8f6100000000p-5, -0x1.7ed8240000000p-4, -0x1.61856a0000000p-5, 0x1.ac039a0000000p-3, 0x1.7bbbee0000000p-2, -0x1.5420320000000p-2, -0x1.13abf80000000p-3, -0x1.a288b40000000p-6, -0x1.3c22b80000000p-7, -0x1.6e84fc0000000p-3, -0x1.2bd3200000000p-2, -0x1.70fece0000000p-4, 0x1.cfcb940000000p-3}
}
, {{0x1.40ae3e0000000p-5, -0x1.1add040000000p-4, -0x1.234ca80000000p-3, -0x1.4a456c0000000p-3, -0x1.c58eb80000000p-4, 0x1.55b3500000000p-2, 0x1.e5c2440000000p-3, 0x1.16e39a0000000p-2, -0x1.0d186a0000000p-3, -0x1.1a78400000000p-2, -0x1.c1ce220000000p-2, -0x1.9529fe0000000p-4, -0x1.7bea640000000p-3, -0x1.13fe1a0000000p-4, -0x1.0d72ce0000000p-4, -0x1.1c3d220000000p-4}
, {0x1.8087e20000000p-4, -0x1.67c6e60000000p-3, -0x1.22b7840000000p-5, 0x1.e1d6700000000p-3, -0x1.ef19900000000p-4, -0x1.378fd20000000p-7, 0x1.180a4c0000000p-4, 0x1.fc13880000000p-10, -0x1.7016e00000000p-3, 0x1.04984c0000000p-7, -0x1.9fd6de0000000p-6, -0x1.19a91a0000000p-2, -0x1.23a8000000000p-3, 0x1.ecf09c0000000p-8, 0x1.f9a76c0000000p-4, 0x1.e5a7600000000p-3}
, {0x1.30c99c0000000p-4, -0x1.7c8ef20000000p-3, -0x1.3804e60000000p-2, -0x1.25d58c0000000p-4, -0x1.ef20ce0000000p-3, 0x1.33c0700000000p-2, 0x1.b3a07c0000000p-2, 0x1.5798080000000p-5, -0x1.4c0f860000000p-8, -0x1.6e53de0000000p-3, -0x1.4901360000000p-3, -0x1.d34d200000000p-3, -0x1.e7a7380000000p-3, -0x1.891f340000000p-6, -0x1.7206820000000p-3, 0x1.214ae40000000p-2}
}
, {{0x1.a0d77e0000000p-6, -0x1.5ba21a0000000p-2, 0x1.e5cd500000000p-9, -0x1.1ada540000000p-3, 0x1.db30460000000p-3, 0x1.c7445c0000000p-2, -0x1.e796e20000000p-6, 0x1.0b80340000000p-3, -0x1.aec5b80000000p-5, 0x1.0545f80000000p-2, -0x1.ece9b60000000p-9, 0x1.3ff5760000000p-7, 0x1.4fe7960000000p-3, 0x1.2040740000000p-1, -0x1.1ed0640000000p-3, 0x1.5d08e60000000p-3}
, {0x1.ff9d080000000p-4, 0x1.4edef00000000p-6, -0x1.e0b4b80000000p-5, -0x1.ca36560000000p-4, 0x1.2510840000000p-4, 0x1.66b0200000000p-3, -0x1.5b8f400000000p-4, -0x1.61dae20000000p-3, -0x1.fe7eb40000000p-5, -0x1.1af4d60000000p-4, 0x1.cf42a40000000p-3, 0x1.df7db80000000p-9, 0x1.c9463c0000000p-4, -0x1.21a2500000000p-4, 0x1.e854ba0000000p-4, -0x1.8aec620000000p-2}
, {0x1.53c9180000000p-4, 0x1.97a0d60000000p-2, -0x1.4f03fc0000000p-2, -0x1.fd98200000000p-8, -0x1.27b7b20000000p-2, -0x1.1c66b60000000p-2, -0x1.868fbc0000000p-4, 0x1.8afcc20000000p-6, -0x1.a51c1e0000000p-6, 0x1.a54cee0000000p-5, 0x1.f806020000000p-6, -0x1.4a41660000000p-1, -0x1.e598420000000p-3, -0x1.8a2f260000000p-2, 0x1.b0b37a0000000p-3, -0x1.9807580000000p-3}
}
, {{-0x1.7535b20000000p-5, -0x1.54f21e0000000p-11, -0x1.aea60a0000000p-3, -0x1.3100340000000p-2, -0x1.3fc4660000000p-3, -0x1.4f217e0000000p-4, -0x1.7ff54a0000000p-7, 0x1.5026360000000p-5, -0x1.1242e00000000p-3, -0x1.2d69480000000p-2, -0x1.0c608c0000000p-2, -0x1.acb3de0000000p-2, -0x1.812bf00000000p-2, -0x1.98f3920000000p-8, -0x1.a5eb2e0000000p-6, 0x1.678ac60000000p-5}
, {-0x1.f293de0000000p-5, -0x1.d901e60000000p-7, -0x1.ca15800000000p-4, -0x1.aa901a0000000p-3, -0x1.75193c0000000p-2, 0x1.ea6e440000000p-5, -0x1.2724fe0000000p-4, 0x1.6656f20000000p-2, -0x1.1d15b00000000p-3, -0x1.dde4b80000000p-4, -0x1.caa9e20000000p-3, 0x1.2046f00000000p-3, -0x1.4a06140000000p-1, -0x1.be7d700000000p-6, -0x1.26d7480000000p-4, -0x1.34b7760000000p-3}
, {-0x1.2e1bda0000000p-2, -0x1.4c10ba0000000p-4, 0x1.2f61fa0000000p-6, -0x1.239ba20000000p-6, -0x1.329f260000000p-6, 0x1.2dfca00000000p-5, -0x1.0a4b8c0000000p-4, -0x1.d529c60000000p-4, 0x1.741bda0000000p-6, 0x1.9ffc7a0000000p-3, -0x1.238fc60000000p-3, 0x1.5df1c80000000p-3, -0x1.b5ecc60000000p-5, 0x1.7c35b60000000p-3, -0x1.dff8500000000p-5, 0x1.9ca4a40000000p-3}
}
, {{-0x1.af31fa0000000p-4, 0x1.6209f00000000p-5, 0x1.7a6c920000000p-5, 0x1.1f8b280000000p-2, -0x1.5751b20000000p-4, 0x1.00449e0000000p-2, 0x1.48d5100000000p-4, -0x1.1c8bc20000000p-5, -0x1.c439140000000p-5, -0x1.eed7a60000000p-6, 0x1.51144c0000000p-4, 0x1.f513d20000000p-4, -0x1.8fa4fe0000000p-4, -0x1.d4aa980000000p-6, -0x1.0f5e6e0000000p-3, -0x1.69052e0000000p-3}
, {0x1.31f96e0000000p-5, 0x1.1ff8ae0000000p-4, 0x1.3d5d520000000p-3, 0x1.17e03e0000000p-5, 0x1.34a1200000000p-4, 0x1.0a9e120000000p-4, 0x1.705f540000000p-3, 0x1.8f82b80000000p-3, -0x1.ca03560000000p-2, 0x1.06ef440000000p-3, -0x1.0329ea0000000p-2, 0x1.d73e3a0000000p-5, -0x1.9deae80000000p-4, 0x1.fda2c20000000p-3, -0x1.9c63300000000p-2, 0x1.4813140000000p-3}
, {0x1.3c0daa0000000p-5, -0x1.fcf38c0000000p-2, 0x1.c586740000000p-3, -0x1.d706740000000p-3, -0x1.4b88560000000p-2, 0x1.e4d58c0000000p-4, 0x1.07d4300000000p-7, -0x1.4a73520000000p-2, -0x1.028e960000000p-7, -0x1.46d6fe0000000p-4, 0x1.386c1e0000000p-3, 0x1.9d88c00000000p-2, -0x1.d9a7300000000p-4, 0x1.9ee0c20000000p-2, 0x1.2715b60000000p-2, 0x1.5571a80000000p-2}
}
, {{0x1.c019a40000000p-4, -0x1.26a7ac0000000p-1, 0x1.3af65e0000000p-3, 0x1.7d89b00000000p-7, -0x1.ddce900000000p-5, -0x1.355ace0000000p-2, 0x1.31252c0000000p-7, 0x1.6839640000000p-3, -0x1.e00b620000000p-3, -0x1.7dc2ce0000000p-2, -0x1.92e3420000000p-4, -0x1.6ed3980000000p-3, 0x1.5388fa0000000p-2, -0x1.5ba6240000000p-3, -0x1.c6cc9e0000000p-5, -0x1.adc6fa0000000p-5}
, {-0x1.c146b00000000p-4, -0x1.109c600000000p-1, 0x1.3e54ca0000000p-5, 0x1.c2b4680000000p-3, -0x1.41de520000000p-2, 0x1.b635ea0000000p-3, -0x1.9422600000000p-3, 0x1.6678900000000p-3, -0x1.9bf3da0000000p-4, -0x1.7e8bec0000000p-3, -0x1.40f8ac0000000p-4, 0x1.f75b020000000p-3, 0x1.5105e20000000p-3, 0x1.d3ada80000000p-5, -0x1.b5df900000000p-4, -0x1.c46b860000000p-4}
, {-0x1.745e720000000p-6, -0x1.1282900000000p-1, 0x1.e13ab20000000p-3, 0x1.60dd0a0000000p-5, 0x1.08dc6c0000000p-3, 0x1.01a3fc0000000p-3, 0x1.354ab20000000p-2, -0x1.6851de0000000p-5, 0x1.3d90f60000000p-5, -0x1.6d50900000000p-3, 0x1.96b5d60000000p-3, 0x1.9c295e0000000p-4, 0x1.7cacda0000000p-3, -0x1.28dc6e0000000p-6, -0x1.64e2100000000p-7, 0x1.085db40000000p-2}
}
, {{-0x1.2b3c5e0000000p-2, -0x1.7d38be0000000p-4, 0x1.95c63c0000000p-4, -0x1.092ade0000000p-2, 0x1.32b7400000000p-6, -0x1.13f0c00000000p-3, -0x1.ee25c80000000p-4, 0x1.b08f9a0000000p-4, -0x1.6c3cc00000000p-5, 0x1.4113500000000p-2, -0x1.ec4b460000000p-2, 0x1.757b920000000p-4, 0x1.39baaa0000000p-4, -0x1.2738380000000p-5, -0x1.2db0c40000000p-1, 0x1.f1b8e00000000p-2}
, {-0x1.aea6120000000p-3, 0x1.9cda540000000p-5, 0x1.c6b68e0000000p-3, 0x1.3dd8d80000000p-3, 0x1.f138e20000000p-3, 0x1.062e560000000p-3, 0x1.4afa120000000p-4, 0x1.5288b20000000p-10, -0x1.77956a0000000p-2, -0x1.ca30fc0000000p-3, 0x1.596c2a0000000p-3, 0x1.055de00000000p-3, -0x1.fe0af40000000p-4, -0x1.2ac17a0000000p-5, -0x1.5239b20000000p-2, -0x1.feb3220000000p-3}
, {-0x1.3c955c0000000p-2, -0x1.70a4900000000p-2, -0x1.472af80000000p-3, -0x1.ac5ff40000000p-3, 0x1.4253660000000p-2, 0x1.0e98460000000p-4, 0x1.5791f40000000p-3, 0x1.63b5320000000p-4, -0x1.7209fa0000000p-2, 0x1.ce26fa0000000p-3, -0x1.f420c60000000p-4, -0x1.4d8cba0000000p-5, 0x1.3a07b00000000p-4, 0x1.a246b40000000p-2, -0x1.2955060000000p-1, 0x1.b310a40000000p-3}
}
, {{0x1.06041e0000000p-2, 0x1.83fd4a0000000p-2, -0x1.8bad1a0000000p-3, 0x1.5acb980000000p-3, 0x1.38428a0000000p-3, 0x1.db2dd20000000p-3, 0x1.1d19e60000000p-4, -0x1.d210fa0000000p-6, 0x1.23642c0000000p-2, 0x1.e4f95a0000000p-3, 0x1.47e9280000000p-3, -0x1.5d2d800000000p-3, 0x1.569af80000000p-3, 0x1.9081fa0000000p-2, 0x1.41afa60000000p-3, 0x1.792ef60000000p-2}
, {0x1.a28a6c0000000p-4, -0x1.d4a5ea0000000p-4, -0x1.09f0c60000000p-2, -0x1.7b82dc0000000p-2, -0x1.6621f60000000p-2, 0x1.885e060000000p-4, 0x1.0469720000000p-6, -0x1.882e620000000p-3, -0x1.44066a0000000p-4, -0x1.4488c80000000p-3, -0x1.b91aca0000000p-2, -0x1.9204b80000000p-3, -0x1.2eab9a0000000p-2, 0x1.f29a240000000p-10, -0x1.759c660000000p-4, 0x1.9b3da00000000p-4}
, {-0x1.6dc28c0000000p-3, -0x1.ee96cc0000000p-3, -0x1.610cf20000000p-3, -0x1.3708520000000p-3, 0x1.f374f60000000p-8, 0x1.48790c0000000p-2, -0x1.0711f00000000p-2, -0x1.755f2a0000000p-4, -0x1.79f7880000000p-3, -0x1.0cd3f40000000p-2, -0x1.d9aece0000000p-3, -0x1.3609be0000000p-2, -0x1.a5472c0000000p-2, 0x1.8287f00000000p-4, -0x1.f9f3860000000p-2, -0x1.01e4de0000000p-6}
}
, {{-0x1.a83fb60000000p-2, -0x1.7d4ade0000000p-3, 0x1.d8a3260000000p-3, -0x1.11e9480000000p-2, -0x1.d1462a0000000p-6, 0x1.18034a0000000p-2, -0x1.5105ba0000000p-4, -0x1.7a13520000000p-4, -0x1.61b0500000000p-3, 0x1.f034d40000000p-5, -0x1.f835a00000000p-4, 0x1.e4a2940000000p-9, -0x1.8adc240000000p-3, -0x1.5b8f2e0000000p-4, -0x1.cc32000000000p-3, -0x1.9ad8040000000p-4}
, {0x1.4f7d540000000p-7, -0x1.e3f0940000000p-3, -0x1.8d38040000000p-2, -0x1.bde6aa0000000p-3, -0x1.6c392a0000000p-3, -0x1.a0bc320000000p-3, -0x1.bc05000000000p-3, 0x1.9c2d580000000p-4, -0x1.e71e640000000p-4, 0x1.70ed9c0000000p-5, -0x1.6449640000000p-3, -0x1.c0147c0000000p-2, -0x1.00d6680000000p-6, -0x1.425cea0000000p-5, -0x1.0d971a0000000p-2, 0x1.c912f80000000p-4}
, {-0x1.fbbbb80000000p-2, 0x1.9c68460000000p-3, 0x1.2bb1aa0000000p-2, -0x1.22d7260000000p-2, 0x1.1f14020000000p-3, 0x1.89c6560000000p-3, -0x1.39973c0000000p-3, 0x1.769dbe0000000p-2, -0x1.da90fa0000000p-3, 0x1.282a6a0000000p-3, -0x1.3ac99a0000000p-3, 0x1.569ba80000000p-2, 0x1.e11d940000000p-3, -0x1.6154ec0000000p-3, -0x1.303fa20000000p-1, -0x1.d9618a0000000p-4}
}
, {{0x1.79dc860000000p-2, -0x1.12ce720000000p-2, 0x1.6929360000000p-4, -0x1.d42ab40000000p-3, -0x1.491a1c0000000p-4, 0x1.073ac60000000p-4, 0x1.c15e7c0000000p-5, 0x1.ae5f300000000p-6, -0x1.69ad8a0000000p-3, -0x1.c282640000000p-2, -0x1.5c34440000000p-4, -0x1.0731760000000p-8, -0x1.15fdc60000000p-3, -0x1.45015c0000000p-2, 0x1.09a4000000000p-3, 0x1.b756720000000p-4}
, {-0x1.40df720000000p-3, 0x1.b467ca0000000p-4, -0x1.2bd5640000000p-3, -0x1.9894f20000000p-3, -0x1.1854e20000000p-6, -0x1.3fed760000000p-2, -0x1.094b8a0000000p-6, -0x1.2391fc0000000p-2, -0x1.4b88c40000000p-7, -0x1.0d316c0000000p-3, -0x1.0a8eca0000000p-2, 0x1.b847ca0000000p-3, 0x1.1c6fae0000000p-5, 0x1.cc776a0000000p-4, -0x1.cd3efa0000000p-4, 0x1.6e6efa0000000p-3}
, {-0x1.3b6dea0000000p-2, -0x1.7476560000000p-3, 0x1.2c3aac0000000p-4, -0x1.4e8df40000000p-4, -0x1.b5c87a0000000p-4, 0x1.2e14d00000000p-2, -0x1.09f7940000000p-2, 0x1.520cce0000000p-3, -0x1.5382940000000p-2, 0x1.27b3ee0000000p-3, 0x1.aec9420000000p-4, 0x1.73bc7a0000000p-2, -0x1.6359320000000p-5, 0x1.7712260000000p-3, -0x1.1aaeca0000000p-4, 0x1.68c55a0000000p-2}
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

typedef float batch_normalization_1_output_type[21][32];

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
#define WEIGHTS_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define NUMBER_T float
#define LONG_NUMBER_T float


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

const float batch_normalization_1_bias[32] = {-0x1.467d780000000p-1, -0x1.cfec740000000p-1, -0x1.450ea60000000p-1, -0x1.b5dfec0000000p-3, -0x1.990a220000000p-2, -0x1.c48bc20000000p-2, -0x1.bfd53a0000000p-2, -0x1.67014e0000000p-1, -0x1.34c9d00000000p-2, -0x1.fcaec00000000p-2, -0x1.7d6dcc0000000p-2, -0x1.1775720000000p-1, -0x1.b591280000000p-3, -0x1.ce7ada0000000p-2, -0x1.dcd5a00000000p-2, -0x1.ec0c180000000p-4, -0x1.5ee7420000000p-1, -0x1.5815f60000000p-1, -0x1.1e08ac0000000p-1, -0x1.f704360000000p-2, -0x1.bcc7700000000p-1, -0x1.c705cc0000000p-2, -0x1.b3e9720000000p-2, -0x1.560eaa0000000p-1, -0x1.665b680000000p-2, -0x1.1b8ff40000000p-1, -0x1.2555c00000000p-4, -0x1.47713a0000000p-2, -0x1.770b860000000p-1, -0x1.51f23e0000000p-1, -0x1.491ae20000000p-1, -0x1.567cd20000000p-2}
;
const float batch_normalization_1_kernel[32] = {0x1.5c83ac0000000p-1, 0x1.b7292e0000000p+0, 0x1.c528a80000000p+0, 0x1.7f1af40000000p+0, 0x1.3374140000000p-1, 0x1.ae42ea0000000p+0, 0x1.1af9940000000p-1, 0x1.a324360000000p-1, 0x1.9bd8e80000000p+0, 0x1.af32040000000p+0, 0x1.57ee200000000p+0, 0x1.0d255e0000000p+0, 0x1.ea107e0000000p-1, 0x1.0de0840000000p+0, 0x1.328c120000000p+0, 0x1.ec8ffa0000000p-1, 0x1.d80e7a0000000p+0, 0x1.19912c0000000p-1, 0x1.2235ec0000000p+0, 0x1.2b77040000000p+1, 0x1.dc49340000000p-1, 0x1.1f86c60000000p+0, 0x1.cfc3d00000000p-1, 0x1.cf914a0000000p+1, 0x1.0922a80000000p+0, 0x1.9ca6e00000000p+0, 0x1.1e56de0000000p-1, 0x1.e44f780000000p-1, 0x1.e57be00000000p-1, 0x1.c882540000000p+0, 0x1.2381a80000000p+0, 0x1.1bcfba0000000p+0}
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

typedef float max_pooling1d_1_output_type[POOL_LENGTH][INPUT_CHANNELS];

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
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define NUMBER_T float
#define LONG_NUMBER_T float


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

typedef float conv1d_2_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

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
#define WEIGHTS_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define NUMBER_T float
#define LONG_NUMBER_T float


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


#error "Data type unsupported by CMSIS-NN"

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


const float  conv1d_2_bias[CONV_FILTERS] = {0x1.c462380000000p-6, -0x1.6b480a0000000p-3, -0x1.2cf7f20000000p-3, 0x1.62855c0000000p-3, 0x1.0331560000000p-3, 0x1.96707e0000000p-3, 0x1.a0df760000000p-6, 0x1.9e50580000000p-5, -0x1.b6a3780000000p-4, -0x1.bf47b60000000p-4, 0x1.ab0d500000000p-5, -0x1.f6e81a0000000p-5, 0x1.cdee720000000p-4, 0x1.23b4900000000p-3, 0x1.4aad420000000p-4, 0x1.60670c0000000p-5, 0x1.8d112e0000000p-8, -0x1.2c5a2c0000000p-5, 0x1.056f660000000p-4, -0x1.99a2100000000p-4, 0x1.3db7260000000p-5, -0x1.f148e40000000p-5, 0x1.5cd8040000000p-4, 0x1.301bb60000000p-3, 0x1.9fecf00000000p-3, 0x1.27fb640000000p-3, 0x1.7a931e0000000p-4, 0x1.a5c2f40000000p-5, 0x1.537d5a0000000p-3, 0x1.5a15420000000p-3, 0x1.0d04b80000000p-9, -0x1.9d2e060000000p-3, -0x1.cd293a0000000p-7, 0x1.a4d0bc0000000p-4, 0x1.a22aa00000000p-4, 0x1.db77160000000p-6, -0x1.32d9d20000000p-6, 0x1.b1e7b20000000p-4, 0x1.e597740000000p-4, 0x1.058d0a0000000p-4, -0x1.6c24460000000p-3, 0x1.9955200000000p-3, 0x1.b99d640000000p-3, 0x1.906a220000000p-5, -0x1.4e5c840000000p-5, 0x1.dc9b3a0000000p-4, -0x1.9354040000000p-9, -0x1.238d9a0000000p-3, 0x1.320b8c0000000p-3, -0x1.923eae0000000p-9, -0x1.40a11c0000000p-3, 0x1.b643140000000p-8, 0x1.fafe700000000p-4, 0x1.2e33560000000p-3, 0x1.8bcc280000000p-3, -0x1.fa1f120000000p-4, 0x1.55588a0000000p-8, -0x1.0b9e400000000p-3, -0x1.5f99a40000000p-5, 0x1.095b500000000p-3, -0x1.45796e0000000p-3, -0x1.90508c0000000p-7, -0x1.b291e60000000p-5, 0x1.7da1300000000p-4}
;

const float  conv1d_2_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS] = {{{0x1.865a440000000p-4, 0x1.38d12c0000000p-4, -0x1.70c9aa0000000p-3, 0x1.eb5ae60000000p-4, -0x1.a1eda40000000p-4, -0x1.b349fc0000000p-4, 0x1.05891c0000000p-5, -0x1.b4a53c0000000p-3, 0x1.83e0ca0000000p-4, -0x1.819a780000000p-5, -0x1.97ede00000000p-6, -0x1.b92bc60000000p-3, 0x1.114f4e0000000p-3, 0x1.53fb940000000p-5, -0x1.1ab6220000000p-3, -0x1.9c25f80000000p-4, -0x1.c179ac0000000p-6, -0x1.7276ce0000000p-5, -0x1.5096f40000000p-3, -0x1.cc2fe20000000p-5, -0x1.2c07660000000p-2, 0x1.85db100000000p-6, -0x1.d5a8e40000000p-4, 0x1.5917420000000p-6, -0x1.23f8120000000p-5, -0x1.2f044a0000000p-4, -0x1.886dfe0000000p-3, 0x1.4ae4b60000000p-4, -0x1.4312280000000p-2, -0x1.c8feee0000000p-4, -0x1.ef31b00000000p-5, -0x1.76b3120000000p-4}
, {-0x1.d0eebe0000000p-4, -0x1.647c0e0000000p-5, -0x1.676e900000000p-4, -0x1.b519fe0000000p-6, 0x1.068c900000000p-6, -0x1.ae53780000000p-4, -0x1.7bfdf60000000p-7, 0x1.1ec1b20000000p-2, 0x1.9558ec0000000p-5, -0x1.1126100000000p-5, -0x1.7f39bc0000000p-7, -0x1.2e6d4e0000000p-3, -0x1.453a000000000p-4, 0x1.6826180000000p-5, -0x1.34ec1c0000000p-3, 0x1.1738f40000000p-3, -0x1.f2c16a0000000p-6, -0x1.93da480000000p-3, 0x1.da80e00000000p-4, 0x1.c48db40000000p-3, -0x1.56dfa60000000p-3, 0x1.9434e40000000p-3, 0x1.3eff8c0000000p-4, 0x1.52343e0000000p-4, -0x1.b9211e0000000p-5, -0x1.7ee4520000000p-5, -0x1.f414040000000p-4, 0x1.9b9ec00000000p-5, 0x1.592c5c0000000p-4, -0x1.e3ceae0000000p-4, 0x1.97e6bc0000000p-4, -0x1.9dc47e0000000p-5}
, {-0x1.4c12120000000p-3, -0x1.8b60480000000p-8, 0x1.5427960000000p-3, -0x1.9abf060000000p-3, -0x1.289c160000000p-5, 0x1.401cd60000000p-5, -0x1.99170a0000000p-3, 0x1.8ba12c0000000p-4, 0x1.75d02c0000000p-5, -0x1.701f5c0000000p-4, -0x1.70a9420000000p-6, -0x1.020c6a0000000p-3, 0x1.5c36cc0000000p-7, 0x1.ba63120000000p-4, -0x1.04e3c40000000p-4, -0x1.59c7040000000p-7, -0x1.1e823c0000000p-4, -0x1.aa8a7e0000000p-5, 0x1.5c21aa0000000p-5, 0x1.35c13e0000000p-2, -0x1.7c81ee0000000p-3, -0x1.0e93fa0000000p-3, 0x1.13d65a0000000p-8, -0x1.2072fe0000000p-5, 0x1.46090a0000000p-3, -0x1.8b429a0000000p-7, -0x1.8c45020000000p-4, -0x1.093c420000000p-4, 0x1.8a5fa40000000p-5, 0x1.5a88080000000p-9, -0x1.29a7920000000p-3, 0x1.a193ac0000000p-4}
}
, {{-0x1.e33dc20000000p-6, 0x1.5864400000000p-4, -0x1.48956e0000000p-3, 0x1.4336500000000p-9, -0x1.acbb020000000p-3, -0x1.08b4ea0000000p-3, -0x1.45e7380000000p-3, -0x1.5586ea0000000p-3, -0x1.6eb1e80000000p-6, 0x1.a35b260000000p-4, -0x1.aa215a0000000p-4, -0x1.b344e60000000p-2, -0x1.e388b60000000p-7, -0x1.07031c0000000p-5, -0x1.bc75700000000p-3, -0x1.7049560000000p-2, -0x1.91cc0a0000000p-5, -0x1.b3b1e40000000p-7, 0x1.115e8e0000000p-5, -0x1.c5ea4e0000000p-4, -0x1.926e620000000p-3, -0x1.6815960000000p-3, -0x1.17e1360000000p-2, 0x1.657c240000000p-3, -0x1.02cb7c0000000p-8, 0x1.a901bc0000000p-5, -0x1.62a2020000000p-5, -0x1.5de3a80000000p-2, -0x1.84aa380000000p-6, -0x1.40d0a80000000p-3, 0x1.317c360000000p-3, 0x1.943d9e0000000p-5}
, {0x1.3f5e840000000p-5, 0x1.115f780000000p-3, -0x1.65bfdc0000000p-4, 0x1.3ba5940000000p-4, 0x1.97f0240000000p-4, 0x1.6d75680000000p-3, -0x1.1b9d320000000p-3, -0x1.ceb80c0000000p-4, 0x1.55ae1c0000000p-3, 0x1.da8cd00000000p-5, -0x1.c7170c0000000p-3, -0x1.0af16a0000000p-2, 0x1.637a8c0000000p-7, 0x1.52296a0000000p-3, -0x1.c2c47a0000000p-6, 0x1.19b79c0000000p-6, -0x1.9c15a40000000p-3, -0x1.6a95280000000p-5, -0x1.26224a0000000p-2, 0x1.2b37fc0000000p-4, -0x1.32261e0000000p-5, 0x1.2bd5160000000p-6, -0x1.39cf9c0000000p-3, 0x1.02c2e80000000p-7, -0x1.927b0e0000000p-7, 0x1.657c540000000p-4, -0x1.9749f60000000p-4, -0x1.64a0960000000p-5, -0x1.c911680000000p-4, -0x1.6140e00000000p-3, 0x1.13950e0000000p-6, 0x1.46e0e20000000p-3}
, {0x1.ad7a620000000p-5, -0x1.56cbb20000000p-7, -0x1.0f5e640000000p-3, -0x1.cda6280000000p-4, 0x1.b72c020000000p-4, 0x1.47f8280000000p-4, -0x1.0b7e240000000p-3, -0x1.accb560000000p-9, 0x1.e1a7540000000p-6, -0x1.a50f860000000p-5, -0x1.6738040000000p-3, 0x1.cb31820000000p-4, -0x1.a3e2c00000000p-6, 0x1.1142f20000000p-4, -0x1.0a54340000000p-5, 0x1.1afee80000000p-2, -0x1.11793a0000000p-3, -0x1.0509440000000p-7, 0x1.0985060000000p-10, -0x1.0dcce00000000p-5, 0x1.20e1720000000p-5, 0x1.a5d0b40000000p-3, 0x1.baa4fa0000000p-6, -0x1.c2d5320000000p-5, 0x1.1895b80000000p-7, 0x1.5b681c0000000p-3, -0x1.f7dee00000000p-4, 0x1.1c81440000000p-6, -0x1.cbf3f40000000p-5, -0x1.960fa00000000p-3, 0x1.4bba480000000p-3, 0x1.ce49400000000p-3}
}
, {{-0x1.a679e40000000p-2, 0x1.54c7bc0000000p-6, 0x1.d4654a0000000p-3, 0x1.1ddabe0000000p-4, 0x1.6ad1480000000p-5, -0x1.9c845c0000000p-3, -0x1.6688380000000p-2, -0x1.076df20000000p-4, 0x1.14a6a80000000p-2, -0x1.26e9960000000p-2, -0x1.7b9f0a0000000p-2, -0x1.6b7fee0000000p-2, -0x1.a98cf80000000p-4, 0x1.862fdc0000000p-5, -0x1.95fb480000000p-5, -0x1.412a7c0000000p-4, -0x1.fe35800000000p-2, -0x1.f7e97e0000000p-5, 0x1.98d3d20000000p-5, 0x1.1c67820000000p-2, -0x1.243b0c0000000p-3, 0x1.2fcb220000000p-4, -0x1.c98fdc0000000p-5, -0x1.6b16a80000000p-5, 0x1.b6684a0000000p-3, 0x1.df91800000000p-4, -0x1.73f3ce0000000p-2, -0x1.ecb3d40000000p-9, 0x1.c791680000000p-4, -0x1.359ffe0000000p-2, 0x1.5ffef20000000p-4, 0x1.4a3bd80000000p-4}
, {-0x1.51040a0000000p-3, 0x1.0f383e0000000p-3, -0x1.f322300000000p-4, -0x1.ff34a80000000p-3, -0x1.3af4000000000p-4, 0x1.fd6dd20000000p-4, -0x1.25d3360000000p-3, -0x1.ae81380000000p-7, 0x1.03336a0000000p-2, -0x1.cfdcb00000000p-4, -0x1.1aef8e0000000p-6, 0x1.7194460000000p-4, -0x1.c44fe40000000p-3, 0x1.1105960000000p-2, -0x1.099d300000000p-2, -0x1.2843d20000000p-4, -0x1.fdf6a80000000p-4, 0x1.2fe3280000000p-3, -0x1.9f837c0000000p-2, -0x1.0a79f60000000p-5, 0x1.7937d80000000p-6, 0x1.302f440000000p-6, -0x1.0fa1da0000000p-3, -0x1.c308660000000p-4, 0x1.0179b00000000p-3, 0x1.0853b20000000p-2, -0x1.edf0580000000p-8, 0x1.e534640000000p-3, -0x1.76cabe0000000p-4, -0x1.269ad40000000p-3, -0x1.4dd4920000000p-4, 0x1.8e7a7e0000000p-3}
, {0x1.2e444e0000000p-5, 0x1.1f87140000000p-4, 0x1.0a226c0000000p-5, 0x1.c498180000000p-3, 0x1.4dcd6c0000000p-2, -0x1.d5af4e0000000p-3, 0x1.51b7ec0000000p-6, 0x1.e8f2f00000000p-8, -0x1.1dad940000000p-9, -0x1.fe7d9a0000000p-6, 0x1.032c2e0000000p-3, 0x1.28cdfc0000000p-2, -0x1.90ce4e0000000p-3, -0x1.64ca140000000p-9, -0x1.dede320000000p-4, 0x1.54c0680000000p-5, 0x1.0b9b660000000p-8, -0x1.3d02e60000000p-3, 0x1.3d9e140000000p-3, 0x1.877f940000000p-5, 0x1.e6b0820000000p-5, 0x1.44a01a0000000p-4, 0x1.58ba5c0000000p-7, -0x1.e88d920000000p-4, 0x1.9da4ac0000000p-5, 0x1.b22cb20000000p-4, 0x1.46490e0000000p-3, -0x1.693fd40000000p-6, 0x1.16cec40000000p-3, -0x1.048a920000000p-3, 0x1.98fd620000000p-4, -0x1.3fde000000000p-6}
}
, {{0x1.fa223a0000000p-3, -0x1.d342600000000p-5, -0x1.249fa40000000p-3, -0x1.dbd9940000000p-4, 0x1.4162640000000p-3, 0x1.70e8540000000p-4, 0x1.4f90c60000000p-4, 0x1.1304580000000p-4, -0x1.1848ee0000000p-5, 0x1.e4bde80000000p-5, -0x1.12dfe00000000p-3, 0x1.6895080000000p-3, -0x1.b9df6e0000000p-6, 0x1.dedc240000000p-5, -0x1.7532e20000000p-4, 0x1.2127fc0000000p-2, -0x1.c4975c0000000p-6, -0x1.2823b40000000p-4, -0x1.b8cf400000000p-4, -0x1.69b47e0000000p-4, 0x1.e19e380000000p-7, 0x1.b859120000000p-3, -0x1.18ab4e0000000p-5, -0x1.245a8c0000000p-3, 0x1.a74be00000000p-7, -0x1.e85a3a0000000p-4, 0x1.7a9b420000000p-4, 0x1.a710460000000p-3, 0x1.80b3a60000000p-4, -0x1.16d6be0000000p-4, -0x1.2b6a880000000p-4, 0x1.ab42720000000p-2}
, {0x1.0b145c0000000p-4, 0x1.0084320000000p-3, -0x1.b8703e0000000p-4, -0x1.4de97e0000000p-8, 0x1.5318a60000000p-5, -0x1.5f360c0000000p-3, 0x1.d155960000000p-5, 0x1.ba82120000000p-4, -0x1.37f01e0000000p-5, 0x1.ad97d40000000p-4, -0x1.424f960000000p-4, -0x1.19f7fc0000000p-4, 0x1.15be340000000p-3, 0x1.f9e3fa0000000p-4, -0x1.40cc800000000p-3, 0x1.f6e0fa0000000p-4, -0x1.17beae0000000p-3, 0x1.254fe40000000p-4, -0x1.8c7bfa0000000p-6, -0x1.d11d760000000p-4, -0x1.8655e20000000p-5, 0x1.8261c80000000p-5, -0x1.a46b400000000p-6, -0x1.5cbf280000000p-3, -0x1.aaed100000000p-3, -0x1.062ad20000000p-4, 0x1.3ebb700000000p-4, 0x1.99c91c0000000p-3, 0x1.0a3bd00000000p-4, -0x1.3657880000000p-3, -0x1.3864440000000p-9, -0x1.0d4bbe0000000p-3}
, {-0x1.e086d80000000p-2, -0x1.0483d60000000p-3, 0x1.790f460000000p-6, -0x1.4ce6060000000p-3, -0x1.9c53740000000p-2, -0x1.59ebc00000000p-4, -0x1.0a3c1a0000000p-3, -0x1.0f9b7e0000000p-2, 0x1.b414900000000p-4, 0x1.d110c80000000p-8, 0x1.7a66640000000p-3, -0x1.98783e0000000p-2, -0x1.2d5d760000000p-3, -0x1.4f42780000000p-5, -0x1.b087120000000p-3, -0x1.8be01a0000000p-4, -0x1.dad40c0000000p-4, -0x1.41cb220000000p-2, -0x1.7f760e0000000p-4, 0x1.9070ce0000000p-3, -0x1.7b537a0000000p-4, 0x1.efa8ca0000000p-6, -0x1.9423ac0000000p-5, -0x1.00245c0000000p-8, 0x1.48ee940000000p-3, 0x1.c31fe00000000p-3, -0x1.d840ca0000000p-2, -0x1.ee86b40000000p-6, -0x1.1564780000000p-3, 0x1.23b4ea0000000p-2, 0x1.63416e0000000p-5, -0x1.693d160000000p-2}
}
, {{-0x1.649d020000000p-4, -0x1.6914ae0000000p-5, -0x1.b7c70c0000000p-4, 0x1.5962e20000000p-8, 0x1.251cce0000000p-6, 0x1.208c320000000p-3, -0x1.986fa00000000p-6, -0x1.bbd1140000000p-5, 0x1.77e0980000000p-4, -0x1.a7dd300000000p-4, 0x1.17af7a0000000p-3, 0x1.87d70e0000000p-10, -0x1.036c660000000p-4, -0x1.9f60060000000p-4, -0x1.454f780000000p-3, 0x1.9babb60000000p-4, 0x1.55af680000000p-4, -0x1.33aa2e0000000p-4, 0x1.94b9c20000000p-3, -0x1.10bd7e0000000p-8, 0x1.ca7fac0000000p-4, -0x1.0ae2660000000p-4, 0x1.26eab80000000p-2, -0x1.9485d80000000p-4, -0x1.0c1f740000000p-4, -0x1.2d02620000000p-3, 0x1.5390da0000000p-4, 0x1.ce68840000000p-4, 0x1.0bc98e0000000p-3, 0x1.1b3f220000000p-2, -0x1.b31df20000000p-5, 0x1.2fae3c0000000p-4}
, {0x1.37ea340000000p-3, -0x1.666fca0000000p-4, -0x1.cfe6d40000000p-7, -0x1.18763c0000000p-7, -0x1.cc27660000000p-4, -0x1.6b7cf00000000p-4, -0x1.223f1c0000000p-4, 0x1.067a360000000p-4, 0x1.c1a4280000000p-8, -0x1.41c7160000000p-9, -0x1.16eac60000000p-4, 0x1.0130320000000p-3, -0x1.fdec2c0000000p-5, -0x1.a3b74c0000000p-4, -0x1.35b8660000000p-5, -0x1.11650a0000000p-3, -0x1.caf1fa0000000p-4, -0x1.2b7c020000000p-4, 0x1.409d7c0000000p-6, 0x1.1a10fa0000000p-4, -0x1.f55ebe0000000p-4, 0x1.583dd00000000p-4, -0x1.a625be0000000p-4, 0x1.89ecec0000000p-5, -0x1.0c224c0000000p-8, 0x1.dd11c00000000p-6, -0x1.94aa100000000p-4, 0x1.2dae0c0000000p-6, 0x1.d0faba0000000p-4, -0x1.b23f640000000p-6, 0x1.137bce0000000p-5, -0x1.fcb2440000000p-5}
, {0x1.132b260000000p-3, -0x1.2d603c0000000p-4, 0x1.12f8240000000p-2, 0x1.9523ae0000000p-4, -0x1.9b4cf80000000p-3, 0x1.555bd60000000p-2, -0x1.6f45620000000p-3, -0x1.471fb60000000p-3, -0x1.cc2edc0000000p-4, 0x1.9c0b2a0000000p-3, -0x1.abb56c0000000p-3, -0x1.6636700000000p-4, 0x1.6d24640000000p-3, 0x1.fa1a880000000p-6, -0x1.1051440000000p-6, 0x1.3f05340000000p-2, 0x1.8952f00000000p-5, -0x1.ada4480000000p-3, -0x1.fbcfc60000000p-4, 0x1.6d39be0000000p-4, 0x1.02793e0000000p-7, 0x1.47db4a0000000p-4, -0x1.c172920000000p-4, 0x1.8f38fe0000000p-4, 0x1.78a0920000000p-4, 0x1.732ef60000000p-4, -0x1.30e26a0000000p-3, -0x1.0464ea0000000p-3, -0x1.806fb80000000p-3, 0x1.1d63060000000p-2, -0x1.d40a7e0000000p-4, 0x1.4beb340000000p-3}
}
, {{-0x1.7e995c0000000p-4, 0x1.ba63940000000p-4, 0x1.9bcf4a0000000p-3, -0x1.1ce25e0000000p-3, -0x1.9ee4fe0000000p-3, 0x1.e2942a0000000p-5, -0x1.fd14360000000p-3, -0x1.3001d60000000p-3, -0x1.e1e7de0000000p-2, 0x1.e07ca80000000p-2, -0x1.b61da40000000p-3, -0x1.6e24180000000p-4, 0x1.030b5c0000000p-4, -0x1.9bcc020000000p-3, -0x1.88f7760000000p-6, -0x1.1f7a480000000p-3, 0x1.03ca020000000p-5, 0x1.f892780000000p-9, -0x1.e836360000000p-2, -0x1.b4e3a60000000p-4, -0x1.4479ce0000000p-7, -0x1.82b19c0000000p-2, -0x1.b14d5c0000000p-3, 0x1.0e3e480000000p-6, -0x1.11fc2a0000000p-2, 0x1.1000340000000p-2, -0x1.e28a7e0000000p-5, -0x1.68afcc0000000p-4, -0x1.64c7be0000000p-2, -0x1.173b6e0000000p-3, -0x1.44301c0000000p-4, -0x1.cd1c500000000p-7}
, {-0x1.eac0f80000000p-3, -0x1.783a180000000p-4, -0x1.0b76140000000p-4, -0x1.0efe3e0000000p-2, -0x1.5d5a740000000p-3, -0x1.a51e900000000p-6, -0x1.9296be0000000p-3, -0x1.d696120000000p-6, -0x1.07874e0000000p-2, 0x1.1b0e740000000p-3, -0x1.d744560000000p-3, 0x1.1267800000000p-4, -0x1.e3dc360000000p-7, 0x1.e314900000000p-7, 0x1.28cbc60000000p-4, 0x1.fe93ec0000000p-4, -0x1.bf385e0000000p-4, 0x1.dfec900000000p-5, 0x1.599a900000000p-6, -0x1.2d01d40000000p-4, 0x1.1996340000000p-2, -0x1.84ef440000000p-3, 0x1.7f733e0000000p-3, -0x1.1c7c860000000p-5, -0x1.6bcbce0000000p-3, 0x1.999c980000000p-4, -0x1.305c600000000p-4, 0x1.d878920000000p-5, -0x1.ad3d340000000p-6, 0x1.680aa20000000p-3, -0x1.1cabfa0000000p-3, 0x1.0a2fbc0000000p-2}
, {0x1.3f393e0000000p-4, -0x1.0386200000000p-4, -0x1.9e75060000000p-6, 0x1.13fc620000000p-7, 0x1.5b75280000000p-4, -0x1.0eecd60000000p-7, -0x1.e8dff80000000p-5, 0x1.ef9b060000000p-5, -0x1.2f6dd40000000p-9, -0x1.2b06d40000000p-6, 0x1.df787c0000000p-5, 0x1.09caa80000000p-5, -0x1.53ca8c0000000p-5, 0x1.8701380000000p-4, -0x1.7815360000000p-7, 0x1.3b42b80000000p-6, -0x1.1bc2b20000000p-2, 0x1.c8e0d00000000p-6, 0x1.19df400000000p-6, -0x1.0d56ce0000000p-3, -0x1.570fc60000000p-4, -0x1.513dee0000000p-3, 0x1.375b900000000p-3, -0x1.3da7180000000p-2, 0x1.0ad02a0000000p-2, 0x1.85cef80000000p-3, 0x1.5d143e0000000p-2, 0x1.00f4620000000p-3, 0x1.1de7b80000000p-2, -0x1.7f17020000000p-6, 0x1.e8860e0000000p-3, -0x1.23284c0000000p-3}
}
, {{0x1.541ba60000000p-4, -0x1.397e020000000p-3, 0x1.f6af9c0000000p-4, 0x1.30a3420000000p-4, 0x1.ca88620000000p-5, -0x1.8b076a0000000p-4, -0x1.015e300000000p-5, -0x1.f296120000000p-6, -0x1.93a5920000000p-5, -0x1.9753780000000p-3, -0x1.a50f8c0000000p-4, 0x1.bfe2040000000p-4, 0x1.165a8c0000000p-4, -0x1.4288080000000p-4, -0x1.68c5200000000p-3, 0x1.0a1fc00000000p-6, -0x1.72e4540000000p-3, 0x1.a7efae0000000p-6, -0x1.ea4eea0000000p-5, -0x1.2fc53c0000000p-3, 0x1.0b28f40000000p-4, -0x1.2e21520000000p-3, 0x1.73b6de0000000p-4, -0x1.437cc60000000p-3, 0x1.22f6ca0000000p-3, -0x1.297ad60000000p-4, 0x1.d19a020000000p-5, 0x1.0c74920000000p-5, 0x1.1fccde0000000p-3, -0x1.6ed00c0000000p-6, 0x1.df1c3c0000000p-5, 0x1.40cc4a0000000p-5}
, {-0x1.04fa340000000p-3, -0x1.0dae580000000p-3, 0x1.b16ed00000000p-4, -0x1.0487020000000p-4, 0x1.246b380000000p-4, -0x1.75ec900000000p-4, 0x1.05ea140000000p-6, 0x1.76614a0000000p-3, 0x1.e41d0c0000000p-4, 0x1.0c14960000000p-4, 0x1.3a24e80000000p-5, 0x1.110e3c0000000p-2, -0x1.510a540000000p-4, -0x1.7b764a0000000p-4, -0x1.b4f0c20000000p-3, 0x1.bc31aa0000000p-4, -0x1.133d360000000p-2, -0x1.4616e20000000p-6, 0x1.075ce00000000p-5, -0x1.06f3980000000p-6, 0x1.1a59660000000p-4, -0x1.d9250a0000000p-4, 0x1.5baf480000000p-5, 0x1.bf6c1a0000000p-7, -0x1.d5269a0000000p-4, 0x1.98bee00000000p-5, 0x1.7f763e0000000p-3, -0x1.afa1920000000p-4, 0x1.8aff720000000p-3, -0x1.87eb400000000p-4, -0x1.024f480000000p-4, 0x1.28ee2a0000000p-4}
, {-0x1.c3bf980000000p-2, 0x1.2c73940000000p-5, 0x1.6941a40000000p-3, -0x1.c532d40000000p-5, -0x1.9801680000000p-4, -0x1.21f6ee0000000p-5, -0x1.72e77a0000000p-4, -0x1.aa594e0000000p-5, 0x1.58d2ce0000000p-2, -0x1.3963ce0000000p-4, 0x1.f3e9320000000p-8, -0x1.36ba780000000p-3, -0x1.dde9bc0000000p-5, 0x1.c451100000000p-4, -0x1.8714140000000p-2, -0x1.0abec80000000p-3, 0x1.6a3f800000000p-4, -0x1.66ead60000000p-2, 0x1.17f7c40000000p-3, 0x1.7ae2a20000000p-3, -0x1.26612e0000000p-2, -0x1.b485680000000p-3, 0x1.979ac00000000p-7, 0x1.ff92d40000000p-6, -0x1.1ac4540000000p-4, -0x1.edf4240000000p-4, -0x1.4a1e4e0000000p-6, -0x1.6504d40000000p-5, 0x1.6291180000000p-3, 0x1.8890e80000000p-10, -0x1.8886100000000p-4, -0x1.18a8be0000000p-3}
}
, {{-0x1.548ef00000000p-2, 0x1.5fda840000000p-3, 0x1.aac0e40000000p-7, 0x1.3f354a0000000p-6, -0x1.4ae79c0000000p-2, 0x1.abd0020000000p-6, -0x1.04b94c0000000p-3, 0x1.5e23300000000p-3, -0x1.0a4df80000000p-3, -0x1.299d6e0000000p-2, -0x1.6bf19e0000000p-6, -0x1.3fd7940000000p-2, -0x1.285eb60000000p-3, -0x1.5db5080000000p-3, 0x1.92a1040000000p-5, -0x1.a5825c0000000p-3, -0x1.57f3bc0000000p-5, -0x1.0e20e00000000p-6, -0x1.376a980000000p-4, -0x1.1417460000000p-3, 0x1.3376d40000000p-5, -0x1.e22ef40000000p-6, -0x1.53ee940000000p-3, 0x1.264b540000000p-3, -0x1.a380440000000p-2, 0x1.da306c0000000p-4, 0x1.209f1c0000000p-5, 0x1.3b4a680000000p-5, 0x1.0e4e260000000p-6, -0x1.5d885a0000000p-3, 0x1.08fafe0000000p-4, 0x1.6d403e0000000p-3}
, {-0x1.31deea0000000p-7, 0x1.e6b5100000000p-4, 0x1.5cca2a0000000p-8, -0x1.ff98dc0000000p-6, 0x1.a41b5e0000000p-3, -0x1.e542da0000000p-6, -0x1.3ec33a0000000p-4, 0x1.62f8120000000p-5, -0x1.5a8ff20000000p-7, 0x1.4e0ffc0000000p-4, 0x1.0bf0260000000p-5, 0x1.3889180000000p-4, -0x1.10f0a80000000p-8, -0x1.28dd160000000p-5, 0x1.5768600000000p-4, -0x1.ec8ede0000000p-6, 0x1.339a920000000p-2, -0x1.4fb0ae0000000p-4, 0x1.2421e80000000p-8, -0x1.e1f3640000000p-5, -0x1.47d42c0000000p-4, -0x1.009c2a0000000p-2, 0x1.7829d80000000p-4, -0x1.e572440000000p-6, -0x1.633a4e0000000p-3, -0x1.5860b60000000p-6, 0x1.c29c6e0000000p-4, 0x1.79c63e0000000p-5, 0x1.f70f1e0000000p-5, 0x1.484bf60000000p-3, -0x1.9439e00000000p-4, -0x1.05a7f60000000p-3}
, {-0x1.da25500000000p-3, -0x1.05ba200000000p-3, -0x1.18de660000000p-2, 0x1.6dfcb20000000p-6, 0x1.5daa0c0000000p-3, 0x1.0f17b80000000p-3, -0x1.1e2c760000000p-5, -0x1.eac5520000000p-3, 0x1.52a5f00000000p-3, 0x1.5ccf120000000p-5, 0x1.35c5ae0000000p-5, 0x1.2994c00000000p-5, -0x1.b320360000000p-4, 0x1.08b0ba0000000p-5, 0x1.c001f20000000p-7, 0x1.cad6b00000000p-6, 0x1.0480fc0000000p-4, -0x1.e19b2a0000000p-3, 0x1.e8c2620000000p-4, -0x1.82fb920000000p-6, -0x1.af3a6c0000000p-3, -0x1.7b6efa0000000p-5, -0x1.4e616e0000000p-5, 0x1.fdf9100000000p-3, 0x1.d2cfcc0000000p-3, -0x1.fcd0900000000p-5, -0x1.e23b800000000p-4, -0x1.275c860000000p-3, -0x1.747db60000000p-6, -0x1.5264c80000000p-4, 0x1.4c06ba0000000p-3, -0x1.6f454e0000000p-4}
}
, {{0x1.3a58320000000p-5, -0x1.2479220000000p-4, -0x1.1dbab80000000p-2, 0x1.cc8cf40000000p-4, 0x1.ce18b80000000p-4, -0x1.ed117c0000000p-4, 0x1.533b9a0000000p-5, -0x1.dc329a0000000p-7, 0x1.5f81200000000p-3, -0x1.e11eba0000000p-4, 0x1.b24ec20000000p-3, 0x1.119c960000000p-8, 0x1.9d82f40000000p-6, 0x1.370b220000000p-2, -0x1.b25fb00000000p-3, 0x1.e033820000000p-3, -0x1.9eef700000000p-2, -0x1.593ee60000000p-4, -0x1.599d5c0000000p-7, 0x1.a254f60000000p-2, -0x1.6cf3560000000p-7, 0x1.0920b00000000p-1, -0x1.a17a400000000p-5, 0x1.89c3300000000p-7, 0x1.26a7000000000p-2, 0x1.b236040000000p-3, 0x1.68f5500000000p-5, -0x1.8415940000000p-4, 0x1.c637fc0000000p-6, -0x1.9a138e0000000p-3, -0x1.9d7c7e0000000p-6, 0x1.dcf3de0000000p-3}
, {-0x1.695c9e0000000p-5, 0x1.6dce440000000p-3, 0x1.ce8a3a0000000p-4, 0x1.5d94020000000p-4, 0x1.3952840000000p-6, -0x1.c18e2a0000000p-3, -0x1.dbff580000000p-4, 0x1.b42f9c0000000p-3, -0x1.e6fcec0000000p-4, -0x1.3cb9d40000000p-4, -0x1.1eaacc0000000p-2, 0x1.f670980000000p-4, 0x1.e13a080000000p-8, -0x1.28756c0000000p-3, -0x1.66b01a0000000p-3, -0x1.9bc7a40000000p-3, -0x1.bc61500000000p-4, 0x1.a9d5740000000p-5, 0x1.1ee29e0000000p-4, 0x1.dd77c40000000p-3, -0x1.c2071e0000000p-3, 0x1.100b1a0000000p-3, -0x1.512c2e0000000p-2, 0x1.f87d300000000p-4, 0x1.5f3dec0000000p-5, -0x1.2e4bf20000000p-3, -0x1.0f5d780000000p-4, 0x1.92ccb00000000p-4, -0x1.c0f2000000000p-4, 0x1.9b1e180000000p-8, -0x1.db82180000000p-3, 0x1.1fda280000000p-4}
, {-0x1.88289a0000000p-3, 0x1.5548060000000p-3, -0x1.2073920000000p-3, -0x1.2aee540000000p-3, 0x1.a50e400000000p-4, -0x1.e145400000000p-4, -0x1.0191380000000p-3, 0x1.edbf2a0000000p-3, 0x1.22dbf20000000p-3, -0x1.ac0cb00000000p-3, -0x1.4dfe7e0000000p-5, 0x1.51a26c0000000p-3, -0x1.52dc9c0000000p-5, 0x1.0750bc0000000p-5, -0x1.6351400000000p-1, 0x1.661f7a0000000p-3, -0x1.cf1a020000000p-2, -0x1.8e74ba0000000p-3, -0x1.b8eba40000000p-2, 0x1.c49aca0000000p-3, -0x1.256d4e0000000p-4, 0x1.16f1040000000p-2, -0x1.fa95be0000000p-6, 0x1.8981680000000p-5, -0x1.9c231e0000000p-5, -0x1.0e93b00000000p-5, -0x1.3ef1f00000000p-5, 0x1.a31ef20000000p-4, 0x1.1f36720000000p-3, 0x1.0f46c40000000p-3, 0x1.053e180000000p-3, 0x1.3c0d120000000p-3}
}
, {{-0x1.8711220000000p-3, 0x1.02582a0000000p-2, -0x1.006b8a0000000p-4, -0x1.7884900000000p-6, 0x1.8c1d0a0000000p-8, 0x1.6cd14c0000000p-6, -0x1.39543c0000000p-4, 0x1.f6645a0000000p-3, -0x1.5d64f60000000p-5, -0x1.32c6240000000p-5, -0x1.8316000000000p-3, 0x1.2e61e40000000p-4, -0x1.530d620000000p-4, -0x1.0de1660000000p-5, -0x1.e1dffe0000000p-5, 0x1.7828aa0000000p-3, -0x1.1e05240000000p-3, -0x1.51c2ee0000000p-3, 0x1.6a2f660000000p-5, -0x1.6e86400000000p-4, -0x1.5c50d00000000p-4, -0x1.2cadb40000000p-5, -0x1.8a24d40000000p-3, 0x1.3c30f40000000p-4, -0x1.4ba4fe0000000p-3, -0x1.e037580000000p-4, -0x1.1be5c40000000p-4, 0x1.2924f60000000p-2, 0x1.50edb80000000p-3, -0x1.c18cb80000000p-4, 0x1.5b09a20000000p-7, 0x1.bfcf220000000p-3}
, {0x1.e139720000000p-5, -0x1.cc3c2e0000000p-6, 0x1.4a0b720000000p-3, 0x1.54f4260000000p-7, -0x1.a8d1380000000p-4, 0x1.ce01740000000p-3, -0x1.812c0a0000000p-3, -0x1.5f354e0000000p-4, -0x1.2cf71e0000000p-5, -0x1.5bd4020000000p-3, -0x1.e558ea0000000p-5, -0x1.a35c040000000p-5, -0x1.689ee80000000p-4, -0x1.d4b80e0000000p-5, -0x1.c7179e0000000p-4, 0x1.a3141a0000000p-4, -0x1.f3433c0000000p-6, 0x1.36a03e0000000p-3, -0x1.147b520000000p-2, -0x1.15d0d80000000p-3, -0x1.0acb2c0000000p-5, -0x1.c4d3c80000000p-3, 0x1.e001e60000000p-3, -0x1.d38a920000000p-11, -0x1.7810860000000p-3, 0x1.e7bc820000000p-3, -0x1.2e8d8a0000000p-3, 0x1.2cf9520000000p-3, 0x1.9bdb620000000p-4, -0x1.708a960000000p-3, 0x1.48d3aa0000000p-3, -0x1.43f0060000000p-4}
, {-0x1.548df60000000p-5, -0x1.1e062a0000000p-2, -0x1.03e9820000000p-2, 0x1.7e585a0000000p-4, -0x1.ac64c20000000p-4, 0x1.037ac20000000p-4, 0x1.c2e9980000000p-5, -0x1.e729280000000p-2, 0x1.5028680000000p-4, -0x1.b533c20000000p-4, 0x1.fb157a0000000p-3, -0x1.633fea0000000p-2, -0x1.2a665a0000000p-6, 0x1.bf2ab40000000p-3, 0x1.207d0e0000000p-2, -0x1.e34de80000000p-4, 0x1.4f12760000000p-3, 0x1.3e26c20000000p-3, 0x1.48087c0000000p-8, 0x1.4fcefc0000000p-6, 0x1.5e6b6e0000000p-5, -0x1.976ee80000000p-4, -0x1.1ebe160000000p-6, -0x1.1de7cc0000000p-2, 0x1.da9ff80000000p-6, 0x1.f96fa00000000p-4, 0x1.3aa3e80000000p-3, -0x1.ad3bb40000000p-3, -0x1.89fd820000000p-7, -0x1.340a100000000p-3, 0x1.c126300000000p-3, 0x1.e54b880000000p-5}
}
, {{-0x1.6fd7d20000000p-3, 0x1.fb50d00000000p-4, 0x1.dfce2e0000000p-6, 0x1.9653600000000p-3, -0x1.88f7e40000000p-2, -0x1.28dd380000000p-4, -0x1.c248a40000000p-4, 0x1.d5d2740000000p-7, -0x1.4d9c200000000p-3, -0x1.0ee7580000000p-4, 0x1.3c72080000000p-8, -0x1.f962e80000000p-4, 0x1.527b5a0000000p-4, -0x1.1a99fc0000000p-5, -0x1.110ca20000000p-4, -0x1.065c120000000p-8, -0x1.d7c0d00000000p-5, -0x1.a89dd80000000p-4, -0x1.33fcd60000000p-4, -0x1.257acc0000000p-6, -0x1.5df5620000000p-4, 0x1.03c3e00000000p-2, -0x1.cec96e0000000p-3, -0x1.5b80b80000000p-2, -0x1.bd61960000000p-3, -0x1.db999c0000000p-7, 0x1.12286e0000000p-4, -0x1.52351a0000000p-3, -0x1.4487c60000000p-3, 0x1.47a9720000000p-4, -0x1.4bd02a0000000p-9, -0x1.ee1a9a0000000p-6}
, {0x1.c2bf8e0000000p-7, -0x1.8998340000000p-4, 0x1.9eca2a0000000p-5, -0x1.3a0d560000000p-5, -0x1.19a25e0000000p-5, 0x1.8156f00000000p-5, -0x1.4e81420000000p-3, 0x1.4468000000000p-9, -0x1.f3be520000000p-5, 0x1.264a3a0000000p-3, -0x1.c3398e0000000p-3, -0x1.4cdcea0000000p-7, 0x1.1620de0000000p-3, -0x1.53e9fe0000000p-4, 0x1.3267c00000000p-4, 0x1.6d3c2a0000000p-3, 0x1.2e79900000000p-5, 0x1.8e2dea0000000p-3, -0x1.01b6360000000p-4, 0x1.0b4a4c0000000p-7, -0x1.b758c60000000p-5, 0x1.a6fde00000000p-4, 0x1.e250300000000p-3, -0x1.166d580000000p-3, 0x1.8381680000000p-9, 0x1.9be78e0000000p-7, -0x1.1b7ad80000000p-5, -0x1.00f50c0000000p-7, 0x1.85348e0000000p-3, 0x1.7dbbc00000000p-3, 0x1.3b55600000000p-3, 0x1.a057e60000000p-4}
, {-0x1.1b4ff00000000p-1, -0x1.027e660000000p-4, 0x1.18c6f20000000p-3, 0x1.c593740000000p-3, 0x1.cb99ba0000000p-5, -0x1.77591c0000000p-3, -0x1.6b032a0000000p-3, -0x1.0320d60000000p-3, 0x1.f39d140000000p-8, 0x1.087bea0000000p-3, 0x1.703a7e0000000p-3, -0x1.0fb1600000000p-2, -0x1.f34d1a0000000p-8, 0x1.59b6740000000p-3, 0x1.a6a9100000000p-3, 0x1.190a6a0000000p-5, 0x1.e634220000000p-5, -0x1.942a940000000p-5, 0x1.0e5ca80000000p-3, 0x1.15df7e0000000p-5, -0x1.a6e2ba0000000p-8, -0x1.3cdf060000000p-5, -0x1.f929cc0000000p-6, -0x1.5f5f200000000p-5, 0x1.0b1ce40000000p-3, 0x1.e87b0a0000000p-4, 0x1.48247a0000000p-4, 0x1.5fe5380000000p-5, -0x1.c602a80000000p-5, 0x1.c3a5e00000000p-5, -0x1.cff4080000000p-3, -0x1.09f6e60000000p-3}
}
, {{0x1.0c9c500000000p-7, 0x1.5ce9a40000000p-4, 0x1.ea16020000000p-6, -0x1.0140260000000p-3, 0x1.a262560000000p-5, 0x1.0638120000000p-2, -0x1.fc911e0000000p-5, -0x1.0677220000000p-5, 0x1.4f4fcc0000000p-5, -0x1.41dec20000000p-3, -0x1.9eceac0000000p-3, -0x1.1300e60000000p-5, -0x1.25815a0000000p-5, -0x1.4cff800000000p-3, -0x1.c286b40000000p-3, -0x1.c177a60000000p-5, -0x1.54d7f40000000p-5, 0x1.06ec060000000p-4, -0x1.a014a40000000p-5, -0x1.7e0f9a0000000p-3, 0x1.2a89840000000p-4, -0x1.c2d55e0000000p-4, 0x1.3b4b740000000p-3, 0x1.4a3e6c0000000p-3, 0x1.2f4e6c0000000p-3, 0x1.76a20a0000000p-4, -0x1.3a86cc0000000p-4, -0x1.e12d180000000p-5, 0x1.e2da660000000p-3, -0x1.1b3a7e0000000p-3, 0x1.8608000000000p-4, 0x1.aea9f00000000p-4}
, {-0x1.5117ec0000000p-3, -0x1.e5badc0000000p-10, 0x1.164c460000000p-3, -0x1.268b8a0000000p-7, -0x1.4f397e0000000p-3, 0x1.5231780000000p-2, -0x1.58641c0000000p-4, -0x1.b7c4740000000p-2, -0x1.a756d20000000p-6, -0x1.26e56e0000000p-3, -0x1.3662c20000000p-3, -0x1.411ede0000000p-2, -0x1.25d7580000000p-4, -0x1.3329000000000p-5, -0x1.4d6b2c0000000p-5, 0x1.143fbe0000000p-4, -0x1.7d68900000000p-8, 0x1.93b1b80000000p-3, 0x1.f928f60000000p-4, -0x1.5e7d3c0000000p-4, 0x1.29a5f00000000p-3, -0x1.75ca500000000p-5, 0x1.9b3b560000000p-3, 0x1.a173660000000p-6, 0x1.ad13720000000p-3, -0x1.6ab7240000000p-8, -0x1.9970c00000000p-4, -0x1.f002fe0000000p-3, 0x1.1b31bc0000000p-4, 0x1.1126420000000p-3, -0x1.071c220000000p-7, 0x1.5878200000000p-6}
, {-0x1.2313660000000p-2, -0x1.99bf180000000p-4, -0x1.02d43a0000000p-3, 0x1.1e8e180000000p-4, -0x1.8070460000000p-3, -0x1.891be80000000p-5, 0x1.2b08e80000000p-5, -0x1.5d4b040000000p-2, -0x1.48f6ac0000000p-4, -0x1.3fba180000000p-3, -0x1.ae23520000000p-4, -0x1.ab65ea0000000p-5, -0x1.99f5a80000000p-11, 0x1.4ea0980000000p-5, 0x1.d063be0000000p-6, 0x1.72560c0000000p-8, -0x1.0757b40000000p-3, -0x1.783dc40000000p-6, -0x1.19e8fc0000000p-4, 0x1.d20b880000000p-6, -0x1.297cd40000000p-5, -0x1.ed4fba0000000p-6, 0x1.b064a20000000p-3, -0x1.5466100000000p-5, -0x1.37b6520000000p-3, -0x1.7543360000000p-6, -0x1.f64fc80000000p-4, -0x1.17debc0000000p-2, 0x1.61794e0000000p-4, 0x1.2d00c40000000p-7, 0x1.e799ea0000000p-4, 0x1.35dc500000000p-3}
}
, {{-0x1.ec28f60000000p-4, 0x1.6f7f340000000p-5, -0x1.4af43e0000000p-10, 0x1.363cc60000000p-5, -0x1.d5cd1a0000000p-5, 0x1.2ab8860000000p-3, -0x1.22ee140000000p-2, -0x1.baa0dc0000000p-7, 0x1.7fb73e0000000p-3, 0x1.321c1a0000000p-9, -0x1.3f78a40000000p-3, 0x1.409ccc0000000p-4, 0x1.0913de0000000p-3, 0x1.570f8c0000000p-4, -0x1.339e8e0000000p-4, 0x1.bcca180000000p-3, -0x1.829dd40000000p-3, 0x1.cfa2660000000p-10, 0x1.84022a0000000p-4, 0x1.a9dc800000000p-3, -0x1.19ae240000000p-8, -0x1.af84f00000000p-4, 0x1.44a3160000000p-3, -0x1.7c9cf20000000p-4, 0x1.c529760000000p-4, 0x1.0ec03c0000000p-5, -0x1.32d5260000000p-5, -0x1.797ed20000000p-6, 0x1.420f560000000p-2, 0x1.8dc1d00000000p-3, 0x1.eff9b60000000p-3, 0x1.125e4c0000000p-3}
, {-0x1.85479a0000000p-4, 0x1.b38fce0000000p-4, -0x1.a8ef4e0000000p-3, -0x1.5d594c0000000p-4, -0x1.e339620000000p-4, 0x1.6735e20000000p-8, -0x1.9ab88c0000000p-3, -0x1.6a03460000000p-3, 0x1.ae1c980000000p-5, -0x1.caff780000000p-4, -0x1.ddfd900000000p-3, 0x1.e7f96e0000000p-4, -0x1.3221560000000p-3, -0x1.65ce840000000p-4, 0x1.ddd18e0000000p-5, -0x1.e23c5e0000000p-5, -0x1.5b56740000000p-2, 0x1.251aa60000000p-4, -0x1.0409700000000p-8, -0x1.7702da0000000p-3, 0x1.71c6640000000p-6, -0x1.0197aa0000000p-3, -0x1.953eb60000000p-3, 0x1.358cfa0000000p-6, 0x1.271fba0000000p-5, 0x1.b950cc0000000p-5, 0x1.d0705a0000000p-5, -0x1.1a6f7c0000000p-6, -0x1.7dc1d40000000p-5, -0x1.070a0e0000000p-3, -0x1.8ad1a40000000p-5, -0x1.1327c80000000p-3}
, {0x1.cc76540000000p-5, -0x1.2044dc0000000p-5, -0x1.1830f00000000p-3, 0x1.67da9c0000000p-3, 0x1.bb98660000000p-4, 0x1.0403aa0000000p-7, -0x1.fa885a0000000p-3, 0x1.5688ee0000000p-3, -0x1.3b82fc0000000p-5, -0x1.b42a580000000p-5, -0x1.23821c0000000p-4, -0x1.76263e0000000p-6, 0x1.4adc400000000p-4, 0x1.d2a8ee0000000p-3, 0x1.b44dbe0000000p-5, 0x1.f71b040000000p-3, -0x1.4e51ce0000000p-2, 0x1.5b7a980000000p-6, -0x1.1c63820000000p-5, 0x1.338e800000000p-4, 0x1.83a3a20000000p-10, 0x1.991b040000000p-5, -0x1.12263e0000000p-4, -0x1.03f9b80000000p-2, -0x1.26b58e0000000p-3, 0x1.2f971e0000000p-5, 0x1.2a93520000000p-4, -0x1.a949120000000p-4, 0x1.eff2480000000p-6, 0x1.98b0260000000p-4, 0x1.53cf9c0000000p-7, 0x1.248a980000000p-3}
}
, {{0x1.3614ca0000000p-5, -0x1.85d7320000000p-3, 0x1.10a70e0000000p-4, -0x1.49a1b40000000p-3, 0x1.0135b80000000p-3, -0x1.cf9c500000000p-5, -0x1.7999bc0000000p-3, 0x1.9f82460000000p-5, -0x1.58715a0000000p-3, -0x1.505bc00000000p-3, -0x1.db8acc0000000p-5, 0x1.d2b82a0000000p-3, 0x1.b92b880000000p-6, -0x1.5fe7e80000000p-3, -0x1.52e1500000000p-4, 0x1.2c1fde0000000p-5, 0x1.e755620000000p-4, 0x1.2ed9000000000p-3, 0x1.e99c0a0000000p-4, -0x1.c894a80000000p-3, 0x1.afe7f60000000p-4, -0x1.44d48e0000000p-2, 0x1.2957ca0000000p-2, 0x1.b74a7a0000000p-3, -0x1.f1436e0000000p-5, -0x1.bcea660000000p-3, 0x1.30f3b80000000p-3, 0x1.6426e40000000p-3, 0x1.df9a720000000p-4, -0x1.5137a80000000p-3, 0x1.18553a0000000p-4, 0x1.582bbe0000000p-3}
, {0x1.3a2bb00000000p-4, -0x1.6854920000000p-4, 0x1.1699240000000p-4, -0x1.4a1b780000000p-3, -0x1.afa7ba0000000p-3, 0x1.8aff9a0000000p-4, 0x1.283f600000000p-3, 0x1.d1ce260000000p-5, -0x1.725e160000000p-3, 0x1.a01e3c0000000p-5, -0x1.3f6fdc0000000p-10, 0x1.13bf000000000p-3, 0x1.0be7460000000p-4, 0x1.318f7a0000000p-7, 0x1.312e520000000p-4, -0x1.71f4400000000p-5, 0x1.898fd60000000p-4, 0x1.3b95f00000000p-3, -0x1.3617b40000000p-3, -0x1.3513de0000000p-2, 0x1.8c2f7c0000000p-4, -0x1.7408dc0000000p-3, 0x1.fb3c040000000p-4, -0x1.c14dc80000000p-3, -0x1.1f22660000000p-2, -0x1.08a4760000000p-4, -0x1.85e2a20000000p-3, 0x1.176ef60000000p-4, 0x1.46e5460000000p-3, -0x1.6e304e0000000p-9, 0x1.c39e4c0000000p-5, -0x1.cca55e0000000p-5}
, {-0x1.522bbe0000000p-3, -0x1.43223a0000000p-4, 0x1.4d52f00000000p-4, 0x1.5e08480000000p-7, -0x1.7f999e0000000p-3, -0x1.e8f76e0000000p-5, -0x1.f5f9c80000000p-4, -0x1.d3b6580000000p-3, -0x1.8aaf9c0000000p-3, -0x1.380ffe0000000p-3, -0x1.972b0e0000000p-3, -0x1.9462a40000000p-8, 0x1.aa678c0000000p-5, 0x1.f587f00000000p-4, 0x1.0e0af40000000p-6, 0x1.ddf4480000000p-7, -0x1.374dd80000000p-3, -0x1.cefee40000000p-3, 0x1.0999540000000p-4, -0x1.8a6e180000000p-5, 0x1.6d6bd40000000p-7, -0x1.4fc5a60000000p-3, -0x1.ef6dac0000000p-6, 0x1.98795e0000000p-5, -0x1.c49ace0000000p-5, 0x1.2be4ac0000000p-7, -0x1.3678640000000p-2, -0x1.361b6e0000000p-3, 0x1.12a0200000000p-6, 0x1.1855040000000p-6, 0x1.32bd4a0000000p-5, -0x1.0e85e80000000p-2}
}
, {{-0x1.1a604c0000000p-2, 0x1.e8846a0000000p-3, -0x1.ac16f20000000p-4, -0x1.25bdf40000000p-2, 0x1.15e7880000000p-4, 0x1.b599da0000000p-3, -0x1.40a5a00000000p-4, -0x1.50ed480000000p-7, 0x1.fcd4660000000p-3, -0x1.761c2c0000000p-3, -0x1.4702bc0000000p-3, -0x1.6bd0c80000000p-3, 0x1.6de6240000000p-4, 0x1.88b9c40000000p-6, -0x1.0304a20000000p-3, 0x1.054efa0000000p-2, -0x1.fdc2880000000p-6, -0x1.3444620000000p-3, -0x1.ceca2e0000000p-2, 0x1.1190700000000p-2, -0x1.8714a20000000p-5, -0x1.54db8a0000000p-3, -0x1.8441b40000000p-3, -0x1.1037c60000000p-3, -0x1.282ac00000000p-4, 0x1.8b484a0000000p-4, 0x1.56439e0000000p-3, -0x1.10fc2c0000000p-5, -0x1.377f680000000p-4, -0x1.2df3a80000000p-2, -0x1.dc5f960000000p-4, 0x1.012e440000000p-2}
, {-0x1.0af4da0000000p-3, 0x1.5072340000000p-10, 0x1.e7c91c0000000p-4, 0x1.9282760000000p-4, 0x1.574be60000000p-3, -0x1.60cbf00000000p-6, -0x1.1c03a60000000p-3, -0x1.f146d40000000p-8, -0x1.152c6a0000000p-7, 0x1.42445a0000000p-4, -0x1.4c94dc0000000p-3, -0x1.2a12d60000000p-5, -0x1.a67f920000000p-4, 0x1.1159940000000p-4, 0x1.2dc2d00000000p-3, -0x1.9572640000000p-3, 0x1.a82ffc0000000p-4, -0x1.4b12200000000p-2, 0x1.812e9e0000000p-5, 0x1.ae695e0000000p-11, 0x1.6ab2520000000p-6, -0x1.8b077e0000000p-3, -0x1.b9e6da0000000p-4, 0x1.cce4560000000p-5, 0x1.d0961e0000000p-6, 0x1.8fef580000000p-4, -0x1.0a16f40000000p-5, -0x1.a947b60000000p-3, -0x1.2aecfc0000000p-3, 0x1.36da620000000p-9, 0x1.c6bed20000000p-8, 0x1.1bf8f20000000p-4}
, {0x1.2af23a0000000p-4, 0x1.5809d20000000p-5, 0x1.b862340000000p-6, 0x1.39f6b60000000p-4, 0x1.d0ba6a0000000p-4, 0x1.2dcbea0000000p-3, -0x1.cd93b40000000p-3, -0x1.450d5c0000000p-2, 0x1.a5e9620000000p-6, 0x1.116c700000000p-3, -0x1.792e4a0000000p-5, -0x1.b69c5c0000000p-4, -0x1.e11ac40000000p-5, 0x1.d503260000000p-3, 0x1.80011a0000000p-4, 0x1.c4e2d40000000p-4, 0x1.ddfe660000000p-5, -0x1.6b08620000000p-3, -0x1.9adf260000000p-4, 0x1.7708980000000p-3, 0x1.1efbf60000000p-4, -0x1.dcb6620000000p-4, -0x1.b11df80000000p-4, -0x1.1600aa0000000p-4, 0x1.0005300000000p-4, 0x1.42674e0000000p-3, -0x1.3f35940000000p-7, -0x1.2f07980000000p-2, -0x1.00d12c0000000p-2, -0x1.9a87ac0000000p-5, -0x1.fa3fb40000000p-6, -0x1.fc43760000000p-4}
}
, {{-0x1.24ddda0000000p-2, 0x1.1b985e0000000p-9, -0x1.34ca340000000p-4, -0x1.82128c0000000p-5, -0x1.44f4160000000p-3, -0x1.912ee20000000p-4, -0x1.47eaaa0000000p-3, -0x1.6982c80000000p-3, -0x1.2b271c0000000p-3, -0x1.4dd33c0000000p-2, 0x1.5038160000000p-3, -0x1.4ea6dc0000000p-4, -0x1.d8e43a0000000p-3, -0x1.0498ca0000000p-3, 0x1.3a1aae0000000p-4, -0x1.a8a5240000000p-2, -0x1.7a1aaa0000000p-3, 0x1.a499400000000p-3, 0x1.099b400000000p-4, -0x1.5741d40000000p-3, 0x1.bfe3120000000p-4, -0x1.add4d00000000p-3, 0x1.5d45980000000p-6, -0x1.4dd3860000000p-5, -0x1.0ef7f60000000p-3, 0x1.db77560000000p-7, -0x1.62a0440000000p-5, 0x1.8e34b60000000p-5, 0x1.3be3580000000p-3, -0x1.312fc20000000p-4, 0x1.9f4d8e0000000p-8, 0x1.0a1f5e0000000p-6}
, {-0x1.0c5b000000000p-2, -0x1.4499200000000p-4, 0x1.45474c0000000p-5, 0x1.91eb5a0000000p-4, -0x1.53a56c0000000p-7, -0x1.82a1aa0000000p-4, -0x1.1ef87a0000000p-3, -0x1.f9f7920000000p-7, 0x1.bc66680000000p-7, -0x1.ca30b00000000p-3, 0x1.b5c58e0000000p-4, 0x1.3a060c0000000p-4, 0x1.b670520000000p-5, -0x1.36a81c0000000p-5, 0x1.82e1560000000p-5, -0x1.44d9100000000p-3, -0x1.93d5220000000p-4, 0x1.76f8de0000000p-10, -0x1.2a35280000000p-8, 0x1.8c480c0000000p-8, -0x1.1d4fb20000000p-4, -0x1.aee3820000000p-2, -0x1.3808160000000p-5, 0x1.73e4580000000p-4, -0x1.a875ca0000000p-4, -0x1.558e4e0000000p-5, -0x1.94be600000000p-7, -0x1.a93c8a0000000p-5, 0x1.ae38de0000000p-2, 0x1.fd2cb60000000p-5, -0x1.8044160000000p-3, -0x1.a4f7be0000000p-6}
, {0x1.6cc1f80000000p-7, -0x1.8552fc0000000p-8, -0x1.ae47f00000000p-6, -0x1.8ec13c0000000p-5, 0x1.5ca2e80000000p-3, 0x1.c8f5660000000p-6, 0x1.19d2580000000p-3, 0x1.de964e0000000p-4, 0x1.0ff28e0000000p-5, -0x1.97c28c0000000p-4, 0x1.2b40760000000p-4, 0x1.a655f00000000p-3, -0x1.14afe40000000p-4, -0x1.3f24760000000p-12, -0x1.da58e60000000p-9, -0x1.bc3ae60000000p-11, -0x1.403aec0000000p-3, -0x1.a90b9c0000000p-3, 0x1.a1c4be0000000p-3, -0x1.0910ae0000000p-4, -0x1.129cba0000000p-3, -0x1.b3604e0000000p-5, 0x1.b342f20000000p-3, -0x1.5c8b920000000p-3, -0x1.72b1f40000000p-6, -0x1.48a2d40000000p-3, 0x1.6f0eda0000000p-3, 0x1.806f420000000p-4, 0x1.66033e0000000p-3, -0x1.342dc00000000p-3, 0x1.23bada0000000p-2, 0x1.1a425a0000000p-3}
}
, {{-0x1.5cefa20000000p-4, 0x1.0e168c0000000p-3, -0x1.1f94180000000p-2, -0x1.c349540000000p-3, -0x1.d4ba9c0000000p-3, -0x1.13cf7e0000000p-4, -0x1.42d00e0000000p-4, -0x1.d2c1460000000p-4, -0x1.2dac340000000p-3, -0x1.d21a660000000p-4, 0x1.6daea20000000p-4, -0x1.504f1c0000000p-13, 0x1.0e44820000000p-4, 0x1.68ca980000000p-7, 0x1.b56d6a0000000p-8, -0x1.d32d500000000p-6, 0x1.32480a0000000p-7, -0x1.044b100000000p-2, -0x1.66ad0c0000000p-4, -0x1.7777a00000000p-5, -0x1.2d7c8c0000000p-5, -0x1.0e24740000000p-2, 0x1.696f3a0000000p-5, 0x1.354cd60000000p-3, -0x1.28149a0000000p-2, 0x1.5decc40000000p-4, -0x1.472bc80000000p-3, -0x1.889daa0000000p-5, 0x1.15a6340000000p-5, 0x1.3599e60000000p-3, -0x1.91ea360000000p-7, -0x1.e40fba0000000p-3}
, {-0x1.12efbe0000000p-5, 0x1.d749960000000p-4, -0x1.9a93f20000000p-4, 0x1.a5f3260000000p-7, -0x1.d517ec0000000p-3, 0x1.29d2f20000000p-3, 0x1.8ae5260000000p-5, -0x1.72ad0e0000000p-4, -0x1.657be00000000p-3, 0x1.015d7c0000000p-4, -0x1.1665e60000000p-3, -0x1.3a3d6e0000000p-4, 0x1.566f2c0000000p-4, -0x1.aa94080000000p-5, -0x1.027c900000000p-4, -0x1.022dfa0000000p-3, 0x1.a329c40000000p-3, 0x1.f906000000000p-4, -0x1.694b7a0000000p-5, -0x1.3a36560000000p-3, 0x1.c626480000000p-5, -0x1.cbe62a0000000p-3, 0x1.0161860000000p-2, -0x1.aebf4a0000000p-7, -0x1.ced11c0000000p-4, -0x1.1c34680000000p-2, -0x1.72436c0000000p-5, 0x1.816a580000000p-4, -0x1.a5bcd60000000p-4, -0x1.5830e20000000p-2, -0x1.44f5260000000p-4, -0x1.1422a80000000p-3}
, {0x1.af5b860000000p-3, -0x1.1b47de0000000p-8, 0x1.681fd40000000p-3, 0x1.5ae5260000000p-3, -0x1.cba41e0000000p-5, 0x1.11ea700000000p-2, 0x1.2685f80000000p-4, -0x1.4b37ac0000000p-3, -0x1.b1f78c0000000p-5, 0x1.e4b2f00000000p-5, -0x1.ed6f960000000p-3, 0x1.eb4af00000000p-4, -0x1.4c64380000000p-4, 0x1.4dad820000000p-5, 0x1.8684020000000p-3, 0x1.8467ea0000000p-5, -0x1.d403440000000p-4, 0x1.6b663a0000000p-3, -0x1.b5c1320000000p-7, -0x1.11e4be0000000p-4, 0x1.9dcaa60000000p-3, -0x1.e8283a0000000p-8, -0x1.045fa20000000p-5, -0x1.2ec26e0000000p-2, -0x1.2f506a0000000p-3, 0x1.6a23860000000p-5, -0x1.1ebc920000000p-5, -0x1.3587b80000000p-4, 0x1.0426160000000p-3, 0x1.4f998e0000000p-3, -0x1.332e6c0000000p-4, 0x1.08b74c0000000p-8}
}
, {{-0x1.6af7700000000p-3, 0x1.7d72480000000p-4, 0x1.34a4540000000p-3, -0x1.d09df40000000p-4, -0x1.9d1e1a0000000p-2, 0x1.4b1c700000000p-5, -0x1.62ce820000000p-3, -0x1.cc79ca0000000p-5, -0x1.4078bc0000000p-2, -0x1.9fe2de0000000p-3, -0x1.18220c0000000p-2, -0x1.63817a0000000p-2, -0x1.55f1d40000000p-4, -0x1.d9600c0000000p-4, -0x1.8087860000000p-3, -0x1.31c91a0000000p-2, 0x1.bffc880000000p-5, 0x1.214d5a0000000p-4, -0x1.4fd6480000000p-3, -0x1.730a200000000p-4, 0x1.5182320000000p-4, -0x1.0b84420000000p-3, 0x1.01ea6a0000000p-3, 0x1.2091c40000000p-4, -0x1.d306d40000000p-3, -0x1.eb83ea0000000p-5, -0x1.08a3160000000p-3, -0x1.ec9b940000000p-5, -0x1.5bac1a0000000p-3, 0x1.545fe00000000p-3, -0x1.993d9a0000000p-3, 0x1.7114580000000p-2}
, {-0x1.008dc00000000p-6, 0x1.509d6c0000000p-3, -0x1.7686400000000p-5, -0x1.b1ea3a0000000p-3, -0x1.3705940000000p-3, 0x1.1e19220000000p-3, -0x1.a7deba0000000p-4, -0x1.801b980000000p-4, -0x1.4337000000000p-4, 0x1.74d8320000000p-4, -0x1.97b0160000000p-3, 0x1.326ca60000000p-4, -0x1.18a3f00000000p-3, 0x1.3d2ae20000000p-2, -0x1.cb12900000000p-5, -0x1.e961e40000000p-4, -0x1.7494420000000p-3, -0x1.10c5740000000p-3, -0x1.c7579a0000000p-4, 0x1.bf29420000000p-6, -0x1.6432020000000p-5, -0x1.a53cca0000000p-3, 0x1.1b51900000000p-4, 0x1.519b100000000p-7, 0x1.142d8e0000000p-3, -0x1.8cf8380000000p-5, -0x1.d6bbf40000000p-4, -0x1.d0fd0e0000000p-4, 0x1.8557b80000000p-3, -0x1.da34180000000p-5, 0x1.1f19620000000p-3, -0x1.03c4160000000p-10}
, {-0x1.cc469c0000000p-4, -0x1.a20f580000000p-3, 0x1.a72be20000000p-3, -0x1.36f66e0000000p-2, -0x1.915d1a0000000p-8, 0x1.3d48ac0000000p-6, -0x1.1f84320000000p-5, 0x1.2da3060000000p-3, -0x1.4d7cf80000000p-2, 0x1.e9c7860000000p-5, -0x1.3c092a0000000p-2, 0x1.1dd11c0000000p-3, -0x1.a1f2ac0000000p-4, -0x1.a1fc660000000p-3, -0x1.9df9560000000p-10, 0x1.8152b60000000p-3, 0x1.cd9dc60000000p-3, 0x1.d573bc0000000p-9, -0x1.79cee40000000p-4, -0x1.9b70400000000p-2, 0x1.8fc2e00000000p-4, -0x1.619ee80000000p-4, 0x1.c353700000000p-4, 0x1.ded4680000000p-4, -0x1.a099940000000p-2, -0x1.6f95720000000p-4, 0x1.47b3ea0000000p-4, 0x1.b8a0ea0000000p-5, -0x1.e793520000000p-4, 0x1.8c1bd00000000p-3, -0x1.762bf60000000p-3, 0x1.8695200000000p-3}
}
, {{-0x1.8680740000000p-3, 0x1.1e76440000000p-6, -0x1.1f5e880000000p-4, 0x1.476daa0000000p-5, -0x1.8e9fc40000000p-4, -0x1.7b7ec20000000p-5, 0x1.516da20000000p-5, 0x1.2c7c240000000p-4, -0x1.120ad40000000p-3, -0x1.b31b280000000p-5, 0x1.f049360000000p-4, -0x1.63143c0000000p-5, 0x1.be20020000000p-3, -0x1.00676c0000000p-2, -0x1.98bc840000000p-3, -0x1.114b600000000p-2, 0x1.41294e0000000p-3, -0x1.3f00ae0000000p-3, 0x1.feebc60000000p-3, -0x1.8531520000000p-4, -0x1.4f5d1a0000000p-3, -0x1.40c7560000000p-2, 0x1.1805860000000p-2, 0x1.e7df340000000p-5, -0x1.e8e3c60000000p-6, -0x1.a9beea0000000p-7, -0x1.4150e00000000p-4, 0x1.47bd340000000p-4, -0x1.fa36200000000p-5, 0x1.5dc2760000000p-3, 0x1.15fa4a0000000p-4, -0x1.c190120000000p-5}
, {-0x1.6167060000000p-13, -0x1.451a340000000p-8, 0x1.614f480000000p-6, 0x1.8d9d100000000p-5, -0x1.0b777a0000000p-9, 0x1.25c11c0000000p-4, 0x1.b5cca40000000p-3, -0x1.67b1d20000000p-3, -0x1.7042140000000p-7, 0x1.a54dbe0000000p-5, 0x1.e2053c0000000p-3, 0x1.0f60aa0000000p-6, -0x1.27ff5c0000000p-5, 0x1.91bb560000000p-7, -0x1.121f9e0000000p-3, 0x1.01787a0000000p-8, 0x1.5bd4540000000p-4, -0x1.fba1ac0000000p-4, -0x1.906c720000000p-7, 0x1.eab0100000000p-5, 0x1.de9b5a0000000p-4, -0x1.5d748a0000000p-5, -0x1.0922d00000000p-3, 0x1.6146600000000p-3, -0x1.7647200000000p-3, -0x1.1c30200000000p-2, 0x1.fcdfb80000000p-4, -0x1.8c8ffa0000000p-5, -0x1.3a2fe40000000p-3, 0x1.e6aa4c0000000p-5, 0x1.768c7c0000000p-7, -0x1.61858e0000000p-4}
, {-0x1.4fe9860000000p-2, -0x1.7bd26a0000000p-10, -0x1.a026ec0000000p-2, -0x1.06d6280000000p-2, 0x1.0f92080000000p-4, 0x1.a825520000000p-5, 0x1.450f2e0000000p-6, 0x1.c091940000000p-4, 0x1.7591f80000000p-6, -0x1.c916140000000p-3, 0x1.9edfca0000000p-5, 0x1.0077f20000000p-6, 0x1.75cabe0000000p-3, 0x1.57b1ce0000000p-2, -0x1.fb88f00000000p-4, 0x1.3dbb060000000p-4, -0x1.c341180000000p-2, 0x1.fff3b20000000p-5, -0x1.a1a4e60000000p-3, 0x1.2503d00000000p-7, -0x1.8d9c200000000p-3, 0x1.3de84a0000000p-2, 0x1.cbabec0000000p-4, 0x1.4579b60000000p-3, 0x1.5b93400000000p-5, -0x1.653c640000000p-4, 0x1.2932220000000p-5, 0x1.dd48ce0000000p-4, 0x1.3654a20000000p-2, -0x1.c7dd1a0000000p-5, 0x1.8e31240000000p-3, 0x1.f6530a0000000p-4}
}
, {{-0x1.2800780000000p-4, -0x1.2e79000000000p-3, 0x1.2096820000000p-6, -0x1.ba34c40000000p-3, -0x1.1eeae20000000p-3, -0x1.b65f360000000p-4, -0x1.185bee0000000p-4, 0x1.d1b0720000000p-4, -0x1.21a54a0000000p-2, 0x1.126f560000000p-4, -0x1.0c38780000000p-3, -0x1.e2b87e0000000p-4, -0x1.54e17c0000000p-2, 0x1.d0880e0000000p-3, -0x1.32996c0000000p-3, -0x1.be71b80000000p-3, -0x1.6b8a360000000p-4, -0x1.7de8e00000000p-8, 0x1.dab0740000000p-4, -0x1.0c02ce0000000p-4, -0x1.578b760000000p-4, -0x1.4f33480000000p-3, 0x1.13ce9e0000000p-3, 0x1.2b096a0000000p-3, -0x1.590e8e0000000p-3, 0x1.49c03c0000000p-5, -0x1.045cc60000000p-3, -0x1.dceb1c0000000p-4, -0x1.a1c8220000000p-4, 0x1.6182140000000p-2, -0x1.9659a80000000p-4, 0x1.197d280000000p-10}
, {0x1.4c998a0000000p-4, 0x1.1fa3420000000p-3, -0x1.5774a60000000p-10, -0x1.1c86f60000000p-3, -0x1.b4bf960000000p-3, -0x1.3d994e0000000p-4, -0x1.53922c0000000p-6, 0x1.7be89e0000000p-2, 0x1.e50fea0000000p-3, -0x1.7a9d320000000p-6, -0x1.9216e60000000p-5, -0x1.a110ca0000000p-5, -0x1.dc907e0000000p-4, 0x1.58eee40000000p-6, 0x1.547d820000000p-3, -0x1.65f85a0000000p-8, -0x1.043fe60000000p-5, 0x1.80262e0000000p-6, 0x1.6c654c0000000p-5, 0x1.387fe60000000p-2, 0x1.689f920000000p-3, 0x1.c2e22a0000000p-4, -0x1.e1728c0000000p-5, -0x1.85af640000000p-3, 0x1.a7a0a20000000p-6, -0x1.4b207a0000000p-6, -0x1.01f0d00000000p-4, 0x1.9528a80000000p-4, 0x1.63840a0000000p-5, -0x1.5771360000000p-5, 0x1.574c4c0000000p-6, 0x1.257c660000000p-5}
, {-0x1.0e15a80000000p-3, 0x1.34a1900000000p-4, 0x1.82f6520000000p-4, -0x1.31305c0000000p-2, -0x1.9a87ea0000000p-3, -0x1.9f80be0000000p-5, -0x1.2bb0e00000000p-4, 0x1.3e4b180000000p-3, -0x1.e44b8e0000000p-5, 0x1.4274780000000p-4, 0x1.f861620000000p-5, 0x1.e79a120000000p-5, -0x1.6190520000000p-9, -0x1.7ce97a0000000p-2, -0x1.2e15d60000000p-3, -0x1.b828fa0000000p-3, 0x1.b054500000000p-6, -0x1.61eb480000000p-3, -0x1.12caa00000000p-3, -0x1.e3a3380000000p-3, -0x1.4d4ffc0000000p-4, 0x1.2b58600000000p-5, 0x1.87b9820000000p-5, 0x1.5117680000000p-3, -0x1.41171a0000000p-5, -0x1.00238a0000000p-5, -0x1.2750740000000p-4, 0x1.10578c0000000p-4, -0x1.0d07a80000000p-8, 0x1.0dcf1e0000000p-8, 0x1.1074320000000p-4, 0x1.916f7a0000000p-4}
}
, {{-0x1.7ecf980000000p-3, -0x1.09cc840000000p-2, -0x1.02b5f40000000p-3, 0x1.9a401e0000000p-3, -0x1.b2ff220000000p-4, -0x1.05db2e0000000p-4, -0x1.f100ee0000000p-4, -0x1.0678e00000000p-3, 0x1.3902860000000p-4, 0x1.28a1aa0000000p-5, 0x1.875f700000000p-4, -0x1.278aba0000000p-5, -0x1.7329240000000p-4, -0x1.1bc3ea0000000p-3, 0x1.e58f7a0000000p-4, -0x1.31b9e00000000p-4, -0x1.5281060000000p-4, -0x1.3f7fec0000000p-3, -0x1.797d460000000p-4, 0x1.e364ac0000000p-6, 0x1.b242c20000000p-4, -0x1.c115740000000p-5, 0x1.583b900000000p-5, -0x1.43f4c20000000p-3, 0x1.7a0d440000000p-7, 0x1.d013d20000000p-4, 0x1.57e16a0000000p-8, -0x1.4ec03c0000000p-4, 0x1.52fb340000000p-4, -0x1.4e8cb40000000p-3, -0x1.239e880000000p-4, 0x1.8f50880000000p-10}
, {-0x1.3b35f60000000p-2, -0x1.cea6580000000p-10, -0x1.f9eaf00000000p-4, -0x1.32cdb00000000p-3, 0x1.11962e0000000p-4, -0x1.156fe40000000p-4, 0x1.18ec000000000p-3, -0x1.523b000000000p-6, 0x1.35a09a0000000p-3, -0x1.458df00000000p-2, -0x1.8e55260000000p-8, 0x1.4a6e900000000p-3, 0x1.1e3fe40000000p-7, 0x1.f92dc60000000p-5, 0x1.11b7780000000p-4, 0x1.0572600000000p-5, -0x1.2a45c60000000p-2, 0x1.42b7f80000000p-6, 0x1.4262080000000p-4, 0x1.1dd8fe0000000p-4, -0x1.bd3ea00000000p-6, 0x1.03f54e0000000p-3, 0x1.682b8c0000000p-6, -0x1.d3168e0000000p-6, 0x1.c272900000000p-3, -0x1.77a6400000000p-10, 0x1.152fb40000000p-3, -0x1.65fdd00000000p-4, 0x1.7e67520000000p-5, -0x1.579f4c0000000p-3, 0x1.af59680000000p-5, -0x1.60ec3c0000000p-6}
, {0x1.c1491a0000000p-7, -0x1.0be8ee0000000p-4, -0x1.01e11e0000000p-3, -0x1.680d120000000p-5, 0x1.2921320000000p-5, 0x1.d902480000000p-4, -0x1.0f3f580000000p-2, 0x1.c240180000000p-5, 0x1.09b8620000000p-3, 0x1.317e1e0000000p-8, -0x1.fae2de0000000p-3, -0x1.b232900000000p-8, 0x1.80f0e40000000p-7, -0x1.6527ce0000000p-8, -0x1.4fe6240000000p-5, 0x1.1081500000000p-4, -0x1.ac82120000000p-2, 0x1.f0dbbc0000000p-4, 0x1.06475e0000000p-4, 0x1.ed92fa0000000p-4, -0x1.2ea6180000000p-5, 0x1.3d3eb40000000p-3, 0x1.7b85760000000p-7, -0x1.5593bc0000000p-3, 0x1.2f9bdc0000000p-6, 0x1.9c862c0000000p-4, -0x1.bbfa880000000p-9, 0x1.2e62ee0000000p-3, 0x1.2bb9760000000p-3, -0x1.5ad7a00000000p-2, 0x1.3b96cc0000000p-2, 0x1.0621fa0000000p-5}
}
, {{-0x1.792ade0000000p-3, -0x1.a49ff20000000p-4, 0x1.7eb0660000000p-5, 0x1.6771f80000000p-3, -0x1.98c3d80000000p-4, -0x1.05af120000000p-3, -0x1.8ec9700000000p-3, -0x1.15a40e0000000p-2, -0x1.2c22040000000p-5, 0x1.8f27f80000000p-3, -0x1.d9980e0000000p-4, 0x1.caf9340000000p-15, 0x1.0cbbb40000000p-2, 0x1.0911cc0000000p-3, 0x1.1a51f60000000p-2, -0x1.0fcae80000000p-2, -0x1.b2b1a60000000p-6, -0x1.f6ecf20000000p-3, 0x1.0f8f780000000p-2, -0x1.63811e0000000p-8, 0x1.7eacd00000000p-3, 0x1.66a9b20000000p-5, -0x1.351c560000000p-3, -0x1.21ec860000000p-6, -0x1.33c2ee0000000p-6, 0x1.f9d8a80000000p-3, 0x1.d71e4e0000000p-5, -0x1.9a06480000000p-2, -0x1.755a460000000p-3, 0x1.201fa00000000p-2, -0x1.f04bda0000000p-5, 0x1.3b6d160000000p-6}
, {-0x1.d8ba000000000p-8, -0x1.9dd9b80000000p-3, -0x1.9adba40000000p-5, 0x1.156c7a0000000p-4, -0x1.24e82e0000000p-3, 0x1.c68a8a0000000p-6, -0x1.22e55c0000000p-5, 0x1.0637140000000p-3, 0x1.319e840000000p-5, 0x1.013b780000000p-6, -0x1.10997a0000000p-4, -0x1.6a0ae20000000p-5, -0x1.016f060000000p-3, 0x1.a550640000000p-5, -0x1.53df360000000p-5, -0x1.a422560000000p-4, -0x1.bc0fa60000000p-3, 0x1.27ea820000000p-2, -0x1.28a1560000000p-5, 0x1.7eaca00000000p-4, -0x1.2f7e620000000p-4, -0x1.21a7660000000p-3, 0x1.c6610c0000000p-4, -0x1.1aaa0e0000000p-4, 0x1.12dda20000000p-4, 0x1.0ddac80000000p-3, -0x1.8c2be20000000p-4, -0x1.403e6e0000000p-3, 0x1.0cafea0000000p-2, -0x1.a056380000000p-7, 0x1.3848e00000000p-5, 0x1.aff7960000000p-4}
, {0x1.8a05f40000000p-3, -0x1.f2de720000000p-4, 0x1.a75e6c0000000p-4, -0x1.11c07e0000000p-2, -0x1.78ddbe0000000p-4, 0x1.0273460000000p-2, -0x1.6413c60000000p-4, 0x1.4255a20000000p-2, 0x1.17bff60000000p-3, 0x1.af46240000000p-6, -0x1.862e260000000p-4, 0x1.3d41300000000p-3, 0x1.dfceb60000000p-5, -0x1.62e2ee0000000p-4, -0x1.5928ee0000000p-3, 0x1.5783600000000p-4, -0x1.05ce240000000p-3, -0x1.cdc6ce0000000p-4, -0x1.83f1e80000000p-5, 0x1.c7f08a0000000p-6, -0x1.5132780000000p-7, -0x1.79ab340000000p-5, 0x1.9868520000000p-3, 0x1.c9f8e80000000p-4, 0x1.de6c720000000p-5, -0x1.048e420000000p-5, 0x1.cc6ba60000000p-4, 0x1.42d3ca0000000p-3, 0x1.76410a0000000p-3, -0x1.2629b80000000p-2, -0x1.b342120000000p-8, 0x1.4aaf920000000p-3}
}
, {{0x1.f8cb700000000p-5, -0x1.4c95840000000p-4, -0x1.f6a6760000000p-5, -0x1.e69b6a0000000p-2, 0x1.1b38840000000p-3, 0x1.5679240000000p-3, 0x1.7a171a0000000p-7, 0x1.b4813e0000000p-3, -0x1.2d75a80000000p-6, -0x1.3460160000000p-3, 0x1.d6af3c0000000p-6, 0x1.a4e6400000000p-3, -0x1.ff8b9c0000000p-3, -0x1.47eb620000000p-2, -0x1.5aa5c80000000p-2, -0x1.0163820000000p-5, -0x1.674e880000000p-5, 0x1.221a4e0000000p-3, 0x1.a1c5660000000p-4, -0x1.a2fb120000000p-4, -0x1.ae9a8e0000000p-5, -0x1.94a8820000000p-4, 0x1.f7b7620000000p-3, 0x1.831d500000000p-7, 0x1.32e9280000000p-4, -0x1.0742340000000p-1, 0x1.55c23c0000000p-3, 0x1.7c83fa0000000p-3, 0x1.0e76de0000000p-2, -0x1.5ce82a0000000p-4, 0x1.0393e40000000p-3, 0x1.6be7dc0000000p-3}
, {-0x1.74012a0000000p-4, -0x1.9db4d00000000p-4, 0x1.6585bc0000000p-4, -0x1.0891ea0000000p-2, 0x1.ca0fcc0000000p-4, 0x1.e9716c0000000p-4, 0x1.cd10ca0000000p-6, 0x1.3fdace0000000p-4, -0x1.327a320000000p-3, 0x1.da1afc0000000p-4, -0x1.a5b6a00000000p-5, 0x1.6bead20000000p-5, -0x1.81ca5e0000000p-4, -0x1.be08140000000p-3, -0x1.b58b2c0000000p-3, 0x1.213c580000000p-5, -0x1.9afffc0000000p-5, -0x1.427a980000000p-4, -0x1.6957ae0000000p-5, -0x1.0634360000000p-3, -0x1.9387700000000p-3, -0x1.3cfaae0000000p-3, 0x1.6215960000000p-3, 0x1.8223b60000000p-4, 0x1.aede680000000p-7, -0x1.651dcc0000000p-4, -0x1.b14b9e0000000p-5, 0x1.79fe100000000p-6, -0x1.d075a60000000p-4, -0x1.27262e0000000p-8, -0x1.9449960000000p-5, 0x1.d6671a0000000p-7}
, {-0x1.4ab76a0000000p-2, 0x1.1b12d40000000p-3, 0x1.f280da0000000p-3, -0x1.9048320000000p-3, -0x1.3caddc0000000p-2, 0x1.9ce9be0000000p-5, -0x1.8949720000000p-4, 0x1.83d11a0000000p-6, -0x1.9c3e2e0000000p-7, 0x1.8776d20000000p-3, -0x1.163e7e0000000p-3, -0x1.4e685e0000000p-2, 0x1.04c6e20000000p-3, -0x1.5a3c380000000p-5, -0x1.49a7f80000000p-2, -0x1.af57dc0000000p-4, 0x1.21cb5c0000000p-4, -0x1.c4de180000000p-2, -0x1.3ccabe0000000p-3, 0x1.1402600000000p-3, -0x1.30367a0000000p-2, -0x1.5e6bbc0000000p-3, -0x1.6ba18a0000000p-4, 0x1.a8833e0000000p-4, 0x1.6078b00000000p-2, 0x1.276b400000000p-4, -0x1.68438a0000000p-1, -0x1.bb398a0000000p-2, -0x1.efd5ca0000000p-3, 0x1.011f160000000p-2, -0x1.76328a0000000p-3, -0x1.6433a60000000p-3}
}
, {{-0x1.14d5380000000p-4, 0x1.b0509a0000000p-5, 0x1.9a59560000000p-3, 0x1.f75dd60000000p-5, -0x1.98671c0000000p-2, -0x1.351ce20000000p-7, -0x1.7389ac0000000p-4, -0x1.f354520000000p-6, -0x1.46a9760000000p-8, -0x1.fe86a60000000p-3, 0x1.1694200000000p-3, -0x1.0292640000000p-3, 0x1.e93f1a0000000p-5, -0x1.ddb6040000000p-4, 0x1.a3c4a00000000p-5, -0x1.6b38940000000p-6, 0x1.91c3e00000000p-3, -0x1.43477c0000000p-4, 0x1.d61b3e0000000p-3, -0x1.4ed7820000000p-4, 0x1.0549b40000000p-7, -0x1.e860680000000p-4, 0x1.7178040000000p-2, -0x1.6355200000000p-7, 0x1.ee2d780000000p-5, -0x1.b0e8d40000000p-11, 0x1.9ad0e00000000p-4, -0x1.8f977e0000000p-3, -0x1.cd85c40000000p-5, 0x1.69591e0000000p-2, -0x1.5a810e0000000p-6, -0x1.1d5b5a0000000p-5}
, {-0x1.ba035c0000000p-4, 0x1.e3ec020000000p-4, -0x1.2bb2240000000p-2, 0x1.683c200000000p-7, -0x1.de0b2e0000000p-4, -0x1.0d97f80000000p-3, -0x1.2561f00000000p-5, 0x1.b5bc360000000p-7, 0x1.d7fb2e0000000p-5, -0x1.ece9ea0000000p-7, 0x1.19fe000000000p-4, -0x1.223c560000000p-5, -0x1.9c812e0000000p-3, -0x1.b5302a0000000p-4, 0x1.d437760000000p-4, 0x1.36929a0000000p-6, -0x1.2c650c0000000p-3, -0x1.a03c440000000p-3, 0x1.8c60fa0000000p-7, -0x1.dd7ac40000000p-5, -0x1.ef1f420000000p-8, -0x1.49e5cc0000000p-5, -0x1.10119a0000000p-5, -0x1.3f91160000000p-4, -0x1.af91040000000p-8, 0x1.59e82a0000000p-6, 0x1.6fecf40000000p-5, 0x1.50e88c0000000p-6, 0x1.2346440000000p-3, -0x1.317a160000000p-4, 0x1.c5ff120000000p-5, 0x1.9df5d20000000p-4}
, {-0x1.f5b8d60000000p-3, -0x1.b271ca0000000p-4, -0x1.35b8f60000000p-2, 0x1.0503d00000000p-4, 0x1.b076540000000p-4, 0x1.82607e0000000p-6, 0x1.085e640000000p-3, -0x1.5a1f520000000p-3, 0x1.0432720000000p-3, 0x1.a147160000000p-7, 0x1.6bff440000000p-6, 0x1.a3ef060000000p-4, -0x1.d318040000000p-3, -0x1.04ae9e0000000p-3, 0x1.23ffb80000000p-7, -0x1.2d8f4e0000000p-3, -0x1.c9dce40000000p-4, -0x1.f3673e0000000p-4, 0x1.daf0c00000000p-4, 0x1.14807a0000000p-3, -0x1.eb71fe0000000p-3, 0x1.6c7d160000000p-4, 0x1.2b2c180000000p-6, 0x1.0b7e3c0000000p-4, 0x1.fe96b80000000p-4, -0x1.6185480000000p-4, -0x1.c1d8940000000p-4, 0x1.a457ac0000000p-4, -0x1.bf66100000000p-4, 0x1.9141b20000000p-5, 0x1.c9f7440000000p-3, 0x1.9a26680000000p-7}
}
, {{0x1.bfd9520000000p-4, -0x1.a867940000000p-4, -0x1.300d1e0000000p-8, -0x1.1f95800000000p-3, 0x1.252d7c0000000p-3, 0x1.0769640000000p-3, 0x1.e5823a0000000p-5, 0x1.db0b520000000p-3, 0x1.7416640000000p-4, -0x1.05057c0000000p-3, 0x1.152ab00000000p-3, 0x1.bf7d440000000p-5, -0x1.b574780000000p-8, 0x1.6c90f00000000p-6, -0x1.8823200000000p-5, 0x1.1ef6fc0000000p-3, -0x1.16f6000000000p-5, 0x1.69b4600000000p-3, 0x1.8e30340000000p-4, 0x1.3c44480000000p-2, -0x1.09f94c0000000p-3, -0x1.bc58840000000p-7, 0x1.45c1ec0000000p-5, -0x1.2a27c40000000p-3, 0x1.2b74be0000000p-2, 0x1.57e1ca0000000p-6, 0x1.15c59e0000000p-3, 0x1.3e09b20000000p-3, 0x1.369d9a0000000p-2, -0x1.8d55a20000000p-4, 0x1.5cc2da0000000p-5, 0x1.0620c00000000p-3}
, {-0x1.98a4e80000000p-4, -0x1.16b04a0000000p-4, 0x1.4274ca0000000p-7, -0x1.2888dc0000000p-3, -0x1.ee56560000000p-4, -0x1.5f71360000000p-6, 0x1.e7d9120000000p-13, -0x1.90c6fa0000000p-5, 0x1.4a9ade0000000p-4, 0x1.1cd6780000000p-5, -0x1.be7c400000000p-5, -0x1.21b23e0000000p-2, 0x1.e266480000000p-5, 0x1.d404da0000000p-8, 0x1.f776a40000000p-4, -0x1.adc7b40000000p-4, 0x1.796ace0000000p-5, -0x1.6c1d180000000p-2, -0x1.7d58560000000p-5, 0x1.d5741a0000000p-5, -0x1.dd837a0000000p-5, -0x1.e4e3ea0000000p-4, -0x1.320b6c0000000p-3, -0x1.a2197a0000000p-4, 0x1.d2b3900000000p-4, -0x1.2f14f00000000p-5, -0x1.561cd80000000p-3, -0x1.f3fb3e0000000p-7, 0x1.85cdec0000000p-5, -0x1.4036c00000000p-3, 0x1.51d9300000000p-6, 0x1.b681600000000p-4}
, {0x1.70f6500000000p-5, -0x1.d0da780000000p-8, 0x1.824d620000000p-6, 0x1.810cde0000000p-10, 0x1.6aac640000000p-4, -0x1.8189400000000p-4, 0x1.d68f6c0000000p-5, 0x1.5389100000000p-3, -0x1.2113220000000p-2, -0x1.415c820000000p-6, 0x1.13509e0000000p-5, 0x1.7c59f40000000p-3, 0x1.47d5f80000000p-3, 0x1.3881300000000p-3, 0x1.0cbfb80000000p-5, 0x1.1bcf040000000p-4, 0x1.15a75c0000000p-4, -0x1.4cbc2c0000000p-7, 0x1.2424140000000p-4, -0x1.d9e7580000000p-5, -0x1.a023420000000p-4, -0x1.7c632c0000000p-6, -0x1.1017260000000p-3, -0x1.b1780e0000000p-3, -0x1.4342600000000p-3, -0x1.3f94700000000p-4, 0x1.2645a20000000p-3, 0x1.7c3e200000000p-4, -0x1.1afc080000000p-4, -0x1.2093820000000p-3, -0x1.7a23c40000000p-4, 0x1.259ed60000000p-4}
}
, {{-0x1.2e7a300000000p-6, 0x1.2c60200000000p-3, -0x1.97f0e80000000p-2, -0x1.34c6620000000p-2, -0x1.80f1c20000000p-5, 0x1.7169f60000000p-3, 0x1.4135540000000p-4, -0x1.30033c0000000p-3, -0x1.c969500000000p-14, -0x1.431ed60000000p-6, -0x1.41ba080000000p-4, 0x1.e139380000000p-3, -0x1.0fb4020000000p-2, -0x1.f006120000000p-7, -0x1.0ef40c0000000p-2, -0x1.bb31640000000p-4, 0x1.2e50ec0000000p-7, -0x1.aafcb60000000p-5, -0x1.9797300000000p-5, -0x1.20a1aa0000000p-2, -0x1.54d59e0000000p-3, -0x1.9ef3620000000p-3, -0x1.2978640000000p-3, 0x1.6ec5aa0000000p-4, -0x1.40ad220000000p-3, -0x1.caedf20000000p-5, -0x1.c848ea0000000p-4, 0x1.afae5e0000000p-3, -0x1.e144d40000000p-4, 0x1.efac200000000p-4, 0x1.ee46520000000p-3, 0x1.42ab5e0000000p-3}
, {-0x1.b1973e0000000p-5, 0x1.492ff60000000p-3, 0x1.a4554c0000000p-4, 0x1.8f482a0000000p-3, -0x1.f178620000000p-4, -0x1.a5c73c0000000p-2, -0x1.5ae0640000000p-3, 0x1.1db8d20000000p-8, -0x1.000e600000000p-3, 0x1.89e16a0000000p-5, -0x1.d9ee2c0000000p-3, 0x1.2db3f80000000p-4, -0x1.1613480000000p-2, 0x1.46622a0000000p-4, -0x1.6e3d8a0000000p-3, 0x1.85bb420000000p-2, -0x1.1ae0b80000000p-2, -0x1.20ae820000000p-2, 0x1.bde41a0000000p-5, 0x1.b3779c0000000p-9, 0x1.8f38a80000000p-5, 0x1.1c525e0000000p-5, -0x1.9fa1aa0000000p-5, -0x1.93d00c0000000p-8, -0x1.2995340000000p-3, 0x1.70fdf20000000p-3, -0x1.18d7480000000p-4, -0x1.08a5d60000000p-6, 0x1.4a29c60000000p-4, -0x1.52269c0000000p-2, 0x1.62a8c00000000p-3, -0x1.7580800000000p-6}
, {-0x1.45a0220000000p-3, -0x1.1fb6460000000p-5, -0x1.fc93be0000000p-2, 0x1.2919460000000p-4, -0x1.a3efcc0000000p-4, 0x1.0b5ed80000000p-2, -0x1.3769e60000000p-2, -0x1.8efaa20000000p-5, 0x1.f288080000000p-2, -0x1.dc785c0000000p-4, -0x1.bc21b20000000p-4, -0x1.828bd00000000p-5, -0x1.a601980000000p-3, 0x1.1347b00000000p-1, -0x1.d9cf560000000p-3, 0x1.e3afa60000000p-4, -0x1.03ce6c0000000p-2, -0x1.07e1480000000p-2, -0x1.64f9d80000000p-2, 0x1.ca74340000000p-3, 0x1.4463e00000000p-6, 0x1.c94a900000000p-3, -0x1.06abee0000000p-3, -0x1.3104c40000000p-2, 0x1.2aed120000000p-4, 0x1.2b820a0000000p-2, -0x1.412fca0000000p-4, 0x1.6690560000000p-4, -0x1.2259840000000p-4, -0x1.5c1b960000000p-2, 0x1.164bae0000000p-5, 0x1.7e38660000000p-3}
}
, {{-0x1.f20f3c0000000p-5, 0x1.f1d6c60000000p-8, 0x1.0109ec0000000p-3, 0x1.b01f100000000p-4, 0x1.6926920000000p-6, 0x1.1103780000000p-8, -0x1.42ec0a0000000p-5, -0x1.156ece0000000p-2, 0x1.cba02c0000000p-3, -0x1.2d857a0000000p-4, -0x1.75b0860000000p-8, 0x1.0813dc0000000p-3, -0x1.94cf640000000p-5, 0x1.5796ce0000000p-3, 0x1.5636de0000000p-5, 0x1.8bf3ce0000000p-5, -0x1.0111ce0000000p-2, 0x1.534dd00000000p-5, -0x1.04bffc0000000p-5, -0x1.24d39c0000000p-4, 0x1.b66c2e0000000p-4, -0x1.8a3f900000000p-6, -0x1.effca60000000p-3, -0x1.5e346a0000000p-2, 0x1.42607e0000000p-3, 0x1.2312060000000p-3, -0x1.b222740000000p-9, -0x1.0aa5e40000000p-3, -0x1.03f5740000000p-3, 0x1.f00eae0000000p-5, -0x1.a9f4ec0000000p-8, -0x1.b5dd500000000p-7}
, {-0x1.1f1b8e0000000p-5, 0x1.38ccee0000000p-3, 0x1.3153840000000p-4, 0x1.2fe9280000000p-5, -0x1.96d3580000000p-6, -0x1.3ec7a00000000p-2, -0x1.c151920000000p-6, 0x1.0f9b240000000p-4, 0x1.6d7dd40000000p-4, -0x1.0382de0000000p-2, -0x1.d7279e0000000p-4, 0x1.8a94bc0000000p-3, 0x1.a11ffa0000000p-5, -0x1.1e596a0000000p-3, 0x1.6fd0080000000p-3, -0x1.1f45800000000p-3, -0x1.4ca87c0000000p-3, -0x1.fec66c0000000p-7, 0x1.6ea6400000000p-3, 0x1.df2cce0000000p-8, -0x1.7a551c0000000p-6, 0x1.d1b4120000000p-5, -0x1.8460c20000000p-4, -0x1.8805ca0000000p-3, -0x1.0bde380000000p-4, 0x1.8a49320000000p-5, 0x1.15f58e0000000p-4, -0x1.757afc0000000p-4, 0x1.2865c20000000p-3, -0x1.a3a81e0000000p-7, 0x1.563f120000000p-4, -0x1.281a260000000p-3}
, {-0x1.c1e4860000000p-9, -0x1.96ddbe0000000p-3, -0x1.eb437c0000000p-4, 0x1.174eba0000000p-4, -0x1.392cc60000000p-7, 0x1.0f99380000000p-2, -0x1.3249680000000p-3, 0x1.a1a7960000000p-6, 0x1.445d280000000p-3, 0x1.f67ff80000000p-8, -0x1.52fc020000000p-3, 0x1.7ed0d00000000p-3, -0x1.506bc00000000p-3, 0x1.0fee000000000p-4, -0x1.e0ad6c0000000p-3, 0x1.2c4a860000000p-3, -0x1.c101160000000p-4, -0x1.88b5500000000p-3, -0x1.04548e0000000p-6, 0x1.1b025c0000000p-5, -0x1.7d29ce0000000p-5, 0x1.78f6280000000p-5, -0x1.072cae0000000p-4, -0x1.c0ad160000000p-3, 0x1.c2dae80000000p-4, 0x1.7167a00000000p-3, 0x1.35a4960000000p-6, -0x1.3829e60000000p-4, 0x1.b281900000000p-6, 0x1.1644fe0000000p-4, -0x1.a92ff80000000p-7, 0x1.a056300000000p-3}
}
, {{-0x1.e3eeda0000000p-4, 0x1.b72f8c0000000p-8, -0x1.ee8f0a0000000p-6, -0x1.11b27c0000000p-7, 0x1.e7e3000000000p-4, 0x1.4f13900000000p-5, -0x1.d73e2c0000000p-4, 0x1.72b7020000000p-4, -0x1.08b41e0000000p-5, -0x1.b593f80000000p-11, -0x1.21261a0000000p-4, 0x1.6fda380000000p-2, -0x1.efa64a0000000p-3, 0x1.4f82960000000p-5, 0x1.ed2f9e0000000p-5, -0x1.e02b320000000p-4, -0x1.6178fe0000000p-6, 0x1.6f4b800000000p-4, 0x1.f4bc700000000p-3, -0x1.fd12020000000p-3, 0x1.e35e5e0000000p-3, -0x1.585f940000000p-4, 0x1.b14afa0000000p-3, -0x1.e24ae00000000p-4, -0x1.a6415e0000000p-4, -0x1.4185f60000000p-3, 0x1.b84f2a0000000p-3, -0x1.99b6c60000000p-3, 0x1.e60e280000000p-3, 0x1.ccb2ee0000000p-4, -0x1.4b4a480000000p-5, 0x1.42850e0000000p-4}
, {0x1.b7204c0000000p-5, -0x1.e3869e0000000p-4, 0x1.38c4b00000000p-7, -0x1.be820c0000000p-4, -0x1.7923e40000000p-4, -0x1.1c3dae0000000p-3, 0x1.9c50a40000000p-3, 0x1.ad5f240000000p-3, -0x1.d9be100000000p-9, 0x1.48c5300000000p-5, -0x1.78a15e0000000p-9, 0x1.ed67f60000000p-5, -0x1.376a580000000p-9, -0x1.0e28960000000p-2, -0x1.470d900000000p-4, -0x1.33964e0000000p-4, -0x1.a926020000000p-5, -0x1.664b9e0000000p-3, -0x1.3cff020000000p-7, 0x1.614d860000000p-6, -0x1.71ab640000000p-3, 0x1.519ab00000000p-4, -0x1.3ce8e40000000p-6, -0x1.47cb7c0000000p-9, -0x1.4c6b980000000p-4, -0x1.91d9dc0000000p-9, 0x1.c7eabc0000000p-5, 0x1.1c208a0000000p-3, -0x1.463d460000000p-8, -0x1.482d3e0000000p-3, -0x1.a6886a0000000p-5, -0x1.2e3cc20000000p-3}
, {-0x1.66cf120000000p-3, 0x1.bf51ee0000000p-3, 0x1.434b640000000p-4, -0x1.863cb00000000p-3, -0x1.f4abc40000000p-3, 0x1.a6ada40000000p-3, -0x1.3213fe0000000p-6, -0x1.773d9e0000000p-3, 0x1.9ef61e0000000p-7, 0x1.aabcde0000000p-4, 0x1.e317de0000000p-4, -0x1.673f120000000p-3, -0x1.d937a60000000p-7, 0x1.8014b20000000p-4, -0x1.869de80000000p-5, -0x1.74cca80000000p-4, -0x1.233c640000000p-7, -0x1.5251280000000p-3, -0x1.9464860000000p-3, -0x1.3285fc0000000p-4, -0x1.87c7b60000000p-4, 0x1.7e920e0000000p-3, 0x1.b680420000000p-6, 0x1.8844bc0000000p-5, 0x1.66b3a40000000p-3, 0x1.1ab88c0000000p-3, -0x1.31bdbc0000000p-2, 0x1.f69bfa0000000p-6, -0x1.6351cc0000000p-3, -0x1.591ca20000000p-6, -0x1.60ee640000000p-3, -0x1.29f1a60000000p-6}
}
, {{0x1.6b2dde0000000p-5, 0x1.2b78360000000p-3, -0x1.52c72a0000000p-3, 0x1.5e03240000000p-4, -0x1.06d2540000000p-4, -0x1.67ee620000000p-3, -0x1.1718fa0000000p-3, -0x1.3045fc0000000p-2, 0x1.eea4d40000000p-4, 0x1.a0f6180000000p-5, -0x1.41c60c0000000p-6, -0x1.f4cd1c0000000p-3, 0x1.17e1060000000p-3, -0x1.c801ce0000000p-3, 0x1.c6a8a60000000p-6, -0x1.0554700000000p-2, -0x1.c3cea20000000p-4, -0x1.2fc8bc0000000p-2, -0x1.9da99a0000000p-4, -0x1.5e622c0000000p-3, -0x1.1611700000000p-2, 0x1.38bdac0000000p-4, -0x1.911e620000000p-3, 0x1.c3baa40000000p-4, -0x1.6f14020000000p-4, 0x1.c9ce0c0000000p-3, 0x1.b32b880000000p-4, -0x1.b732dc0000000p-4, 0x1.87ae5a0000000p-4, -0x1.57872c0000000p-3, 0x1.9784760000000p-4, 0x1.62cc860000000p-3}
, {0x1.561fa80000000p-4, 0x1.42d6820000000p-3, -0x1.c2c4d60000000p-6, -0x1.80312e0000000p-4, 0x1.21b1e20000000p-5, 0x1.b664580000000p-4, 0x1.a02c4a0000000p-6, -0x1.e4ea8c0000000p-5, 0x1.ed262c0000000p-6, 0x1.baa06e0000000p-5, -0x1.4a2b4a0000000p-5, -0x1.47d98e0000000p-5, -0x1.0adf2e0000000p-9, -0x1.986d960000000p-3, 0x1.1fa7540000000p-3, 0x1.de0d3e0000000p-4, 0x1.7f43720000000p-4, 0x1.7835ba0000000p-6, -0x1.68ca260000000p-6, -0x1.ff275c0000000p-5, -0x1.8e64a20000000p-4, 0x1.4ffe080000000p-3, -0x1.f0c6c00000000p-9, -0x1.4d54e80000000p-4, -0x1.1aea1e0000000p-4, 0x1.6288100000000p-7, -0x1.2cc3fa0000000p-5, 0x1.b3f00c0000000p-4, 0x1.02cf1a0000000p-3, -0x1.ff11f00000000p-3, 0x1.38204a0000000p-3, 0x1.434cbc0000000p-5}
, {0x1.91b3ae0000000p-3, -0x1.a48c2e0000000p-3, 0x1.58058c0000000p-7, -0x1.45dd4e0000000p-3, 0x1.1eff620000000p-5, 0x1.f1a8f80000000p-3, -0x1.849c7e0000000p-5, 0x1.3051180000000p-2, 0x1.6bb31a0000000p-5, 0x1.f89d4c0000000p-4, 0x1.52260a0000000p-5, 0x1.90d6ae0000000p-3, 0x1.ef7fa60000000p-4, 0x1.b517880000000p-4, 0x1.93576a0000000p-4, 0x1.df0ff20000000p-3, -0x1.b370920000000p-4, 0x1.0174d60000000p-3, 0x1.9bf54a0000000p-3, 0x1.5b12960000000p-3, 0x1.7ca6500000000p-4, -0x1.af631a0000000p-3, 0x1.d8a7f60000000p-4, -0x1.5b67dc0000000p-3, -0x1.5969980000000p-3, -0x1.10f6ee0000000p-2, 0x1.38852a0000000p-3, 0x1.f9b3400000000p-4, 0x1.88d3520000000p-3, -0x1.1823a80000000p-5, -0x1.71c74c0000000p-5, -0x1.becf2e0000000p-4}
}
, {{-0x1.c243e20000000p-6, -0x1.0143760000000p-3, -0x1.278b5c0000000p-3, 0x1.0864fc0000000p-5, 0x1.d7f6da0000000p-4, 0x1.4c34ca0000000p-4, -0x1.6058960000000p-3, -0x1.4520200000000p-5, 0x1.a68a8a0000000p-4, 0x1.bfe40c0000000p-3, -0x1.547e2c0000000p-2, 0x1.bf98900000000p-3, -0x1.20512a0000000p-3, 0x1.091aa60000000p-5, 0x1.d09c380000000p-3, 0x1.a97cbc0000000p-4, -0x1.28b55a0000000p-3, -0x1.551fe80000000p-3, -0x1.0c60820000000p-4, -0x1.534dde0000000p-4, 0x1.fdc2ba0000000p-3, -0x1.4071fa0000000p-3, -0x1.feb4c20000000p-4, -0x1.e89a980000000p-3, 0x1.b0c6ce0000000p-3, -0x1.506c520000000p-5, 0x1.78b8b60000000p-4, -0x1.a3941a0000000p-5, -0x1.2d18bc0000000p-4, -0x1.83ef7c0000000p-3, -0x1.b13a180000000p-5, -0x1.35ced60000000p-4}
, {0x1.0ee68e0000000p-4, 0x1.0a42720000000p-3, -0x1.31ba260000000p-3, 0x1.9f3fee0000000p-3, 0x1.73bdc40000000p-6, 0x1.aeb1da0000000p-4, -0x1.20fd920000000p-4, 0x1.3d50420000000p-3, 0x1.7c27860000000p-4, 0x1.a751d60000000p-4, -0x1.6dd1a60000000p-3, 0x1.e5a1060000000p-4, -0x1.e895680000000p-6, 0x1.39f3e40000000p-3, 0x1.b389660000000p-6, 0x1.069d920000000p-4, -0x1.c5fcde0000000p-3, -0x1.92b3b00000000p-3, 0x1.13ffb80000000p-3, 0x1.84a4a60000000p-9, 0x1.7c2b060000000p-5, -0x1.77782a0000000p-3, -0x1.424a940000000p-3, -0x1.54e2c20000000p-6, -0x1.0591c20000000p-4, -0x1.27edb80000000p-3, -0x1.a6259a0000000p-6, -0x1.5333840000000p-7, -0x1.33382c0000000p-4, -0x1.863f960000000p-3, 0x1.8ac9c40000000p-5, 0x1.679cfc0000000p-7}
, {0x1.596b200000000p-4, 0x1.6fceb60000000p-3, -0x1.0a652a0000000p-3, -0x1.483f4e0000000p-3, 0x1.28d42c0000000p-5, -0x1.c515260000000p-5, -0x1.62ee340000000p-3, 0x1.082e640000000p-2, 0x1.3e5bac0000000p-4, 0x1.7462820000000p-2, -0x1.89c14a0000000p-3, 0x1.5039220000000p-5, 0x1.a47d4a0000000p-3, 0x1.8dd3060000000p-5, -0x1.7c87c40000000p-5, 0x1.045a440000000p-7, -0x1.d1413e0000000p-3, -0x1.569b560000000p-5, -0x1.8a1cc60000000p-3, -0x1.2032500000000p-6, 0x1.0777340000000p-5, 0x1.715dc20000000p-6, 0x1.239a9e0000000p-4, -0x1.a9edbe0000000p-3, -0x1.18379e0000000p-4, -0x1.07e8260000000p-4, 0x1.3002f00000000p-5, 0x1.67f8560000000p-4, 0x1.ed98420000000p-3, -0x1.5417ce0000000p-3, -0x1.420cfa0000000p-3, -0x1.e94d1c0000000p-6}
}
, {{-0x1.3afa140000000p-2, 0x1.2d3dfc0000000p-7, -0x1.1282ae0000000p-3, -0x1.cc7eb80000000p-3, -0x1.59a0120000000p-2, -0x1.1d796a0000000p-3, 0x1.29d8d40000000p-4, -0x1.0b4c2e0000000p-3, 0x1.f03b5e0000000p-8, -0x1.7fc0d60000000p-3, 0x1.6b165e0000000p-5, -0x1.6070e60000000p-3, -0x1.4d7b100000000p-3, -0x1.4014d00000000p-3, -0x1.dbeab60000000p-4, -0x1.87c1aa0000000p-2, -0x1.33650c0000000p-5, -0x1.1e9f4e0000000p-5, -0x1.f3d38c0000000p-4, -0x1.a649400000000p-4, -0x1.3c3c260000000p-3, -0x1.f94fc40000000p-3, 0x1.441aa80000000p-4, -0x1.e01f520000000p-5, -0x1.53cba60000000p-4, 0x1.1e03f20000000p-8, -0x1.1535220000000p-3, 0x1.26e0de0000000p-3, 0x1.02065a0000000p-4, 0x1.798ad80000000p-4, 0x1.dff1200000000p-4, -0x1.9855e60000000p-4}
, {-0x1.4caa8e0000000p-3, 0x1.565bfa0000000p-3, -0x1.99dc1c0000000p-6, -0x1.6c7e680000000p-3, -0x1.238e0e0000000p-4, -0x1.d8c7fc0000000p-5, -0x1.c4779e0000000p-4, -0x1.973dac0000000p-5, 0x1.4fd8f00000000p-5, -0x1.d9a4ee0000000p-10, 0x1.1e383e0000000p-3, -0x1.2e1ef80000000p-4, -0x1.979eac0000000p-5, -0x1.5e4dcc0000000p-5, -0x1.c5da1a0000000p-4, -0x1.f2caba0000000p-4, -0x1.d131660000000p-4, -0x1.2ad47a0000000p-3, -0x1.7c90720000000p-7, -0x1.0c178a0000000p-5, -0x1.f64c4e0000000p-3, -0x1.39d18a0000000p-5, 0x1.8848be0000000p-3, -0x1.6cc4ee0000000p-4, -0x1.0f751e0000000p-3, 0x1.237ea80000000p-6, -0x1.e583700000000p-4, -0x1.6a4f200000000p-4, 0x1.ceb60e0000000p-3, -0x1.c7ce920000000p-4, 0x1.c6f54e0000000p-4, 0x1.62c30a0000000p-3}
, {-0x1.6cdfe20000000p-4, 0x1.0f343e0000000p-4, 0x1.4013d40000000p-9, -0x1.1634de0000000p-2, -0x1.e1bca80000000p-3, -0x1.4cf7ea0000000p-6, 0x1.99e6260000000p-3, -0x1.b10d240000000p-8, -0x1.61b7c00000000p-4, -0x1.f394a00000000p-7, 0x1.92a2ce0000000p-4, -0x1.84b37c0000000p-3, -0x1.e330d60000000p-4, 0x1.83389a0000000p-8, -0x1.58a26a0000000p-5, -0x1.948a000000000p-3, -0x1.22165e0000000p-2, -0x1.9951c80000000p-5, -0x1.688bba0000000p-9, 0x1.0137920000000p-3, -0x1.b789de0000000p-4, 0x1.692c740000000p-3, 0x1.1ac0320000000p-3, 0x1.382d220000000p-3, 0x1.c206100000000p-4, -0x1.ad8bc20000000p-4, -0x1.7e8b320000000p-3, -0x1.d8076c0000000p-5, 0x1.3fa5480000000p-3, 0x1.9cac500000000p-3, 0x1.28117e0000000p-3, 0x1.24531a0000000p-4}
}
, {{0x1.a784a60000000p-3, 0x1.064fe00000000p-3, 0x1.16aa800000000p-3, -0x1.d821280000000p-3, -0x1.c7d7f60000000p-2, 0x1.c47aac0000000p-4, -0x1.624f7c0000000p-3, -0x1.3259fc0000000p-3, -0x1.ac8cfc0000000p-9, 0x1.471ba40000000p-5, -0x1.187a8e0000000p-3, -0x1.c781520000000p-6, 0x1.ea24280000000p-5, -0x1.9f861a0000000p-6, 0x1.26c8c40000000p-3, -0x1.5088620000000p-3, 0x1.c886400000000p-4, 0x1.a4f4d40000000p-4, -0x1.458b4c0000000p-3, -0x1.dd0cb20000000p-3, -0x1.2c102c0000000p-3, -0x1.13fa7e0000000p-2, 0x1.0c77620000000p-5, 0x1.c377840000000p-3, -0x1.479fc80000000p-2, -0x1.89f8540000000p-2, -0x1.aee4620000000p-3, -0x1.3ec72a0000000p-2, -0x1.5d91300000000p-4, -0x1.fd2a1a0000000p-3, -0x1.61b69e0000000p-5, -0x1.5af9740000000p-3}
, {0x1.dff98a0000000p-4, 0x1.c488ea0000000p-4, -0x1.2079420000000p-5, -0x1.8a92ea0000000p-6, 0x1.8a62940000000p-9, 0x1.5da18c0000000p-3, -0x1.17b7bc0000000p-3, 0x1.a3d81e0000000p-4, -0x1.dc337e0000000p-3, 0x1.859b980000000p-4, -0x1.e6e8ea0000000p-3, -0x1.eae62a0000000p-6, 0x1.2e2db00000000p-2, -0x1.95cba40000000p-8, 0x1.06c3960000000p-3, 0x1.a5a1120000000p-4, -0x1.e836980000000p-5, -0x1.1963de0000000p-3, -0x1.2687620000000p-3, 0x1.431ed80000000p-5, 0x1.40a95a0000000p-5, 0x1.de44560000000p-5, -0x1.2720ba0000000p-4, -0x1.2f4d680000000p-2, -0x1.e6a4a40000000p-3, -0x1.b5064a0000000p-5, -0x1.ff948c0000000p-3, -0x1.539b980000000p-4, 0x1.00511a0000000p-4, -0x1.0628180000000p-4, 0x1.09fb2c0000000p-5, 0x1.48f8d40000000p-3}
, {-0x1.ce25880000000p-6, -0x1.c305a80000000p-2, -0x1.7c00bc0000000p-4, -0x1.33762e0000000p-2, 0x1.9d7f440000000p-7, -0x1.5c880a0000000p-5, -0x1.69ddb40000000p-3, 0x1.ab26560000000p-5, -0x1.5cc4400000000p-3, 0x1.b4ff520000000p-3, -0x1.ffdbc40000000p-3, 0x1.3e670e0000000p-5, 0x1.74b79e0000000p-5, 0x1.655b100000000p-5, -0x1.79285c0000000p-5, 0x1.ce0d9c0000000p-4, -0x1.a36b4c0000000p-5, 0x1.b313a40000000p-4, -0x1.63faac0000000p-3, 0x1.050fbe0000000p-3, 0x1.36abd40000000p-3, -0x1.e42eac0000000p-5, -0x1.76d4240000000p-4, -0x1.f8ac760000000p-4, -0x1.16a3a20000000p-2, -0x1.1021d40000000p-9, -0x1.0839ee0000000p-2, -0x1.6242600000000p-4, -0x1.268ee80000000p-6, 0x1.f1a9c60000000p-6, -0x1.3f8d9c0000000p-4, -0x1.a53d180000000p-3}
}
, {{0x1.4d438c0000000p-4, -0x1.f048300000000p-5, -0x1.4375120000000p-3, -0x1.b7ac620000000p-3, 0x1.89b8b20000000p-7, 0x1.62eb5a0000000p-4, -0x1.ce46180000000p-4, 0x1.ace5760000000p-3, 0x1.86f7e40000000p-6, -0x1.9f2cde0000000p-3, 0x1.83fef80000000p-6, 0x1.fddb3c0000000p-4, 0x1.dbdf400000000p-7, 0x1.acac080000000p-7, -0x1.5e2fdc0000000p-3, 0x1.0c5a220000000p-4, -0x1.07742e0000000p-3, -0x1.b1b7700000000p-8, -0x1.1fa1840000000p-3, 0x1.d87d060000000p-4, -0x1.6f0da60000000p-4, -0x1.a49e2c0000000p-3, 0x1.7466660000000p-2, -0x1.173b320000000p-3, 0x1.1a3d3e0000000p-3, -0x1.7f89260000000p-3, 0x1.f360ee0000000p-5, 0x1.ab70b00000000p-4, 0x1.cad1240000000p-4, -0x1.6d41960000000p-2, 0x1.211cb80000000p-3, 0x1.1512dc0000000p-2}
, {-0x1.4d949a0000000p-6, 0x1.81b8780000000p-3, -0x1.b2d0a80000000p-4, 0x1.cd74640000000p-4, -0x1.2120a80000000p-7, 0x1.e8170c0000000p-4, 0x1.a45b640000000p-6, 0x1.87d3c40000000p-3, 0x1.c337520000000p-3, 0x1.6f14880000000p-6, -0x1.5af9920000000p-5, 0x1.e432da0000000p-5, -0x1.2cae0c0000000p-4, 0x1.f156160000000p-7, -0x1.0bda780000000p-5, 0x1.bacd6a0000000p-6, -0x1.ac6b380000000p-3, 0x1.14e9140000000p-6, -0x1.8cf24a0000000p-4, 0x1.7f193c0000000p-4, -0x1.edd3e80000000p-4, -0x1.70441a0000000p-6, -0x1.420fa40000000p-5, -0x1.a0544a0000000p-6, 0x1.806eaa0000000p-4, -0x1.1e21fe0000000p-8, 0x1.68c7860000000p-5, 0x1.8f21440000000p-5, -0x1.fb7eb20000000p-5, -0x1.0137240000000p-2, 0x1.09acce0000000p-9, 0x1.54fdf80000000p-2}
, {-0x1.8e12a40000000p-1, 0x1.09f5b20000000p-2, 0x1.1baac20000000p-5, 0x1.8a155e0000000p-3, -0x1.4e66160000000p-5, -0x1.16c7660000000p-2, -0x1.a2f4180000000p-4, -0x1.0fa8020000000p-2, 0x1.2de17a0000000p-3, -0x1.ea286e0000000p-3, 0x1.2330540000000p-3, -0x1.2ba9180000000p-2, -0x1.02e70e0000000p-1, 0x1.75be140000000p-6, -0x1.4fd01e0000000p-3, 0x1.2a1a0a0000000p-5, 0x1.3cd4f00000000p-2, -0x1.06231e0000000p-2, 0x1.92a3220000000p-5, 0x1.9f0cf40000000p-4, -0x1.2518c40000000p-2, -0x1.5a61480000000p-4, -0x1.9128340000000p-3, 0x1.0a2acc0000000p-3, -0x1.03030a0000000p-7, 0x1.0ce53a0000000p-2, -0x1.78852e0000000p-3, 0x1.d289620000000p-7, -0x1.a4d7700000000p-4, 0x1.18728c0000000p-3, -0x1.9ec8bc0000000p-3, -0x1.5833780000000p-3}
}
, {{-0x1.2f0a640000000p-3, 0x1.0fe3800000000p-3, -0x1.6868940000000p-5, 0x1.57a7fc0000000p-3, 0x1.1153220000000p-2, -0x1.1bb1fc0000000p-3, -0x1.9d8a040000000p-3, -0x1.9cfe800000000p-5, 0x1.03399a0000000p-2, 0x1.7dff180000000p-7, -0x1.8feb5c0000000p-8, -0x1.63f5120000000p-3, -0x1.4dfc3e0000000p-4, 0x1.0543440000000p-3, 0x1.2726000000000p-6, 0x1.e8eee80000000p-5, 0x1.bd52180000000p-5, -0x1.29cf7c0000000p-3, 0x1.04e7040000000p-10, 0x1.2f56280000000p-3, -0x1.ef9d360000000p-4, 0x1.2b52440000000p-3, -0x1.792e440000000p-3, -0x1.16ca720000000p-3, 0x1.8fac2c0000000p-3, 0x1.a1d33a0000000p-3, 0x1.357a880000000p-7, 0x1.1383f80000000p-3, 0x1.e3802e0000000p-7, -0x1.e896180000000p-4, -0x1.21f17c0000000p-5, 0x1.0d6a320000000p-3}
, {-0x1.94a80a0000000p-3, -0x1.b65faa0000000p-6, -0x1.d4c1fa0000000p-5, 0x1.27aaf80000000p-3, -0x1.ca269a0000000p-6, -0x1.bfa6140000000p-4, -0x1.b582960000000p-5, -0x1.9816ac0000000p-3, -0x1.5ead900000000p-8, 0x1.512b3c0000000p-3, 0x1.a739ea0000000p-5, -0x1.7bfaec0000000p-3, 0x1.07a8620000000p-2, -0x1.b0e54e0000000p-6, -0x1.b01ace0000000p-6, -0x1.ee7c580000000p-4, 0x1.7fa1be0000000p-6, 0x1.df36c20000000p-6, -0x1.0a59e40000000p-7, -0x1.cb5c8e0000000p-4, -0x1.8d7aee0000000p-3, 0x1.da1ca60000000p-6, -0x1.08b26c0000000p-4, 0x1.6453b40000000p-3, 0x1.ad84bc0000000p-4, 0x1.4cd9ca0000000p-5, -0x1.dee7860000000p-8, 0x1.b5df180000000p-6, -0x1.604ad80000000p-3, 0x1.bb72c40000000p-3, 0x1.23511e0000000p-3, -0x1.a6d5400000000p-5}
, {0x1.7213960000000p-4, -0x1.0e1c260000000p-6, -0x1.06e0d60000000p-4, -0x1.0d90940000000p-5, -0x1.7093920000000p-4, 0x1.6d00aa0000000p-2, 0x1.62e8d80000000p-6, 0x1.2798c80000000p-2, 0x1.6ec2dc0000000p-5, -0x1.91c31e0000000p-6, -0x1.4a91800000000p-6, 0x1.c94ebe0000000p-6, 0x1.2a57d40000000p-4, -0x1.2112dc0000000p-2, -0x1.c373ca0000000p-5, 0x1.c9c26c0000000p-5, 0x1.b5063a0000000p-5, -0x1.80645a0000000p-8, 0x1.cad93a0000000p-3, -0x1.84220e0000000p-3, 0x1.0303900000000p-4, 0x1.22b1080000000p-5, -0x1.d1d9900000000p-4, -0x1.dc1aee0000000p-5, -0x1.d317d80000000p-7, -0x1.6acf1a0000000p-4, 0x1.5d1e4a0000000p-3, 0x1.d9ad200000000p-5, 0x1.08e1b20000000p-2, -0x1.3f66560000000p-5, 0x1.37ba4a0000000p-3, 0x1.ce80220000000p-4}
}
, {{-0x1.8147920000000p-3, 0x1.12442a0000000p-2, 0x1.ab23c00000000p-4, 0x1.af47520000000p-4, -0x1.42f8bc0000000p-2, 0x1.8659a00000000p-4, -0x1.8ce0420000000p-2, -0x1.4ca53e0000000p-3, -0x1.3ced8c0000000p-5, 0x1.46bd8a0000000p-2, -0x1.6faa5e0000000p-2, -0x1.c295940000000p-4, -0x1.3b8b1a0000000p-8, -0x1.c843420000000p-5, 0x1.588b4e0000000p-3, -0x1.69b2380000000p-2, -0x1.75bb500000000p-4, -0x1.4055320000000p-2, -0x1.05fd6a0000000p-3, 0x1.6a74880000000p-6, -0x1.b9b0720000000p-9, -0x1.185df60000000p-3, -0x1.15531c0000000p-2, -0x1.710e3a0000000p-8, 0x1.2f58ec0000000p-7, 0x1.c79ff20000000p-4, 0x1.031dcc0000000p-3, -0x1.17be700000000p-2, -0x1.8f092a0000000p-2, 0x1.a78d4a0000000p-3, -0x1.67134e0000000p-3, -0x1.393c640000000p-6}
, {-0x1.6970f20000000p-5, -0x1.19ebdc0000000p-2, -0x1.1c14280000000p-3, -0x1.ae4aac0000000p-5, 0x1.235fdc0000000p-3, 0x1.07ddae0000000p-3, -0x1.26fc0c0000000p-5, -0x1.2bdb3a0000000p-4, -0x1.07eb2a0000000p-5, 0x1.9a71d40000000p-3, 0x1.0fa5720000000p-4, -0x1.51b00e0000000p-3, 0x1.236c760000000p-7, -0x1.1c6c960000000p-4, 0x1.12afc00000000p-13, 0x1.537e760000000p-4, -0x1.82e69e0000000p-6, -0x1.f4fc1e0000000p-4, 0x1.3f337a0000000p-3, -0x1.1674520000000p-6, 0x1.5717e80000000p-3, -0x1.8e19a20000000p-3, -0x1.6c40cc0000000p-5, -0x1.76f0940000000p-3, 0x1.1bbe4c0000000p-4, -0x1.873b8e0000000p-3, 0x1.d9373a0000000p-4, -0x1.b2427e0000000p-4, 0x1.eada940000000p-5, 0x1.3453540000000p-5, -0x1.7b0b5a0000000p-7, 0x1.c6fe1c0000000p-4}
, {-0x1.2818f80000000p-3, -0x1.6c8df20000000p-3, -0x1.8559660000000p-2, -0x1.3faa8e0000000p-6, 0x1.3ef7300000000p-3, 0x1.38c2e00000000p-5, 0x1.2335720000000p-4, 0x1.3c74260000000p-2, 0x1.500ece0000000p-3, 0x1.74528c0000000p-7, 0x1.90607e0000000p-6, 0x1.4763020000000p-3, -0x1.f5fbf20000000p-5, -0x1.252e3a0000000p-4, -0x1.1dc0380000000p-3, 0x1.bb84440000000p-5, 0x1.f8d5a40000000p-5, 0x1.5207480000000p-2, -0x1.92a4a80000000p-5, 0x1.d049420000000p-9, 0x1.8cc9a40000000p-4, 0x1.8a73f20000000p-5, 0x1.bd4fe60000000p-3, -0x1.ff54640000000p-4, 0x1.20ea820000000p-4, 0x1.0515f40000000p-5, 0x1.0e6c060000000p-2, 0x1.8e105e0000000p-3, 0x1.15c31c0000000p-2, -0x1.9e0d6c0000000p-2, -0x1.b6c9d40000000p-4, 0x1.bc2fe00000000p-2}
}
, {{-0x1.47a2cc0000000p-2, 0x1.152ede0000000p-2, 0x1.3f9b3a0000000p-2, 0x1.bfdca60000000p-6, -0x1.85aaf20000000p-2, -0x1.5909040000000p-3, -0x1.ea774c0000000p-3, -0x1.8e32420000000p-2, 0x1.4c4c160000000p-8, -0x1.9754540000000p-3, -0x1.34ae1e0000000p-3, -0x1.381d360000000p-3, 0x1.02857e0000000p-2, -0x1.143a580000000p-4, 0x1.2567580000000p-4, -0x1.1c14880000000p-4, -0x1.7b562e0000000p-4, -0x1.cead460000000p-3, -0x1.02d5380000000p-4, -0x1.0251880000000p-5, -0x1.be3f220000000p-8, 0x1.cdfd4c0000000p-5, -0x1.3781000000000p-2, 0x1.8a39f40000000p-4, -0x1.871aea0000000p-2, 0x1.78ffa40000000p-4, -0x1.37c0de0000000p-5, -0x1.031a340000000p-2, -0x1.3f2ac20000000p-3, 0x1.72c7940000000p-3, -0x1.2e23c20000000p-3, -0x1.3095700000000p-3}
, {0x1.4818f40000000p-3, -0x1.042b920000000p-5, 0x1.580cf40000000p-3, -0x1.506e340000000p-2, 0x1.4c57c00000000p-4, 0x1.8b46aa0000000p-6, -0x1.2ca9a20000000p-3, 0x1.be0ad40000000p-7, -0x1.4c6f5c0000000p-2, 0x1.b9254e0000000p-3, -0x1.15f3040000000p-3, 0x1.bad32a0000000p-4, -0x1.b838840000000p-5, -0x1.a9be360000000p-4, -0x1.b0f6c20000000p-4, 0x1.391ad60000000p-3, 0x1.32fa8c0000000p-4, 0x1.82e0620000000p-6, 0x1.69dca80000000p-3, -0x1.2cc6120000000p-3, 0x1.5c88a60000000p-3, -0x1.2029100000000p-3, 0x1.3686f20000000p-4, -0x1.012b160000000p-7, -0x1.002ce00000000p-8, -0x1.f19f160000000p-3, 0x1.8b498e0000000p-4, 0x1.a9c84c0000000p-4, -0x1.05737a0000000p-3, 0x1.cd9d720000000p-3, -0x1.43e6520000000p-5, 0x1.acf8040000000p-4}
, {0x1.297faa0000000p-3, -0x1.842e520000000p-4, -0x1.b243f20000000p-5, 0x1.d200f80000000p-8, 0x1.2330ba0000000p-4, -0x1.64a1360000000p-3, -0x1.1529160000000p-4, 0x1.65aec80000000p-2, -0x1.086a780000000p-3, 0x1.1a99a40000000p-4, -0x1.cadd740000000p-5, 0x1.099b120000000p-2, 0x1.79e8220000000p-5, -0x1.77a69a0000000p-3, -0x1.32e8580000000p-3, 0x1.aae0040000000p-7, 0x1.53a0760000000p-3, 0x1.05c5460000000p-3, 0x1.54cd840000000p-3, -0x1.39361c0000000p-2, -0x1.18636c0000000p-5, -0x1.152d0e0000000p-3, 0x1.bff27e0000000p-5, -0x1.3059d40000000p-3, -0x1.5a020c0000000p-3, -0x1.01e0140000000p-2, 0x1.97fadc0000000p-2, 0x1.c719d40000000p-8, 0x1.54de9a0000000p-3, -0x1.8eae940000000p-4, -0x1.9006720000000p-4, -0x1.5508da0000000p-4}
}
, {{-0x1.02c4560000000p-4, -0x1.2cd2ec0000000p-5, -0x1.6790420000000p-3, -0x1.18e44e0000000p-4, 0x1.c8b3940000000p-11, 0x1.e18b320000000p-5, -0x1.536f300000000p-5, -0x1.28a4a40000000p-8, 0x1.10a5160000000p-3, -0x1.9439c20000000p-5, 0x1.2143ae0000000p-3, 0x1.42f7d00000000p-8, 0x1.008bdc0000000p-5, 0x1.48a42c0000000p-3, -0x1.e156600000000p-4, -0x1.2cec1a0000000p-9, -0x1.03f8880000000p-2, -0x1.b6da540000000p-8, -0x1.c061260000000p-4, 0x1.7772320000000p-3, 0x1.2d4af80000000p-5, 0x1.a985a00000000p-5, -0x1.ede2c20000000p-5, -0x1.2a050c0000000p-5, 0x1.cd3c220000000p-3, 0x1.5c8d8c0000000p-3, -0x1.57d6d80000000p-3, 0x1.03cd3a0000000p-3, 0x1.8739040000000p-3, 0x1.e933f00000000p-5, 0x1.0faf020000000p-3, 0x1.cbb1b40000000p-6}
, {-0x1.adfd660000000p-4, -0x1.4ed1020000000p-3, 0x1.f7690a0000000p-7, 0x1.c9b3560000000p-5, -0x1.1c8d6c0000000p-3, 0x1.f4258e0000000p-4, 0x1.9e773e0000000p-6, -0x1.43707c0000000p-4, 0x1.46699e0000000p-3, -0x1.1f79be0000000p-3, -0x1.1580be0000000p-3, -0x1.1b80fe0000000p-3, 0x1.350aaa0000000p-3, 0x1.767fe20000000p-6, 0x1.1efcac0000000p-5, -0x1.40c3da0000000p-5, -0x1.69164a0000000p-5, 0x1.8704f40000000p-5, -0x1.87081e0000000p-4, 0x1.4adfd60000000p-3, -0x1.2585c20000000p-3, 0x1.9b67e80000000p-6, -0x1.d5d3e60000000p-4, -0x1.8afb8e0000000p-3, 0x1.b069ae0000000p-3, 0x1.19a51c0000000p-4, -0x1.20cd3a0000000p-4, -0x1.32f27e0000000p-4, -0x1.27cdea0000000p-6, -0x1.2a6c460000000p-4, -0x1.18f6840000000p-7, 0x1.1860700000000p-4}
, {-0x1.343f660000000p-4, -0x1.37db080000000p-4, 0x1.6c74c40000000p-2, -0x1.60c9e00000000p-4, -0x1.066f3e0000000p-2, -0x1.1a65a40000000p-3, -0x1.4efe040000000p-2, 0x1.6342d60000000p-2, -0x1.9e51d20000000p-3, 0x1.4ef6240000000p-4, -0x1.0445480000000p-3, 0x1.aa17a40000000p-6, -0x1.0e1abe0000000p-3, -0x1.4177120000000p-6, -0x1.41001e0000000p-2, 0x1.e0e0760000000p-5, -0x1.eb3dc20000000p-3, -0x1.b690ee0000000p-3, 0x1.6251540000000p-6, -0x1.323cb00000000p-4, -0x1.259a7e0000000p-5, -0x1.398afa0000000p-5, -0x1.6c32940000000p-9, 0x1.5d0fde0000000p-3, -0x1.ce918a0000000p-2, 0x1.8f28f60000000p-3, -0x1.d8f4d40000000p-4, -0x1.1745da0000000p-2, 0x1.2c59d60000000p-4, 0x1.a554ec0000000p-2, -0x1.d278560000000p-3, -0x1.236ff20000000p-3}
}
, {{-0x1.535b3e0000000p-6, 0x1.bb2c540000000p-3, 0x1.80fbb00000000p-3, -0x1.253e7c0000000p-5, 0x1.eb14780000000p-6, -0x1.490c320000000p-3, -0x1.769c8c0000000p-3, -0x1.218ad00000000p-4, 0x1.42adc20000000p-4, -0x1.94dae00000000p-4, 0x1.7790000000000p-6, 0x1.b502460000000p-7, -0x1.4c2dda0000000p-4, -0x1.6cf8800000000p-6, -0x1.0a7ef20000000p-3, 0x1.2fd9620000000p-2, 0x1.a322d40000000p-4, -0x1.f0898c0000000p-4, 0x1.ac1b9c0000000p-4, -0x1.6101580000000p-3, 0x1.24a4ac0000000p-7, -0x1.2e67260000000p-3, 0x1.fafeb20000000p-4, -0x1.81978c0000000p-5, -0x1.bc29f60000000p-3, -0x1.7e168e0000000p-3, -0x1.032b5c0000000p-3, 0x1.89c14a0000000p-4, -0x1.a613e60000000p-7, -0x1.e334180000000p-6, -0x1.39e1620000000p-3, -0x1.4123720000000p-4}
, {-0x1.4d44960000000p-3, 0x1.5b07100000000p-3, 0x1.1acfb60000000p-2, -0x1.01dd220000000p-2, -0x1.56a37a0000000p-2, 0x1.a1520c0000000p-7, -0x1.4e420e0000000p-2, -0x1.73e2920000000p-4, 0x1.3fe0e80000000p-3, -0x1.10710e0000000p-2, -0x1.2725a80000000p-6, 0x1.6a24ca0000000p-5, -0x1.a3797a0000000p-3, -0x1.a2b15a0000000p-6, 0x1.4519c40000000p-5, -0x1.3601820000000p-3, -0x1.1803860000000p-3, 0x1.fe2a9a0000000p-7, -0x1.26fe720000000p-5, -0x1.f13bc40000000p-6, -0x1.b0f6280000000p-3, 0x1.ba4cae0000000p-6, 0x1.e15e760000000p-9, 0x1.1e95680000000p-2, -0x1.9d3b3e0000000p-6, -0x1.71435e0000000p-4, 0x1.e1f9920000000p-6, 0x1.cecd9c0000000p-4, 0x1.c00bae0000000p-5, 0x1.c7b08a0000000p-4, 0x1.0cad640000000p-5, -0x1.3271a20000000p-8}
, {0x1.504c820000000p-5, 0x1.658cda0000000p-5, -0x1.1aff260000000p-4, 0x1.d61ec00000000p-4, 0x1.0c2ea60000000p-4, 0x1.5e3eec0000000p-2, 0x1.14b0800000000p-4, 0x1.67acd40000000p-4, 0x1.4c87920000000p-4, -0x1.9b0fa80000000p-2, 0x1.3c09360000000p-3, 0x1.cd0ed40000000p-3, -0x1.4ae18e0000000p-4, 0x1.f60b040000000p-7, 0x1.9185500000000p-4, -0x1.8f77ee0000000p-3, 0x1.1ab96a0000000p-2, 0x1.a272d80000000p-6, 0x1.5834740000000p-3, -0x1.5309f40000000p-3, -0x1.16f9b80000000p-3, -0x1.3333580000000p-2, 0x1.0a2e580000000p-2, 0x1.8f21e60000000p-10, 0x1.ac34160000000p-4, 0x1.c031220000000p-7, 0x1.5768860000000p-2, -0x1.f96f6c0000000p-5, 0x1.22bec80000000p-2, -0x1.2bcb900000000p-6, 0x1.01fb420000000p-1, 0x1.355d240000000p-4}
}
, {{-0x1.93b0e40000000p-5, -0x1.2b79580000000p-3, 0x1.141fce0000000p-2, -0x1.2d29d80000000p-2, -0x1.005ce20000000p-5, 0x1.07a2ae0000000p-2, -0x1.f72d560000000p-4, 0x1.22e2da0000000p-4, -0x1.0ac5680000000p-3, 0x1.9ab3000000000p-3, -0x1.02936e0000000p-4, -0x1.1230240000000p-4, -0x1.1583f40000000p-4, -0x1.358c060000000p-3, -0x1.eaad300000000p-3, 0x1.0ef3960000000p-4, 0x1.513fd20000000p-2, 0x1.795de40000000p-5, -0x1.7721300000000p-3, -0x1.edc5f00000000p-3, -0x1.52a38a0000000p-3, -0x1.dec8920000000p-2, 0x1.19e8240000000p-2, -0x1.f0eab60000000p-3, -0x1.2bba580000000p-2, -0x1.3f69b20000000p-2, -0x1.c57f240000000p-5, 0x1.73aff60000000p-5, 0x1.eb39480000000p-5, 0x1.4efeee0000000p-4, -0x1.b101cc0000000p-4, -0x1.049e4e0000000p-3}
, {0x1.ae634a0000000p-4, -0x1.2804640000000p-4, 0x1.f2e7f00000000p-4, -0x1.f1bf2e0000000p-5, 0x1.a8c5220000000p-4, -0x1.3387fa0000000p-5, 0x1.17d4280000000p-4, -0x1.719d200000000p-3, -0x1.f98ece0000000p-3, 0x1.eb84200000000p-4, -0x1.42921a0000000p-5, 0x1.96c0be0000000p-7, 0x1.6dbaac0000000p-4, 0x1.4424240000000p-5, -0x1.f266c60000000p-7, -0x1.ad73c60000000p-6, 0x1.d853100000000p-3, -0x1.aa8b400000000p-3, 0x1.b3d4220000000p-6, -0x1.358e320000000p-2, 0x1.35c2500000000p-8, -0x1.a103dc0000000p-4, 0x1.b7d4080000000p-3, 0x1.0e16fe0000000p-7, -0x1.fa9fa00000000p-5, -0x1.7ed46c0000000p-7, -0x1.0fce3c0000000p-6, -0x1.8038480000000p-4, -0x1.91a4220000000p-4, 0x1.c49b820000000p-6, -0x1.122b8a0000000p-2, -0x1.c43dd00000000p-3}
, {0x1.dcd3140000000p-5, -0x1.31b3240000000p-4, -0x1.2d8e220000000p-3, 0x1.deb57e0000000p-3, -0x1.60fabc0000000p-4, -0x1.38679c0000000p-3, 0x1.50e63e0000000p-4, -0x1.c710240000000p-3, 0x1.fd3ff80000000p-4, 0x1.0e9f2a0000000p-4, 0x1.5419bc0000000p-3, -0x1.26a89e0000000p-2, 0x1.524d840000000p-3, 0x1.2a7d1c0000000p-4, -0x1.af1b7c0000000p-6, 0x1.83acc60000000p-5, -0x1.2cf2c80000000p-4, -0x1.a3459c0000000p-2, 0x1.0032ba0000000p-4, -0x1.1f6be60000000p-5, -0x1.389ea00000000p-5, 0x1.59b1860000000p-6, -0x1.15c63a0000000p-5, -0x1.0cdfc60000000p-4, 0x1.5fa7e20000000p-4, 0x1.36889c0000000p-4, -0x1.ac4fa20000000p-2, -0x1.2e725a0000000p-2, -0x1.6716500000000p-2, 0x1.94feec0000000p-3, -0x1.2f9f880000000p-3, -0x1.61c3760000000p-2}
}
, {{-0x1.a9af0c0000000p-3, 0x1.1535520000000p-2, -0x1.0b71b20000000p-3, 0x1.5e03d20000000p-3, -0x1.d0631e0000000p-3, -0x1.62fbe40000000p-3, -0x1.54ae200000000p-2, -0x1.422f820000000p-3, 0x1.6c4bb60000000p-3, -0x1.25041e0000000p-3, -0x1.a5089c0000000p-5, -0x1.98300e0000000p-3, 0x1.84413a0000000p-3, 0x1.7da1100000000p-4, -0x1.889bce0000000p-4, -0x1.3b34800000000p-3, -0x1.30d0f60000000p-2, -0x1.546c700000000p-2, -0x1.82e66c0000000p-6, 0x1.d9e8220000000p-3, -0x1.0daef80000000p-4, 0x1.0e4bb40000000p-2, -0x1.b04d6e0000000p-2, -0x1.e782c00000000p-4, -0x1.0cb2220000000p-8, 0x1.6b72c80000000p-2, -0x1.bd0af60000000p-3, -0x1.9e0fe40000000p-3, -0x1.d65e8a0000000p-4, -0x1.b94f0e0000000p-7, -0x1.ae58f00000000p-4, 0x1.530cac0000000p-5}
, {0x1.fe803a0000000p-7, -0x1.5fa2200000000p-6, 0x1.f8401a0000000p-6, -0x1.9803100000000p-4, -0x1.cc391c0000000p-4, -0x1.bb47ac0000000p-6, -0x1.dbca820000000p-3, 0x1.2f722e0000000p-3, 0x1.0d692c0000000p-4, -0x1.cf53360000000p-7, -0x1.5fb2220000000p-4, -0x1.f20ae20000000p-4, 0x1.22e7c00000000p-3, 0x1.edf25e0000000p-4, -0x1.f64da60000000p-3, 0x1.3e73420000000p-4, 0x1.7bab520000000p-6, -0x1.79ebae0000000p-3, -0x1.15db200000000p-3, -0x1.56b31a0000000p-6, 0x1.04be020000000p-7, 0x1.1dc9b80000000p-3, -0x1.1f0ebc0000000p-4, -0x1.1c91ca0000000p-4, -0x1.6e11300000000p-3, -0x1.3ea5300000000p-4, -0x1.6400ec0000000p-5, 0x1.9252ba0000000p-5, 0x1.19f4a00000000p-4, 0x1.191bf60000000p-6, -0x1.36dafe0000000p-3, 0x1.69c91c0000000p-3}
, {0x1.10b4480000000p-2, -0x1.aa14d40000000p-4, 0x1.b1253c0000000p-5, -0x1.ae790e0000000p-3, 0x1.7b98320000000p-4, 0x1.5df2780000000p-7, 0x1.deba9a0000000p-4, 0x1.180a0c0000000p-3, -0x1.d9598a0000000p-5, -0x1.96b8840000000p-7, -0x1.c8bbbe0000000p-6, 0x1.6ba25c0000000p-2, -0x1.25f28e0000000p-4, -0x1.32acf20000000p-3, -0x1.34858e0000000p-2, 0x1.0c1d420000000p-2, -0x1.048efc0000000p-13, -0x1.09a4e00000000p-3, 0x1.70dd160000000p-4, -0x1.a4895a0000000p-6, 0x1.983e340000000p-7, -0x1.341db00000000p-4, 0x1.ea3c940000000p-3, -0x1.4632ae0000000p-5, -0x1.a332460000000p-3, -0x1.82da040000000p-3, 0x1.dd53580000000p-3, 0x1.30c5760000000p-3, 0x1.dd9d6a0000000p-3, -0x1.0f391a0000000p-3, -0x1.87b3040000000p-4, 0x1.4e70ce0000000p-3}
}
, {{0x1.33a13e0000000p-3, -0x1.d19ed00000000p-3, 0x1.9e25d40000000p-3, -0x1.d73d440000000p-3, 0x1.519bd60000000p-3, -0x1.19ce120000000p-3, -0x1.8385480000000p-3, 0x1.0ff2320000000p-3, -0x1.ce25a40000000p-2, 0x1.af02300000000p-3, -0x1.27922c0000000p-1, 0x1.18f5300000000p-2, 0x1.cbfffa0000000p-8, 0x1.e1c9600000000p-6, 0x1.1e6e540000000p-4, 0x1.2c8b8c0000000p-2, 0x1.c0b88a0000000p-7, 0x1.6c783a0000000p-9, 0x1.01e3300000000p-6, 0x1.0edda20000000p-7, 0x1.0ca20a0000000p-2, -0x1.117f880000000p-4, 0x1.37a3180000000p-2, -0x1.f934d80000000p-3, -0x1.ac82e00000000p-7, -0x1.2080040000000p-2, -0x1.a3501c0000000p-3, 0x1.34bd940000000p-3, -0x1.60dd0c0000000p-4, 0x1.77670e0000000p-3, -0x1.2392ec0000000p-1, 0x1.378ec60000000p-2}
, {-0x1.bd67f00000000p-3, -0x1.b2de9a0000000p-3, 0x1.25eeda0000000p-4, -0x1.e52a160000000p-3, 0x1.45ec540000000p-4, -0x1.1da1be0000000p-3, 0x1.364f840000000p-4, 0x1.66beec0000000p-3, -0x1.14a2960000000p-1, 0x1.6deef60000000p-3, -0x1.2b89e20000000p-2, 0x1.63917c0000000p-3, -0x1.c0b3f40000000p-4, -0x1.81be940000000p-3, 0x1.33a5d80000000p-4, 0x1.9e2a4e0000000p-8, -0x1.671a140000000p-8, 0x1.86d3a40000000p-6, -0x1.0dfb8c0000000p-4, -0x1.2989c60000000p-5, 0x1.0551bc0000000p-6, -0x1.5fc4080000000p-3, 0x1.97154e0000000p-5, 0x1.eea43e0000000p-5, 0x1.9104440000000p-3, 0x1.4adfce0000000p-3, -0x1.8c68660000000p-3, -0x1.c528200000000p-11, -0x1.aa6d5a0000000p-4, 0x1.8c2cee0000000p-4, -0x1.65a7fa0000000p-4, -0x1.38f4ac0000000p-3}
, {-0x1.ae4ec00000000p-5, -0x1.88215c0000000p-5, 0x1.3d4fe40000000p-2, -0x1.5f48de0000000p-3, -0x1.329e8e0000000p-3, -0x1.42516a0000000p-5, 0x1.726aaa0000000p-8, 0x1.6d737c0000000p-3, -0x1.4383aa0000000p-2, 0x1.9603400000000p-3, -0x1.1d82d60000000p-2, -0x1.b169bc0000000p-5, -0x1.38a7760000000p-5, -0x1.45b4920000000p-2, -0x1.9db0b40000000p-4, -0x1.5f42580000000p-5, 0x1.852da20000000p-4, -0x1.3b4db80000000p-2, -0x1.892cba0000000p-3, -0x1.64404e0000000p-3, -0x1.903cdc0000000p-5, -0x1.d2a9c40000000p-3, -0x1.f14f340000000p-5, 0x1.a2bec60000000p-4, -0x1.11205e0000000p-2, -0x1.a287980000000p-6, -0x1.1535300000000p-4, -0x1.3e55560000000p-6, -0x1.690aa80000000p-3, 0x1.c840500000000p-3, -0x1.bac8ea0000000p-3, -0x1.535b360000000p-4}
}
, {{-0x1.30a0cc0000000p-3, 0x1.7f3adc0000000p-3, 0x1.f2e4680000000p-5, 0x1.4d8f480000000p-8, -0x1.143be80000000p-2, -0x1.65b4d60000000p-4, 0x1.99fdfc0000000p-4, -0x1.38d35a0000000p-4, 0x1.15801e0000000p-5, -0x1.4968d60000000p-2, -0x1.e532500000000p-7, -0x1.26fc9e0000000p-5, 0x1.c2457a0000000p-4, -0x1.f4b2380000000p-4, 0x1.6167ce0000000p-3, 0x1.729ca40000000p-5, 0x1.65cb9c0000000p-5, -0x1.fa4f140000000p-5, 0x1.16c2920000000p-3, -0x1.06426e0000000p-4, 0x1.c079bc0000000p-3, 0x1.10a4300000000p-2, -0x1.0566940000000p-2, 0x1.93ad800000000p-5, -0x1.2e9ba40000000p-3, 0x1.4f7dec0000000p-3, -0x1.9799a80000000p-5, 0x1.9b045a0000000p-3, 0x1.f237140000000p-4, -0x1.5235d20000000p-3, 0x1.ffe9100000000p-3, 0x1.27aa020000000p-3}
, {-0x1.66794c0000000p-3, -0x1.6b25ba0000000p-3, 0x1.2cacce0000000p-6, 0x1.b0e7140000000p-8, -0x1.ea9d3c0000000p-3, -0x1.4e3ef80000000p-5, 0x1.a0033e0000000p-4, 0x1.a0399a0000000p-4, -0x1.9e2b860000000p-5, -0x1.14d0cc0000000p-5, 0x1.0976180000000p-3, 0x1.32bb4c0000000p-8, -0x1.bd56660000000p-6, 0x1.53ab740000000p-4, 0x1.ecaff20000000p-4, 0x1.ecd9b80000000p-7, 0x1.706ee60000000p-6, 0x1.fafdc40000000p-4, 0x1.62bd100000000p-4, -0x1.cc56c60000000p-9, -0x1.d834c00000000p-4, 0x1.cc40180000000p-6, 0x1.0c6b1a0000000p-2, -0x1.7d6b280000000p-6, -0x1.2c597c0000000p-3, 0x1.13b85e0000000p-9, 0x1.4274ca0000000p-3, 0x1.2524a40000000p-5, 0x1.0cc14a0000000p-4, 0x1.0492d20000000p-5, -0x1.0c85780000000p-6, 0x1.6e0cf60000000p-6}
, {-0x1.a6f3ea0000000p-3, -0x1.f5f9960000000p-3, -0x1.0d26a20000000p-6, -0x1.8ba37c0000000p-3, -0x1.14cfe00000000p-2, -0x1.ba78f00000000p-4, 0x1.15ce5c0000000p-2, -0x1.b657cc0000000p-7, -0x1.7a3aa40000000p-5, -0x1.adecb20000000p-3, 0x1.0ba7760000000p-2, -0x1.6acee20000000p-5, -0x1.a98b4a0000000p-5, -0x1.11c8360000000p-2, 0x1.f0c8200000000p-7, -0x1.bf197e0000000p-5, 0x1.5fa1ea0000000p-4, -0x1.1976600000000p-5, 0x1.1ec0200000000p-2, -0x1.f059800000000p-8, -0x1.91648e0000000p-4, 0x1.609f820000000p-5, 0x1.dd4ec80000000p-3, 0x1.79fa340000000p-3, -0x1.6740c80000000p-4, 0x1.ca277e0000000p-4, -0x1.34f44c0000000p-2, 0x1.8a20d20000000p-3, 0x1.92052a0000000p-5, 0x1.0529020000000p-3, 0x1.9d381e0000000p-5, -0x1.c2708a0000000p-3}
}
, {{0x1.9c39920000000p-4, -0x1.bccc7e0000000p-7, -0x1.7f9bbe0000000p-5, 0x1.2c158e0000000p-4, 0x1.78e1340000000p-3, -0x1.7e92260000000p-6, 0x1.1bbd1c0000000p-4, 0x1.f41b140000000p-3, 0x1.f036fc0000000p-8, 0x1.db818e0000000p-3, 0x1.b22b600000000p-4, 0x1.3e27d80000000p-6, 0x1.780ac00000000p-3, -0x1.0b28840000000p-3, 0x1.0426200000000p-4, 0x1.a1c9680000000p-3, -0x1.94ef9c0000000p-5, -0x1.5c97ca0000000p-5, -0x1.1e5e400000000p-4, -0x1.7269fa0000000p-14, -0x1.11a1620000000p-3, 0x1.5643be0000000p-3, 0x1.6e63760000000p-3, -0x1.7e51620000000p-3, 0x1.b0d2a00000000p-6, -0x1.b6a3200000000p-5, -0x1.b9f9a40000000p-4, 0x1.a565180000000p-4, 0x1.cc8d3c0000000p-3, -0x1.0943280000000p-3, -0x1.a264720000000p-4, -0x1.c56bc40000000p-3}
, {0x1.1915de0000000p-4, 0x1.dcb00e0000000p-6, 0x1.824a240000000p-3, -0x1.7420ea0000000p-4, -0x1.751d920000000p-4, 0x1.a400d20000000p-5, 0x1.723bca0000000p-4, 0x1.4a69e40000000p-3, 0x1.fba1040000000p-5, -0x1.1a05940000000p-5, -0x1.4702e40000000p-4, 0x1.a0f0640000000p-6, 0x1.cd9b7c0000000p-5, -0x1.309b940000000p-4, -0x1.0858ec0000000p-5, -0x1.7fca560000000p-4, -0x1.42b6680000000p-3, -0x1.046d020000000p-5, 0x1.6a0fd60000000p-4, 0x1.eb2efe0000000p-6, -0x1.3891780000000p-4, 0x1.45b3660000000p-3, -0x1.0195920000000p-3, 0x1.469c220000000p-6, -0x1.ba1bce0000000p-5, 0x1.93e7740000000p-5, 0x1.34b42c0000000p-6, 0x1.6de7960000000p-4, -0x1.c4aa3e0000000p-5, 0x1.9f435a0000000p-3, -0x1.059d560000000p-5, 0x1.0292760000000p-5}
, {-0x1.2c72c00000000p-2, 0x1.37f4180000000p-3, 0x1.c07c020000000p-4, 0x1.c684420000000p-3, 0x1.0b936c0000000p-3, -0x1.1ecefa0000000p-4, 0x1.49a7460000000p-7, -0x1.934adc0000000p-3, 0x1.464f380000000p-4, -0x1.5114f20000000p-3, -0x1.9385ce0000000p-6, -0x1.64d6e00000000p-3, 0x1.548afe0000000p-4, 0x1.e8b24c0000000p-4, 0x1.b9cde60000000p-4, 0x1.51f91e0000000p-5, -0x1.63dddc0000000p-3, -0x1.ca55540000000p-3, 0x1.df825e0000000p-3, 0x1.8b8f920000000p-4, 0x1.4e42540000000p-2, 0x1.dcfc080000000p-5, -0x1.3411380000000p-3, 0x1.bef4b40000000p-3, 0x1.9fb6700000000p-3, 0x1.b8d7260000000p-3, -0x1.14f4400000000p-4, -0x1.fd29ce0000000p-3, -0x1.4d54140000000p-6, 0x1.0aa19c0000000p-4, -0x1.bb06120000000p-5, -0x1.3718500000000p-3}
}
, {{0x1.22d69a0000000p-3, -0x1.4089020000000p-3, -0x1.d959c60000000p-8, 0x1.4b0b8a0000000p-3, 0x1.d49b100000000p-3, 0x1.9502340000000p-3, 0x1.d8880a0000000p-4, -0x1.7989420000000p-4, 0x1.e9ee280000000p-4, 0x1.c97a420000000p-4, -0x1.ace39c0000000p-7, 0x1.afc4a40000000p-4, -0x1.6009c80000000p-4, 0x1.1a0bb40000000p-6, -0x1.a4693c0000000p-5, -0x1.8cdb7e0000000p-8, -0x1.b0610c0000000p-5, 0x1.e2ba460000000p-4, 0x1.42a2da0000000p-3, -0x1.5845400000000p-3, 0x1.c95ed00000000p-5, -0x1.c711220000000p-3, 0x1.c14ab00000000p-4, 0x1.3e6e900000000p-4, -0x1.8b80820000000p-5, 0x1.6808c40000000p-7, 0x1.8291780000000p-4, -0x1.267e680000000p-4, 0x1.1a8a560000000p-2, -0x1.0970da0000000p-3, 0x1.7e4dd80000000p-9, 0x1.7d43e80000000p-4}
, {-0x1.af5e2a0000000p-5, 0x1.fbe46a0000000p-4, 0x1.ca0fd40000000p-5, 0x1.fe86860000000p-4, 0x1.06f6660000000p-5, -0x1.6e67520000000p-3, 0x1.f96a600000000p-5, 0x1.9700280000000p-3, -0x1.234a680000000p-5, -0x1.752b540000000p-4, 0x1.aa681c0000000p-4, 0x1.04d9520000000p-4, -0x1.eb38e20000000p-9, -0x1.293f360000000p-4, -0x1.e6b1140000000p-8, -0x1.ec5fb20000000p-6, 0x1.1ed0b20000000p-4, 0x1.3aafec0000000p-3, -0x1.4ded720000000p-8, -0x1.4883900000000p-6, -0x1.7dedea0000000p-5, -0x1.b5bc000000000p-7, -0x1.a5cee80000000p-10, -0x1.5985620000000p-4, 0x1.fb95760000000p-5, -0x1.55fea40000000p-4, 0x1.cbdcf60000000p-4, 0x1.279fe60000000p-4, 0x1.3e410e0000000p-4, 0x1.5200d40000000p-3, 0x1.45a60c0000000p-6, -0x1.9d5fe40000000p-4}
, {0x1.5ab5e20000000p-3, 0x1.ab898c0000000p-7, 0x1.2a9bf80000000p-6, 0x1.a6e71a0000000p-3, -0x1.d5f7180000000p-6, -0x1.a4f1900000000p-6, 0x1.797f100000000p-4, 0x1.5a7f640000000p-4, -0x1.e7c6580000000p-6, -0x1.58818a0000000p-4, 0x1.c159660000000p-3, 0x1.5f46340000000p-5, 0x1.f90fa80000000p-5, -0x1.9885980000000p-4, -0x1.e700900000000p-4, 0x1.f49e240000000p-5, 0x1.0a512c0000000p-5, 0x1.9eed280000000p-4, 0x1.5532220000000p-6, 0x1.128b960000000p-6, -0x1.37b4440000000p-3, 0x1.7feffc0000000p-5, -0x1.3cd57c0000000p-4, -0x1.7969d40000000p-4, -0x1.7edd4c0000000p-5, -0x1.98783c0000000p-5, 0x1.491a940000000p-3, 0x1.33224e0000000p-2, -0x1.3c509c0000000p-4, 0x1.059efe0000000p-3, -0x1.a25cfc0000000p-3, -0x1.2298e60000000p-3}
}
, {{0x1.4638420000000p-4, -0x1.aa08900000000p-4, 0x1.4ac42c0000000p-4, 0x1.ad64b00000000p-5, 0x1.c7dad00000000p-4, -0x1.d176b40000000p-6, -0x1.eee8f80000000p-4, -0x1.ee64000000000p-5, -0x1.5414c80000000p-3, 0x1.588bf40000000p-2, -0x1.a6a50c0000000p-3, 0x1.d788f80000000p-6, -0x1.56a2380000000p-3, -0x1.15edb20000000p-9, -0x1.0031c80000000p-5, -0x1.20678a0000000p-3, 0x1.c805000000000p-4, -0x1.8b789e0000000p-4, 0x1.47816a0000000p-8, -0x1.bc2bd60000000p-3, -0x1.19db100000000p-7, -0x1.3e7e000000000p-2, -0x1.95f0ba0000000p-5, 0x1.3a3b680000000p-5, -0x1.89d5d40000000p-2, -0x1.a0bfda0000000p-5, -0x1.0751d00000000p-6, -0x1.d6da580000000p-4, -0x1.779d1c0000000p-5, 0x1.b7db820000000p-4, -0x1.807aee0000000p-2, -0x1.5f03520000000p-3}
, {0x1.c518ba0000000p-5, -0x1.1258c80000000p-4, 0x1.a881240000000p-5, 0x1.2790280000000p-6, 0x1.6bc8780000000p-4, -0x1.e0994c0000000p-5, -0x1.65fdf20000000p-3, -0x1.0b1a6a0000000p-4, -0x1.122dea0000000p-3, 0x1.6381b60000000p-3, -0x1.e9932a0000000p-3, 0x1.fe3f560000000p-5, 0x1.af9b7c0000000p-4, 0x1.4abc660000000p-4, 0x1.b621200000000p-6, 0x1.22cc460000000p-3, -0x1.c4c48c0000000p-5, -0x1.5272fe0000000p-3, 0x1.5351320000000p-3, -0x1.ae50100000000p-5, 0x1.55cf420000000p-3, -0x1.40596c0000000p-5, 0x1.2747880000000p-3, -0x1.9463a80000000p-7, -0x1.7765e80000000p-3, 0x1.d910540000000p-5, -0x1.7f96020000000p-9, -0x1.3937400000000p-2, 0x1.a1ae560000000p-5, 0x1.0ce5100000000p-4, -0x1.2290720000000p-2, -0x1.493d9e0000000p-3}
, {-0x1.fc1c6c0000000p-3, 0x1.067a920000000p-4, -0x1.1061f80000000p-3, -0x1.14c9780000000p-2, 0x1.7e3d8c0000000p-7, 0x1.ca5f200000000p-7, -0x1.0b12ce0000000p-2, -0x1.93b19c0000000p-3, -0x1.c089060000000p-4, 0x1.d866be0000000p-5, -0x1.1778ec0000000p-4, -0x1.72be2c0000000p-3, -0x1.58278a0000000p-6, 0x1.8eb7e20000000p-4, 0x1.a4340e0000000p-4, 0x1.b17a3e0000000p-4, -0x1.93bec40000000p-3, -0x1.b650020000000p-3, 0x1.94c06a0000000p-4, -0x1.017e2a0000000p-3, -0x1.6e7f340000000p-4, 0x1.e3a6940000000p-6, 0x1.9399fa0000000p-3, 0x1.a9b9c60000000p-6, 0x1.1374520000000p-2, -0x1.4efe540000000p-5, -0x1.1929b20000000p-1, -0x1.5210a20000000p-3, -0x1.93a5700000000p-4, 0x1.73a28a0000000p-7, 0x1.bf0b0a0000000p-6, -0x1.0795060000000p-2}
}
, {{-0x1.b2b6a80000000p-5, -0x1.6063000000000p-2, 0x1.993cda0000000p-5, -0x1.34741c0000000p-4, -0x1.52b5360000000p-8, 0x1.1508660000000p-3, -0x1.8912de0000000p-3, -0x1.259a9a0000000p-3, 0x1.b339c40000000p-3, -0x1.513edc0000000p-3, 0x1.76f8300000000p-12, 0x1.27504a0000000p-2, -0x1.7ab4ce0000000p-3, -0x1.3ab4760000000p-4, -0x1.30106e0000000p-7, 0x1.de2ed20000000p-4, -0x1.00f5060000000p-4, 0x1.46afb20000000p-3, 0x1.68f2100000000p-5, -0x1.c41fc20000000p-2, 0x1.91f8160000000p-2, -0x1.b45b940000000p-4, -0x1.1176da0000000p-4, -0x1.9a6c080000000p-5, 0x1.8c4cae0000000p-5, 0x1.c335900000000p-5, 0x1.885fc40000000p-5, 0x1.342c2a0000000p-3, 0x1.1934300000000p-2, -0x1.bb5a0a0000000p-2, 0x1.bbb4a80000000p-3, 0x1.05162a0000000p-3}
, {-0x1.e5f8a00000000p-3, 0x1.89be340000000p-4, -0x1.a2b3840000000p-4, -0x1.6703200000000p-5, -0x1.5bfc5a0000000p-4, -0x1.f10b680000000p-4, -0x1.0fef680000000p-3, 0x1.81129c0000000p-4, -0x1.e6bef40000000p-7, -0x1.3ff2940000000p-2, 0x1.a3cd420000000p-7, 0x1.6be2aa0000000p-3, -0x1.fad0ec0000000p-6, -0x1.e3e3360000000p-6, 0x1.9104ea0000000p-4, -0x1.dc88780000000p-4, -0x1.b7a6f60000000p-3, -0x1.77b41a0000000p-9, 0x1.e951360000000p-5, 0x1.b750e40000000p-3, -0x1.411f520000000p-8, -0x1.4cc3c40000000p-5, 0x1.d359f00000000p-4, -0x1.e12fd20000000p-7, -0x1.95bcb80000000p-5, -0x1.1aeaa00000000p-4, -0x1.0d7e340000000p-4, -0x1.fc11c60000000p-4, -0x1.3f10d00000000p-3, 0x1.3c3f7e0000000p-5, 0x1.c309f20000000p-10, -0x1.2fbdc40000000p-2}
, {-0x1.71b4980000000p-3, 0x1.2e4c5c0000000p-2, -0x1.86bcfc0000000p-3, 0x1.4a34c00000000p-7, -0x1.efc5b20000000p-5, 0x1.b92e3c0000000p-4, -0x1.6e37e00000000p-3, -0x1.b321b60000000p-6, 0x1.3020740000000p-2, -0x1.bf474a0000000p-4, 0x1.8cd8700000000p-3, -0x1.23f7fc0000000p-4, -0x1.a677fc0000000p-2, 0x1.ba8ece0000000p-3, -0x1.3511900000000p-3, -0x1.03b5de0000000p-6, -0x1.058cf80000000p-2, -0x1.be8be40000000p-4, -0x1.217cb20000000p-3, 0x1.2dafac0000000p-2, 0x1.4a918e0000000p-4, -0x1.611af20000000p-5, -0x1.5dbacc0000000p-4, 0x1.b2a2c80000000p-4, 0x1.1b7b440000000p-2, 0x1.edd0020000000p-4, 0x1.e2a3c00000000p-4, -0x1.cfad300000000p-6, -0x1.9d6bf20000000p-4, 0x1.0327080000000p-3, -0x1.2decb00000000p-9, 0x1.6fff340000000p-3}
}
, {{-0x1.ec5b0c0000000p-5, -0x1.4552cc0000000p-2, 0x1.5fab4e0000000p-2, -0x1.8668fe0000000p-4, -0x1.07f8e00000000p-3, 0x1.dbbea60000000p-4, 0x1.02e5c00000000p-2, -0x1.a31a100000000p-4, -0x1.3d16340000000p-4, 0x1.c625460000000p-3, 0x1.5436280000000p-7, 0x1.c03d7a0000000p-11, 0x1.ed7c3a0000000p-3, -0x1.e3f43c0000000p-4, 0x1.2191fe0000000p-2, -0x1.59885e0000000p-4, 0x1.17b8720000000p-2, 0x1.881be80000000p-5, 0x1.fd0f640000000p-4, 0x1.8a72fe0000000p-4, 0x1.60ab120000000p-2, -0x1.3602480000000p-7, 0x1.af25320000000p-3, -0x1.cddfde0000000p-3, -0x1.31313e0000000p-3, -0x1.fa7e7a0000000p-7, 0x1.2665b80000000p-7, -0x1.4ff05e0000000p-4, -0x1.7555ac0000000p-3, 0x1.07ff720000000p-2, -0x1.77aa5e0000000p-4, -0x1.5ef95e0000000p-8}
, {-0x1.6425aa0000000p-5, -0x1.a982460000000p-5, 0x1.cc4e4c0000000p-3, -0x1.73f1380000000p-4, -0x1.ed048e0000000p-4, -0x1.2b95be0000000p-3, 0x1.e846800000000p-4, 0x1.d5907e0000000p-5, -0x1.50f4b60000000p-4, -0x1.91e8740000000p-5, 0x1.7ffdee0000000p-3, 0x1.0e261c0000000p-4, -0x1.1ffbf40000000p-6, 0x1.3b7ef20000000p-13, 0x1.1a8f8e0000000p-2, 0x1.e7cf2a0000000p-4, 0x1.609b140000000p-4, 0x1.903fbc0000000p-5, 0x1.4ff5440000000p-4, 0x1.2527200000000p-2, 0x1.18ef360000000p-4, 0x1.1c427a0000000p-3, 0x1.fa3aae0000000p-4, 0x1.4730a40000000p-5, -0x1.ced0a80000000p-4, -0x1.8604580000000p-4, -0x1.fc45460000000p-6, 0x1.9998200000000p-5, -0x1.ecd2c20000000p-4, 0x1.b524c40000000p-5, -0x1.e405aa0000000p-4, -0x1.8d7ed80000000p-5}
, {-0x1.289d3c0000000p-4, -0x1.2837d60000000p-3, 0x1.e3d8760000000p-4, -0x1.89c8000000000p-4, 0x1.81b2040000000p-8, -0x1.e5dd160000000p-4, 0x1.8003e60000000p-7, 0x1.c254100000000p-3, -0x1.16a7620000000p-3, 0x1.14bdf80000000p-5, -0x1.9c03860000000p-4, 0x1.5afbc40000000p-3, -0x1.ee380c0000000p-4, 0x1.912ac80000000p-3, 0x1.ad95a40000000p-5, -0x1.ff75d00000000p-4, -0x1.bda2340000000p-3, 0x1.91cf120000000p-4, -0x1.1b9f4e0000000p-3, 0x1.f4dee20000000p-4, -0x1.396a8a0000000p-4, -0x1.b43ad60000000p-4, 0x1.c527200000000p-4, -0x1.4392800000000p-4, 0x1.aa78be0000000p-8, -0x1.62603e0000000p-6, 0x1.9ba9380000000p-4, 0x1.1237c20000000p-3, 0x1.4886720000000p-3, 0x1.20f3e40000000p-6, 0x1.2db7360000000p-4, 0x1.a3ae120000000p-5}
}
, {{0x1.09c7620000000p-4, -0x1.51cfc40000000p-2, -0x1.5441d60000000p-3, 0x1.44b34c0000000p-2, 0x1.9370980000000p-2, -0x1.d789780000000p-3, 0x1.16a4360000000p-5, 0x1.fc0b940000000p-4, 0x1.e18d5a0000000p-3, 0x1.9942c00000000p-4, -0x1.32ae6a0000000p-5, 0x1.38add20000000p-3, -0x1.2920400000000p-6, -0x1.0348a40000000p-3, 0x1.7b453e0000000p-12, 0x1.90bb4e0000000p-3, -0x1.373f860000000p-6, 0x1.874c100000000p-8, 0x1.e305e60000000p-3, 0x1.460c1a0000000p-4, 0x1.3599080000000p-4, 0x1.21d65c0000000p-2, 0x1.282ada0000000p-3, -0x1.31ef500000000p-2, 0x1.fffbc60000000p-3, -0x1.22e6820000000p-3, 0x1.22731c0000000p-4, 0x1.eaf6120000000p-5, 0x1.6363220000000p-4, -0x1.5589020000000p-2, -0x1.2b4d3a0000000p-3, 0x1.d89f8c0000000p-4}
, {-0x1.4b93400000000p-2, -0x1.f795d40000000p-5, -0x1.0ab19a0000000p-6, -0x1.4e0bb20000000p-4, -0x1.db1bf80000000p-3, 0x1.29b7420000000p-3, -0x1.aca2620000000p-6, -0x1.91f3240000000p-3, 0x1.dba65e0000000p-4, -0x1.0b418a0000000p-3, -0x1.13a0cc0000000p-4, -0x1.90cbe80000000p-3, -0x1.e47ffa0000000p-4, -0x1.d6ff100000000p-4, -0x1.e615dc0000000p-3, -0x1.7f6cd60000000p-2, -0x1.3cc66c0000000p-2, -0x1.5fcb560000000p-6, 0x1.3ed8700000000p-7, 0x1.aae3240000000p-5, -0x1.121f480000000p-2, -0x1.e9f34a0000000p-7, -0x1.4c79ec0000000p-3, 0x1.9795620000000p-4, 0x1.1619400000000p-3, -0x1.d7d0360000000p-6, -0x1.67ffcc0000000p-2, -0x1.0d93640000000p-3, 0x1.66fb460000000p-4, -0x1.abe9220000000p-3, -0x1.91b9600000000p-4, -0x1.24d1da0000000p-3}
, {-0x1.097b820000000p-3, 0x1.93ca9c0000000p-5, -0x1.0b2d020000000p-6, -0x1.ecfb440000000p-4, -0x1.68999a0000000p-2, 0x1.f9e3ea0000000p-5, -0x1.6e19140000000p-3, -0x1.cd1b080000000p-5, -0x1.5e8a400000000p-4, -0x1.c793e00000000p-4, -0x1.c594e40000000p-4, -0x1.9f8fe40000000p-4, -0x1.6f012e0000000p-4, 0x1.439a0a0000000p-5, -0x1.4d51de0000000p-4, -0x1.d61f2a0000000p-3, -0x1.1916d40000000p-4, -0x1.616a820000000p-8, -0x1.8dcd3c0000000p-4, 0x1.3647640000000p-3, 0x1.26626e0000000p-6, -0x1.1c58320000000p-5, -0x1.c9ff860000000p-3, -0x1.62af020000000p-5, -0x1.185e020000000p-4, 0x1.7b1fb00000000p-3, -0x1.00f9be0000000p-3, -0x1.d61ca60000000p-3, 0x1.58a4560000000p-4, 0x1.971e980000000p-9, 0x1.5afc8c0000000p-6, -0x1.5709980000000p-5}
}
, {{-0x1.41b1ac0000000p-4, -0x1.50a1b80000000p-3, -0x1.beca900000000p-5, 0x1.2204920000000p-4, -0x1.7ef63c0000000p-3, -0x1.777dc20000000p-5, 0x1.221db60000000p-5, -0x1.519c020000000p-2, 0x1.5cf10e0000000p-5, 0x1.08df180000000p-8, 0x1.3e28700000000p-5, -0x1.d9bd1e0000000p-3, 0x1.ceeb8a0000000p-3, -0x1.1a98180000000p-2, 0x1.91a10c0000000p-4, 0x1.3309380000000p-7, 0x1.07a85a0000000p-5, -0x1.4eb5280000000p-2, -0x1.8331c60000000p-5, -0x1.072fde0000000p-4, 0x1.8408480000000p-3, 0x1.21dd6e0000000p-2, -0x1.abd9500000000p-3, -0x1.22fbbe0000000p-3, -0x1.6fc5920000000p-4, 0x1.c2bd500000000p-3, -0x1.7439080000000p-3, -0x1.9b19440000000p-3, -0x1.ca47da0000000p-3, -0x1.f053d00000000p-4, 0x1.1b738e0000000p-5, 0x1.8340900000000p-5}
, {-0x1.e5e71e0000000p-7, 0x1.055b4c0000000p-10, 0x1.e0067a0000000p-4, 0x1.e2bd120000000p-5, -0x1.c5dbbc0000000p-5, -0x1.14e36c0000000p-4, -0x1.a70b360000000p-7, 0x1.6b55700000000p-6, 0x1.d7f9bc0000000p-7, 0x1.fb13be0000000p-5, 0x1.1711c00000000p-3, 0x1.ec3c8c0000000p-3, 0x1.c28aba0000000p-4, -0x1.2c8ac60000000p-4, -0x1.5b503e0000000p-4, -0x1.26d94a0000000p-4, -0x1.58583e0000000p-9, 0x1.b645340000000p-6, -0x1.78d5f20000000p-4, -0x1.c66ef80000000p-6, 0x1.d909700000000p-7, -0x1.289bbe0000000p-4, -0x1.bdf0de0000000p-3, 0x1.8cac220000000p-4, -0x1.c6504e0000000p-3, -0x1.021d1a0000000p-4, -0x1.24cb940000000p-5, -0x1.3496e20000000p-5, -0x1.4201f80000000p-4, 0x1.92fe600000000p-5, -0x1.7f8d560000000p-3, 0x1.8a78ac0000000p-4}
, {0x1.1c708e0000000p-1, -0x1.e8df620000000p-4, -0x1.5d40120000000p-3, -0x1.cedb400000000p-5, -0x1.8206b40000000p-10, -0x1.8608400000000p-5, 0x1.998f100000000p-9, 0x1.b4ea080000000p-5, 0x1.411d520000000p-3, 0x1.8649cc0000000p-4, -0x1.0d178e0000000p-2, 0x1.6ab43e0000000p-3, 0x1.a717b20000000p-3, 0x1.d34dee0000000p-5, -0x1.7588680000000p-3, 0x1.1463a60000000p-4, -0x1.9280d40000000p-3, 0x1.72d9460000000p-5, -0x1.b536be0000000p-4, 0x1.6b4ea20000000p-5, -0x1.5cd92e0000000p-4, -0x1.35a1b80000000p-10, 0x1.d8025a0000000p-4, -0x1.6aa1e60000000p-3, -0x1.4329c60000000p-5, -0x1.423ade0000000p-4, 0x1.4590400000000p-3, 0x1.8673fa0000000p-3, 0x1.23cfe80000000p-2, -0x1.1d2ea00000000p-3, 0x1.41d1500000000p-3, 0x1.4b82b00000000p-3}
}
, {{0x1.bfe3120000000p-4, 0x1.7df6a00000000p-5, -0x1.ec38460000000p-8, 0x1.2960f80000000p-4, 0x1.5eaf280000000p-6, -0x1.7cea180000000p-8, -0x1.f0aa720000000p-5, 0x1.fd02840000000p-4, 0x1.a7e4860000000p-4, -0x1.dfa52a0000000p-6, -0x1.2f2a200000000p-7, 0x1.03f3d60000000p-4, 0x1.e2208a0000000p-4, 0x1.4b0ef40000000p-4, 0x1.6283240000000p-5, -0x1.6c15b60000000p-4, -0x1.f394bc0000000p-4, -0x1.72726a0000000p-4, -0x1.d2b4bc0000000p-5, 0x1.3131760000000p-3, -0x1.a5abb00000000p-3, 0x1.c137260000000p-4, 0x1.8999360000000p-3, 0x1.a672240000000p-3, -0x1.684d1e0000000p-4, -0x1.06e8380000000p-6, 0x1.b6b05c0000000p-5, -0x1.2e5c680000000p-10, -0x1.d5051a0000000p-7, 0x1.ac6ffa0000000p-5, 0x1.11e55e0000000p-3, 0x1.d613b60000000p-4}
, {-0x1.95d7080000000p-4, 0x1.fcabf40000000p-5, -0x1.47b3960000000p-3, -0x1.7367d00000000p-3, -0x1.dc9ac40000000p-5, -0x1.515a240000000p-5, -0x1.30895c0000000p-4, 0x1.5800f00000000p-5, 0x1.fd1c060000000p-4, -0x1.11fc500000000p-2, 0x1.2dc59c0000000p-3, -0x1.6e7fc40000000p-4, -0x1.6ebcb80000000p-2, -0x1.94e53a0000000p-3, 0x1.b434a40000000p-10, -0x1.e522b60000000p-9, 0x1.389ba20000000p-4, -0x1.7cb15c0000000p-6, -0x1.6d5a240000000p-5, -0x1.3567a20000000p-4, -0x1.d4360c0000000p-3, -0x1.fe89f20000000p-6, 0x1.240d900000000p-3, 0x1.09a28c0000000p-3, 0x1.f20f9e0000000p-4, -0x1.cc84820000000p-4, -0x1.6e16f40000000p-3, -0x1.ba8a120000000p-6, -0x1.29bd160000000p-5, 0x1.3f93460000000p-3, -0x1.001a140000000p-4, -0x1.7ef9a00000000p-6}
, {0x1.2852f40000000p-6, 0x1.a1ffde0000000p-3, 0x1.03e0fa0000000p-9, -0x1.f8b1540000000p-4, -0x1.0bd1520000000p-2, -0x1.0615f00000000p-2, 0x1.ee11fa0000000p-5, -0x1.82b4440000000p-2, 0x1.1dcf8a0000000p-3, -0x1.c466980000000p-3, 0x1.5b65ae0000000p-4, -0x1.65f7660000000p-4, -0x1.c244400000000p-3, -0x1.40ad400000000p-2, -0x1.48be960000000p-2, -0x1.5d0a240000000p-2, 0x1.25d1b80000000p-3, -0x1.be65200000000p-10, -0x1.6870e00000000p-5, -0x1.3b32780000000p-3, -0x1.5e69280000000p-3, 0x1.3c4a5e0000000p-5, 0x1.0986740000000p-6, 0x1.0ad3fc0000000p-3, -0x1.0efb880000000p-4, -0x1.1340220000000p-5, -0x1.e27df40000000p-8, -0x1.0dd7b60000000p-8, -0x1.01a3d80000000p-4, 0x1.1192300000000p-5, -0x1.e43bde0000000p-9, -0x1.dc4ce80000000p-3}
}
, {{0x1.84f3a40000000p-3, 0x1.af3d320000000p-4, -0x1.1ea7fa0000000p-3, -0x1.95022a0000000p-3, -0x1.86a5960000000p-7, 0x1.325e180000000p-3, -0x1.eeee080000000p-3, 0x1.0428480000000p-3, 0x1.18295a0000000p-2, -0x1.a6c4700000000p-4, 0x1.3c24520000000p-6, 0x1.e3a8f20000000p-4, 0x1.27477e0000000p-2, 0x1.f12fc20000000p-3, -0x1.c631980000000p-4, -0x1.7f78220000000p-5, -0x1.3738ae0000000p-6, -0x1.3806380000000p-7, -0x1.1e5cf60000000p-2, 0x1.8ca6940000000p-2, -0x1.b6f2a00000000p-4, -0x1.0d3e0e0000000p-2, -0x1.8bbd6e0000000p-5, -0x1.b1290a0000000p-5, -0x1.ec28f80000000p-4, 0x1.b804c60000000p-3, -0x1.43e4060000000p-3, -0x1.258e1a0000000p-4, -0x1.75a5be0000000p-3, -0x1.285fa40000000p-4, 0x1.cde8fe0000000p-8, 0x1.f6b2ce0000000p-4}
, {0x1.77cdc40000000p-3, 0x1.f6914e0000000p-6, 0x1.0ab8a20000000p-2, -0x1.9fed040000000p-5, -0x1.5905820000000p-3, 0x1.b798ca0000000p-3, -0x1.4931c40000000p-2, -0x1.b2d7420000000p-4, -0x1.9907ce0000000p-3, -0x1.fdf7820000000p-4, -0x1.004d120000000p-2, 0x1.4d71bc0000000p-4, -0x1.90cb8e0000000p-4, -0x1.0f41040000000p-2, -0x1.87a71e0000000p-4, 0x1.724a180000000p-4, -0x1.a82b1e0000000p-3, 0x1.05d72a0000000p-2, 0x1.91c3580000000p-4, -0x1.33981a0000000p-2, 0x1.4d35200000000p-3, -0x1.d575d00000000p-3, 0x1.de15f00000000p-7, 0x1.a6f5e00000000p-6, -0x1.235c520000000p-5, -0x1.4e507a0000000p-4, -0x1.7776e80000000p-4, -0x1.2f6b240000000p-3, 0x1.a6543e0000000p-3, 0x1.79b3d20000000p-6, 0x1.2072a00000000p-3, -0x1.a733da0000000p-5}
, {-0x1.6cbb480000000p-4, -0x1.c5edce0000000p-5, 0x1.3a268c0000000p-5, -0x1.15405e0000000p-3, -0x1.11a3160000000p-3, 0x1.9ad0e60000000p-4, -0x1.11dccc0000000p-4, -0x1.8c65ae0000000p-3, 0x1.07550a0000000p-4, -0x1.9279b80000000p-2, 0x1.af815e0000000p-7, 0x1.663bfe0000000p-3, -0x1.6de8520000000p-7, 0x1.124cfc0000000p-2, -0x1.a0672c0000000p-6, 0x1.dfe0660000000p-4, -0x1.85db200000000p-2, -0x1.ce58c80000000p-4, -0x1.fd64f40000000p-6, -0x1.6466140000000p-3, 0x1.5e0c500000000p-3, 0x1.faea660000000p-3, 0x1.734f600000000p-3, 0x1.0f8ed60000000p-5, 0x1.7c591a0000000p-4, 0x1.0573180000000p-3, -0x1.4cbe9a0000000p-3, -0x1.b619a00000000p-3, 0x1.29f4580000000p-3, 0x1.a77b460000000p-5, 0x1.3a23c80000000p-3, 0x1.2a7baa0000000p-4}
}
, {{0x1.32092c0000000p-6, 0x1.7ee3320000000p-3, -0x1.be69380000000p-4, -0x1.fc05700000000p-4, -0x1.d850d80000000p-4, -0x1.8853f60000000p-4, -0x1.add4600000000p-3, -0x1.7bae3c0000000p-4, 0x1.342cbe0000000p-3, 0x1.44034e0000000p-6, -0x1.67822c0000000p-4, 0x1.14eb520000000p-7, 0x1.b7d3aa0000000p-3, 0x1.62a4080000000p-2, -0x1.ea2e320000000p-4, -0x1.46bf940000000p-3, -0x1.ced0c60000000p-5, -0x1.5e687c0000000p-5, -0x1.7d1dd40000000p-3, 0x1.7fb4e00000000p-5, -0x1.76bd7e0000000p-3, 0x1.51f89c0000000p-10, -0x1.6f84140000000p-2, -0x1.f8ae060000000p-9, -0x1.b777c20000000p-6, 0x1.0894de0000000p-3, -0x1.ee471a0000000p-4, 0x1.ebae420000000p-8, -0x1.c2e3ee0000000p-3, -0x1.0f21ce0000000p-2, -0x1.9309e80000000p-9, 0x1.0860a20000000p-4}
, {0x1.f62cd20000000p-4, 0x1.07b9260000000p-3, -0x1.3cac920000000p-5, 0x1.3cc4100000000p-6, 0x1.6859e20000000p-4, -0x1.31d3240000000p-3, 0x1.a617bc0000000p-5, -0x1.2ef5dc0000000p-4, 0x1.2f45de0000000p-3, -0x1.3598c00000000p-9, 0x1.baadae0000000p-4, 0x1.e1d2740000000p-4, 0x1.6e728a0000000p-5, -0x1.0b32460000000p-6, -0x1.4468900000000p-5, 0x1.3e9cea0000000p-3, -0x1.d7001e0000000p-4, -0x1.81e05a0000000p-4, -0x1.0fd2940000000p-3, 0x1.c1e7460000000p-3, 0x1.b8d0620000000p-5, 0x1.6f63de0000000p-5, 0x1.1997e40000000p-8, -0x1.e58cea0000000p-3, 0x1.9caeba0000000p-4, -0x1.470d640000000p-3, -0x1.8545c80000000p-5, 0x1.62af480000000p-4, 0x1.965f4a0000000p-6, -0x1.fcb28e0000000p-3, 0x1.bf26ea0000000p-3, 0x1.4be30a0000000p-6}
, {0x1.f6dec80000000p-3, -0x1.4a5e800000000p-3, -0x1.e640460000000p-5, 0x1.341b340000000p-3, -0x1.13d7f60000000p-5, 0x1.5e1fd60000000p-4, -0x1.000fa00000000p-3, 0x1.6bf8940000000p-6, -0x1.be85d20000000p-5, 0x1.7dacd20000000p-4, -0x1.cde6a80000000p-4, -0x1.8ed5320000000p-11, 0x1.02e70e0000000p-2, 0x1.ac902c0000000p-3, 0x1.41d9b20000000p-3, 0x1.309eda0000000p-2, -0x1.18ca480000000p-2, 0x1.a8092c0000000p-6, -0x1.a2cac40000000p-6, -0x1.8e31a80000000p-5, 0x1.bca0e80000000p-5, 0x1.8d00840000000p-5, -0x1.b41d3e0000000p-4, -0x1.597fd60000000p-6, -0x1.0771e80000000p-10, 0x1.70867a0000000p-4, -0x1.7f722c0000000p-3, -0x1.dd85340000000p-4, -0x1.5485ca0000000p-4, 0x1.a2c1ee0000000p-5, -0x1.ebb5500000000p-6, -0x1.c5a0920000000p-6}
}
, {{-0x1.ba1f960000000p-2, 0x1.7ea6560000000p-3, -0x1.e8b0a00000000p-3, 0x1.5c397e0000000p-6, -0x1.c4d5f80000000p-6, 0x1.3ee3640000000p-2, -0x1.3a565c0000000p-3, -0x1.e7b5f40000000p-3, 0x1.ba78540000000p-7, -0x1.2873c80000000p-3, -0x1.6e1f780000000p-4, -0x1.6422800000000p-2, -0x1.4d14700000000p-3, 0x1.3672480000000p-3, -0x1.44bc360000000p-5, -0x1.56f1b20000000p-5, 0x1.15f3ec0000000p-4, -0x1.7abd160000000p-4, 0x1.ccea5c0000000p-4, -0x1.75ffae0000000p-5, 0x1.e91fb20000000p-4, -0x1.b4b9bc0000000p-11, -0x1.dc2e220000000p-4, -0x1.5505180000000p-5, -0x1.5c6c880000000p-4, 0x1.abad5a0000000p-2, -0x1.cbe2880000000p-7, -0x1.01c5ce0000000p-2, 0x1.26248e0000000p-5, -0x1.42697a0000000p-5, 0x1.fbd0860000000p-4, 0x1.d94e5a0000000p-6}
, {-0x1.416e980000000p-4, -0x1.91b0700000000p-4, -0x1.4f0c1e0000000p-5, 0x1.8b29da0000000p-4, 0x1.447d840000000p-3, -0x1.5736740000000p-4, 0x1.b347620000000p-6, 0x1.4cdaac0000000p-3, -0x1.0e50f80000000p-3, -0x1.6067b20000000p-9, -0x1.5e42da0000000p-3, 0x1.1694a60000000p-3, 0x1.879a240000000p-4, -0x1.ca1a400000000p-7, -0x1.00f8200000000p-3, -0x1.b61a220000000p-3, 0x1.8f19200000000p-5, 0x1.3f33520000000p-4, 0x1.9cd45e0000000p-4, -0x1.4dbb520000000p-3, 0x1.71583c0000000p-6, -0x1.07b7a80000000p-3, 0x1.d571860000000p-6, -0x1.33d2940000000p-6, -0x1.26b8800000000p-3, 0x1.060cf60000000p-5, 0x1.c43a940000000p-8, -0x1.ab6ada0000000p-3, 0x1.26347a0000000p-3, -0x1.3b11380000000p-6, 0x1.22c2f40000000p-4, 0x1.3702880000000p-4}
, {0x1.26c8300000000p-2, -0x1.2d05cc0000000p-5, 0x1.220fa40000000p-3, 0x1.81ea840000000p-8, -0x1.cf14b20000000p-9, 0x1.b0acd60000000p-4, -0x1.0072900000000p-3, 0x1.341e4a0000000p-3, 0x1.9304ac0000000p-7, 0x1.f265c00000000p-5, -0x1.2328840000000p-3, 0x1.a5eb5a0000000p-6, 0x1.345b680000000p-5, -0x1.b0b4720000000p-3, -0x1.3ff31a0000000p-3, 0x1.8b50d40000000p-4, 0x1.a19d840000000p-3, -0x1.dcc9ec0000000p-5, -0x1.87865c0000000p-6, -0x1.075d4c0000000p-5, -0x1.3b30760000000p-4, 0x1.be820c0000000p-8, 0x1.2bbd8c0000000p-6, -0x1.44810e0000000p-6, -0x1.ab5bf40000000p-4, -0x1.8226120000000p-4, 0x1.7411240000000p-5, 0x1.d57cd80000000p-4, -0x1.4be4dc0000000p-10, -0x1.2218e80000000p-3, -0x1.d198a40000000p-3, 0x1.f6131c0000000p-4}
}
, {{-0x1.0a87e40000000p-2, -0x1.155c000000000p-5, -0x1.3fcfea0000000p-3, -0x1.5564880000000p-6, 0x1.3fc6c40000000p-7, 0x1.5d1a800000000p-3, -0x1.e1e7c60000000p-4, -0x1.8189b80000000p-7, -0x1.059bc40000000p-3, 0x1.c8aa120000000p-4, -0x1.13edaa0000000p-3, -0x1.2b30ac0000000p-6, 0x1.4dc6fa0000000p-7, -0x1.17480c0000000p-8, 0x1.3daa420000000p-8, -0x1.b851980000000p-4, 0x1.c431800000000p-6, -0x1.4f445c0000000p-5, 0x1.7f25980000000p-8, -0x1.74f0200000000p-6, 0x1.b49df60000000p-3, -0x1.42fd4e0000000p-3, 0x1.182e1a0000000p-6, 0x1.be1afc0000000p-5, -0x1.dfb9240000000p-3, 0x1.389bda0000000p-4, 0x1.70316e0000000p-6, -0x1.a540e20000000p-6, -0x1.adfa860000000p-5, -0x1.4c22960000000p-4, 0x1.00e5400000000p-3, 0x1.b939c80000000p-4}
, {0x1.b78e7a0000000p-5, -0x1.3061200000000p-4, -0x1.57bc700000000p-6, 0x1.2415320000000p-3, -0x1.1757880000000p-10, -0x1.8c32b20000000p-5, -0x1.3abde40000000p-5, 0x1.d3e0160000000p-4, -0x1.7312ec0000000p-6, 0x1.01027e0000000p-4, 0x1.cb7c080000000p-4, 0x1.8388da0000000p-5, -0x1.26df840000000p-2, 0x1.5a467e0000000p-3, 0x1.a592380000000p-11, -0x1.1302080000000p-5, 0x1.f982e60000000p-4, 0x1.5c17c40000000p-4, 0x1.0a80700000000p-4, 0x1.cd0e620000000p-6, 0x1.1b8e940000000p-3, -0x1.5413d60000000p-3, 0x1.fcb4f60000000p-5, -0x1.1c32400000000p-8, 0x1.4104040000000p-4, 0x1.6a1d920000000p-3, -0x1.6e784e0000000p-7, 0x1.c6e99e0000000p-4, 0x1.2d06820000000p-3, -0x1.fbdda40000000p-3, -0x1.5d65160000000p-3, 0x1.68f9120000000p-4}
, {-0x1.adf1d80000000p-2, 0x1.ad8bd20000000p-6, 0x1.7a739c0000000p-3, 0x1.e358200000000p-4, -0x1.f864f20000000p-3, -0x1.425a460000000p-5, -0x1.ef1ce80000000p-3, 0x1.7860ac0000000p-8, -0x1.557df20000000p-3, 0x1.60b7be0000000p-6, -0x1.0c12dc0000000p-5, 0x1.bf2dfe0000000p-6, 0x1.7b5aea0000000p-5, -0x1.910f100000000p-3, 0x1.5d63740000000p-4, 0x1.1e0c440000000p-4, 0x1.fc32720000000p-2, 0x1.befa7a0000000p-4, -0x1.5b41ce0000000p-6, -0x1.7b195c0000000p-4, 0x1.44a0220000000p-3, -0x1.ae60ea0000000p-3, 0x1.7372820000000p-3, -0x1.23be120000000p-3, -0x1.5a05d40000000p-2, -0x1.9f5bf20000000p-3, -0x1.6b512a0000000p-9, 0x1.082b6e0000000p-8, -0x1.a514200000000p-5, 0x1.38ba7c0000000p-2, -0x1.358a740000000p-2, -0x1.3749e80000000p-4}
}
, {{0x1.7d8b2a0000000p-4, -0x1.16624e0000000p-3, -0x1.b5915c0000000p-3, -0x1.d7cae00000000p-5, 0x1.5cf42a0000000p-4, -0x1.2271ec0000000p-3, 0x1.e0ca6a0000000p-15, 0x1.73fbe40000000p-4, -0x1.164b340000000p-2, 0x1.fa2d8c0000000p-5, -0x1.268f620000000p-2, 0x1.ca9dae0000000p-3, -0x1.f28e080000000p-4, 0x1.d5219c0000000p-3, -0x1.66f1780000000p-4, 0x1.bf47ee0000000p-8, 0x1.3412dc0000000p-5, -0x1.1dda660000000p-2, -0x1.2c4a9e0000000p-4, 0x1.cb95c40000000p-4, -0x1.84ec9a0000000p-7, -0x1.f38ea20000000p-9, 0x1.e279240000000p-3, -0x1.20b9b00000000p-4, -0x1.4abc140000000p-5, -0x1.189d400000000p-3, 0x1.a72d3e0000000p-4, 0x1.bce7600000000p-5, -0x1.b4a09a0000000p-6, -0x1.27c8040000000p-3, -0x1.dc956c0000000p-3, 0x1.08c6060000000p-3}
, {0x1.9a2c400000000p-4, -0x1.e483420000000p-4, -0x1.0c63240000000p-3, -0x1.a2c4300000000p-9, 0x1.1ad0800000000p-3, -0x1.50fdfe0000000p-7, -0x1.5f9daa0000000p-5, -0x1.149d180000000p-3, -0x1.7a1dee0000000p-6, -0x1.f890da0000000p-5, -0x1.2200a60000000p-4, -0x1.7426700000000p-3, -0x1.1a34460000000p-4, 0x1.33daee0000000p-2, 0x1.b3f59a0000000p-3, 0x1.fa963c0000000p-3, -0x1.e0bd100000000p-6, -0x1.1f69600000000p-3, 0x1.398b5a0000000p-3, 0x1.5af4420000000p-3, 0x1.33d38e0000000p-5, 0x1.05f7a20000000p-4, 0x1.3709a60000000p-7, 0x1.8207ec0000000p-4, 0x1.46f5420000000p-3, 0x1.7a6cb80000000p-3, 0x1.0aec580000000p-3, -0x1.634daa0000000p-6, -0x1.29b39e0000000p-5, -0x1.3982f80000000p-2, 0x1.9ed20c0000000p-4, 0x1.07d6ac0000000p-3}
, {-0x1.0240a20000000p-3, -0x1.a6eb000000000p-3, -0x1.d55e920000000p-3, 0x1.64acdc0000000p-3, -0x1.5a2f6e0000000p-6, -0x1.b782b40000000p-16, 0x1.b6cf9a0000000p-4, -0x1.fbbf800000000p-6, 0x1.57f1740000000p-3, 0x1.20baec0000000p-2, -0x1.0fc45a0000000p-2, 0x1.48a2fa0000000p-4, -0x1.42f2e20000000p-4, 0x1.65e95e0000000p-3, 0x1.0c78c40000000p-5, -0x1.2f16460000000p-5, 0x1.cec6040000000p-6, 0x1.f7f9260000000p-8, 0x1.47edbc0000000p-4, -0x1.80d6720000000p-6, 0x1.4bc6380000000p-4, -0x1.90f55c0000000p-4, -0x1.d1fe4a0000000p-5, -0x1.39dbae0000000p-3, -0x1.2177ea0000000p-7, 0x1.9656e20000000p-3, 0x1.0878d40000000p-2, 0x1.8e4ff60000000p-4, 0x1.4f42300000000p-3, -0x1.73d12c0000000p-12, 0x1.852ed00000000p-3, 0x1.7a3f9a0000000p-3}
}
, {{-0x1.412b020000000p-3, 0x1.188d440000000p-4, 0x1.1b13160000000p-4, -0x1.f222180000000p-4, -0x1.17e2380000000p-4, -0x1.64b59e0000000p-7, -0x1.3f3bc40000000p-3, -0x1.9b95220000000p-4, -0x1.b91a8c0000000p-4, 0x1.42a27c0000000p-4, -0x1.e57cda0000000p-5, -0x1.522c1e0000000p-5, -0x1.dd5ec40000000p-5, -0x1.0b7ddc0000000p-2, 0x1.8adf700000000p-3, 0x1.e13bac0000000p-5, 0x1.a3b6640000000p-3, -0x1.02011c0000000p-4, -0x1.22e5e60000000p-3, 0x1.15220c0000000p-4, 0x1.a17e660000000p-4, -0x1.e255e20000000p-4, -0x1.f624200000000p-3, 0x1.e7beae0000000p-5, 0x1.f2032c0000000p-9, 0x1.1f56160000000p-5, -0x1.a9c4880000000p-3, -0x1.460e6e0000000p-3, -0x1.0b757a0000000p-2, 0x1.cfc5c60000000p-3, -0x1.07ea7c0000000p-8, 0x1.5f011c0000000p-5}
, {-0x1.522abc0000000p-5, 0x1.525e5e0000000p-7, -0x1.5bc59e0000000p-4, -0x1.1da7840000000p-2, -0x1.a080040000000p-4, -0x1.1a03f00000000p-5, -0x1.d84ff60000000p-4, -0x1.c61bbe0000000p-3, -0x1.831fa60000000p-4, 0x1.24efe20000000p-4, 0x1.64945c0000000p-3, 0x1.ddc0480000000p-4, 0x1.0379a20000000p-2, -0x1.b733dc0000000p-4, -0x1.3de9780000000p-3, -0x1.412f7e0000000p-8, 0x1.969c740000000p-3, 0x1.9498900000000p-4, 0x1.2952a20000000p-4, -0x1.bfb8340000000p-4, 0x1.72c6400000000p-3, -0x1.4c80540000000p-3, 0x1.a2f8fc0000000p-4, 0x1.079a3c0000000p-3, -0x1.b11b380000000p-3, -0x1.615ef80000000p-6, -0x1.3c97fa0000000p-2, -0x1.54d64e0000000p-5, 0x1.b822d80000000p-6, 0x1.0010be0000000p-5, 0x1.0740580000000p-5, -0x1.ab81d00000000p-2}
, {0x1.0358380000000p-1, -0x1.062a840000000p-2, 0x1.5ac1e20000000p-3, -0x1.c169420000000p-4, -0x1.5027960000000p-8, 0x1.dd921c0000000p-3, 0x1.7bbaf40000000p-2, -0x1.051a580000000p-6, -0x1.69bfde0000000p-2, 0x1.a590260000000p-3, -0x1.03065c0000000p-2, 0x1.de30d80000000p-5, 0x1.2181aa0000000p-2, 0x1.16396c0000000p-4, 0x1.c497740000000p-4, 0x1.d47fe00000000p-4, 0x1.6832d80000000p-4, 0x1.166c980000000p-3, -0x1.77656a0000000p-2, -0x1.ce08a20000000p-3, 0x1.0f887a0000000p-3, -0x1.8aba880000000p-8, -0x1.0818b00000000p-4, -0x1.b30a540000000p-2, -0x1.745fa00000000p-2, -0x1.0a732e0000000p-2, -0x1.112b7c0000000p-4, 0x1.65e2f20000000p-3, -0x1.6400e20000000p-3, -0x1.3505f80000000p-5, -0x1.5c33bc0000000p-5, 0x1.0be10e0000000p-3}
}
, {{-0x1.4423300000000p-5, 0x1.d9c4fe0000000p-6, 0x1.2db4740000000p-7, -0x1.1971aa0000000p-3, 0x1.5b93b20000000p-5, -0x1.5c9a8c0000000p-6, -0x1.93783c0000000p-3, -0x1.b961400000000p-3, -0x1.94cf180000000p-5, 0x1.68e13e0000000p-3, 0x1.2c3f380000000p-5, 0x1.6537d00000000p-5, -0x1.bd368e0000000p-5, -0x1.2524040000000p-3, -0x1.50d86a0000000p-6, -0x1.f848920000000p-9, 0x1.3929f40000000p-3, 0x1.ca81200000000p-4, 0x1.9f18a00000000p-3, -0x1.11a6700000000p-3, 0x1.0993300000000p-7, -0x1.33c61e0000000p-2, 0x1.6979f20000000p-4, 0x1.4c3b420000000p-7, 0x1.563dc80000000p-4, -0x1.aabd140000000p-3, 0x1.a83b5e0000000p-3, -0x1.7c39e40000000p-3, 0x1.e0ff840000000p-9, 0x1.66c6960000000p-3, 0x1.27fdcc0000000p-3, -0x1.0fb3380000000p-3}
, {-0x1.c088960000000p-4, -0x1.610d2e0000000p-3, 0x1.63602e0000000p-3, -0x1.7ae9400000000p-4, 0x1.c71f9e0000000p-4, 0x1.2077260000000p-6, -0x1.9bef960000000p-4, 0x1.7dbfb60000000p-5, -0x1.1a4e700000000p-3, 0x1.3cdf220000000p-3, -0x1.75c41e0000000p-4, -0x1.f8d0ec0000000p-5, 0x1.da96800000000p-8, -0x1.16db1e0000000p-4, 0x1.44d3a00000000p-14, -0x1.906f400000000p-5, -0x1.135ca80000000p-3, -0x1.a5d55a0000000p-4, -0x1.c5fd6e0000000p-4, -0x1.6be6180000000p-3, 0x1.f9201a0000000p-4, -0x1.b689ec0000000p-4, 0x1.eea4320000000p-4, 0x1.2002340000000p-4, -0x1.26ca8c0000000p-3, 0x1.581a240000000p-13, -0x1.35aade0000000p-3, 0x1.49e6c60000000p-3, 0x1.04da260000000p-3, 0x1.62be120000000p-4, 0x1.0de4fa0000000p-3, -0x1.8ba4780000000p-6}
, {-0x1.03f75c0000000p-12, 0x1.01498c0000000p-3, 0x1.cb82c00000000p-4, -0x1.32e45e0000000p-4, -0x1.0c6a580000000p-3, 0x1.7af17e0000000p-4, -0x1.b33e5e0000000p-4, -0x1.51f7580000000p-6, -0x1.b01d580000000p-3, -0x1.3af8960000000p-6, 0x1.ef153c0000000p-5, 0x1.0317680000000p-5, -0x1.066dbe0000000p-3, -0x1.5e37720000000p-4, 0x1.3d3ea00000000p-3, 0x1.4286940000000p-7, 0x1.a35e6c0000000p-2, 0x1.3b46140000000p-3, 0x1.7212420000000p-3, 0x1.5fc1b20000000p-5, 0x1.0e2d1c0000000p-2, 0x1.7bda8a0000000p-4, 0x1.f009bc0000000p-5, -0x1.19bca80000000p-7, -0x1.e054fe0000000p-3, 0x1.e468cc0000000p-6, 0x1.b545260000000p-3, -0x1.0a549a0000000p-4, -0x1.6d28400000000p-4, -0x1.1e405e0000000p-2, -0x1.1ef3020000000p-2, 0x1.72d2240000000p-7}
}
, {{0x1.6547480000000p-4, 0x1.20f9720000000p-5, 0x1.1691b40000000p-5, 0x1.e02f3c0000000p-7, 0x1.b17c040000000p-3, 0x1.9357720000000p-4, -0x1.623ffc0000000p-4, -0x1.323f300000000p-4, 0x1.de03240000000p-4, -0x1.e4fd820000000p-3, -0x1.3311220000000p-3, 0x1.70908c0000000p-3, -0x1.80c2020000000p-3, 0x1.0a20ee0000000p-6, -0x1.40913a0000000p-3, 0x1.cc049e0000000p-7, -0x1.27ca8e0000000p-5, 0x1.64f28a0000000p-4, -0x1.57bbc80000000p-5, 0x1.8b4e100000000p-3, 0x1.3f9fc00000000p-4, -0x1.840db00000000p-5, -0x1.a15f6c0000000p-4, -0x1.01af4a0000000p-4, 0x1.7f84b20000000p-5, 0x1.c071200000000p-4, -0x1.0c29b80000000p-3, -0x1.def28e0000000p-6, 0x1.348b060000000p-3, -0x1.1dd5400000000p-3, -0x1.a5a0e60000000p-4, 0x1.99ea9e0000000p-4}
, {0x1.0d5eb00000000p-4, -0x1.c5107c0000000p-4, -0x1.82df460000000p-2, 0x1.a437280000000p-3, -0x1.edb74a0000000p-7, 0x1.aa2f120000000p-5, -0x1.1d09280000000p-2, 0x1.0d66980000000p-6, 0x1.d429b40000000p-4, -0x1.6dc91a0000000p-3, -0x1.add4500000000p-4, 0x1.853f2e0000000p-4, 0x1.0468420000000p-5, 0x1.c1e6840000000p-3, -0x1.1ac1ec0000000p-2, 0x1.80273e0000000p-2, -0x1.e81c440000000p-2, -0x1.7092bc0000000p-3, -0x1.b915e20000000p-3, 0x1.a1076a0000000p-3, 0x1.2291fc0000000p-8, 0x1.4cb2bc0000000p-3, 0x1.1439080000000p-3, -0x1.228f300000000p-3, 0x1.9ae2740000000p-4, 0x1.602f9e0000000p-4, -0x1.5fce800000000p-4, -0x1.081cbc0000000p-5, -0x1.12f5460000000p-4, -0x1.9aba740000000p-2, -0x1.1b8cba0000000p-3, 0x1.e71d800000000p-5}
, {-0x1.057fd00000000p-3, 0x1.6d95620000000p-3, -0x1.feb2ea0000000p-2, 0x1.a8ccd00000000p-4, -0x1.56db1c0000000p-5, -0x1.2bbbfc0000000p-5, -0x1.654f460000000p-3, 0x1.76edd60000000p-3, 0x1.d6c1a80000000p-3, -0x1.65e7da0000000p-5, -0x1.7d437e0000000p-3, 0x1.c2be320000000p-4, -0x1.c388760000000p-3, 0x1.43932c0000000p-2, -0x1.ac29ac0000000p-6, -0x1.15548c0000000p-2, -0x1.fa193c0000000p-3, -0x1.af562e0000000p-3, -0x1.c4a0a60000000p-7, 0x1.6cc6f80000000p-3, 0x1.04a08a0000000p-4, -0x1.8cbcf20000000p-6, -0x1.3f884e0000000p-2, -0x1.46e9520000000p-2, 0x1.a23a800000000p-7, 0x1.1c7bd20000000p-3, 0x1.4faace0000000p-6, -0x1.2f65040000000p-3, 0x1.83ba360000000p-7, -0x1.87d0740000000p-2, -0x1.5fc5e80000000p-4, 0x1.0832a00000000p-3}
}
, {{-0x1.a8be100000000p-2, 0x1.47ee2a0000000p-3, 0x1.8930700000000p-7, 0x1.3777160000000p-4, -0x1.4627ac0000000p-4, -0x1.d1f8080000000p-3, -0x1.1442e20000000p-2, -0x1.0632ee0000000p-2, 0x1.87d4aa0000000p-5, -0x1.aa1f2e0000000p-4, 0x1.5ba99a0000000p-6, 0x1.db0c7c0000000p-5, 0x1.274e1e0000000p-5, 0x1.3d18780000000p-4, -0x1.a666780000000p-4, -0x1.84bf5a0000000p-3, 0x1.2b20660000000p-4, -0x1.c1cfd40000000p-3, 0x1.719f060000000p-3, -0x1.4c6a580000000p-4, -0x1.7651f20000000p-3, -0x1.7082720000000p-3, 0x1.3f13e00000000p-4, 0x1.a417d60000000p-5, -0x1.32d1c80000000p-4, 0x1.2771940000000p-3, -0x1.da75520000000p-3, -0x1.17f09e0000000p-2, -0x1.863b100000000p-4, 0x1.9f79460000000p-2, 0x1.c42a840000000p-5, -0x1.24b49a0000000p-3}
, {0x1.254c0a0000000p-6, 0x1.a746220000000p-3, -0x1.9995e00000000p-3, 0x1.181dd60000000p-3, -0x1.1173100000000p-5, -0x1.99b6c40000000p-6, -0x1.94dfda0000000p-3, -0x1.88034a0000000p-5, -0x1.8cbb6c0000000p-7, -0x1.f5c2020000000p-3, -0x1.486e260000000p-3, -0x1.1becea0000000p-6, 0x1.861b7c0000000p-11, 0x1.83bba60000000p-5, -0x1.c431280000000p-4, -0x1.1296b40000000p-2, -0x1.d33ebe0000000p-4, -0x1.0ef06c0000000p-3, -0x1.bba4640000000p-4, 0x1.2cd7b40000000p-4, -0x1.779c5a0000000p-4, -0x1.6e526a0000000p-3, -0x1.a268100000000p-3, 0x1.a528520000000p-4, 0x1.c159f00000000p-4, 0x1.88584e0000000p-4, 0x1.a8ee520000000p-4, -0x1.ea54260000000p-4, 0x1.a4cf4e0000000p-4, -0x1.9079420000000p-5, 0x1.235b160000000p-4, 0x1.7531ac0000000p-3}
, {0x1.0348ce0000000p-2, -0x1.a9bdce0000000p-4, -0x1.9d5b760000000p-3, 0x1.1533000000000p-5, 0x1.15ce9c0000000p-3, -0x1.942b340000000p-6, -0x1.de1b200000000p-4, 0x1.c637b20000000p-4, 0x1.2aa0980000000p-5, -0x1.d16aaa0000000p-4, -0x1.69b0480000000p-4, 0x1.089f900000000p-4, 0x1.d1bbcc0000000p-3, 0x1.01fe300000000p-2, 0x1.32b80c0000000p-7, 0x1.d535c60000000p-4, -0x1.572f320000000p-3, 0x1.8898260000000p-4, -0x1.fca0d20000000p-3, 0x1.e9b61e0000000p-5, -0x1.428e7c0000000p-6, -0x1.89da5a0000000p-6, 0x1.a8ba0a0000000p-5, -0x1.1462440000000p-2, -0x1.15c9320000000p-5, -0x1.efa9aa0000000p-4, 0x1.87e2a40000000p-3, 0x1.021e220000000p-2, 0x1.5ef1c20000000p-3, -0x1.bede0c0000000p-2, -0x1.6892e00000000p-3, 0x1.12e92e0000000p-2}
}
, {{0x1.8203740000000p-3, -0x1.18e9020000000p-2, 0x1.1138f20000000p-3, -0x1.3de0be0000000p-3, 0x1.0a2a2a0000000p-2, 0x1.7ff3760000000p-3, -0x1.c548140000000p-6, 0x1.a8122c0000000p-4, 0x1.364fd40000000p-4, -0x1.7e1c1e0000000p-10, -0x1.931b980000000p-4, 0x1.ec53100000000p-2, -0x1.2b152c0000000p-6, 0x1.3ceb620000000p-4, 0x1.5efb780000000p-4, 0x1.1634340000000p-2, -0x1.6007020000000p-3, 0x1.8ac6940000000p-3, 0x1.2ffa6e0000000p-3, 0x1.282e900000000p-4, 0x1.7a06fe0000000p-4, -0x1.0979b00000000p-7, 0x1.51d1200000000p-2, -0x1.0100980000000p-1, -0x1.1e61420000000p-6, -0x1.f37a900000000p-3, -0x1.76f2a20000000p-4, -0x1.53d1720000000p-7, 0x1.34cc800000000p-2, -0x1.d9c2e20000000p-3, -0x1.6a69720000000p-4, -0x1.f262c60000000p-6}
, {-0x1.1b64b40000000p-4, 0x1.00bca40000000p-7, -0x1.69aca00000000p-4, -0x1.34d5c60000000p-6, 0x1.164d380000000p-5, -0x1.2e490e0000000p-5, -0x1.66e3b60000000p-5, -0x1.1a46620000000p-7, 0x1.da0a4c0000000p-4, -0x1.1c1b280000000p-5, 0x1.1479440000000p-3, -0x1.28abb40000000p-3, 0x1.227be20000000p-5, 0x1.b8579e0000000p-4, 0x1.6e73e40000000p-3, 0x1.e61dd80000000p-4, 0x1.c5904a0000000p-7, -0x1.c080720000000p-4, -0x1.ddf43e0000000p-7, 0x1.4c9cbc0000000p-5, 0x1.ede2980000000p-6, -0x1.0d54fe0000000p-4, 0x1.5207c00000000p-3, -0x1.7ee79a0000000p-4, 0x1.66bbf80000000p-4, 0x1.d0b53a0000000p-6, 0x1.37521a0000000p-5, 0x1.53cc800000000p-5, -0x1.d786aa0000000p-6, -0x1.4433880000000p-4, -0x1.97d8da0000000p-9, 0x1.649b2e0000000p-5}
, {-0x1.b186000000000p-2, -0x1.4eebec0000000p-4, 0x1.9ed7160000000p-7, -0x1.54617e0000000p-4, -0x1.122c200000000p-2, -0x1.11310a0000000p-2, -0x1.0124e80000000p-3, -0x1.734fc00000000p-2, 0x1.5eca2c0000000p-15, 0x1.2dc0f40000000p-4, 0x1.2ca13a0000000p-4, -0x1.22ea760000000p-2, -0x1.869f660000000p-3, 0x1.f0bb700000000p-5, 0x1.6a52bc0000000p-3, -0x1.2f247c0000000p-3, 0x1.bebfa80000000p-5, -0x1.22fcee0000000p-2, 0x1.079a100000000p-3, -0x1.1191980000000p-4, 0x1.31a34e0000000p-5, -0x1.6b1c600000000p-3, -0x1.2737940000000p-2, 0x1.220abc0000000p-2, 0x1.6139940000000p-3, 0x1.dd790c0000000p-3, -0x1.2ff54e0000000p-2, -0x1.400bf40000000p-2, -0x1.1b70820000000p-2, 0x1.6ea4fa0000000p-3, -0x1.15f3a40000000p-3, 0x1.9c21b20000000p-5}
}
, {{-0x1.2bb3dc0000000p-2, 0x1.b067720000000p-3, 0x1.945bb40000000p-4, -0x1.c7c5500000000p-5, -0x1.80f15e0000000p-3, -0x1.e6c04a0000000p-3, -0x1.ef22fe0000000p-4, -0x1.30ed120000000p-3, -0x1.1780560000000p-5, 0x1.179b1a0000000p-5, -0x1.efd0b80000000p-6, -0x1.5a0a8c0000000p-4, -0x1.b252520000000p-3, 0x1.7f75b80000000p-5, 0x1.15b7b00000000p-3, -0x1.2251360000000p-4, -0x1.7857860000000p-9, -0x1.c0f4980000000p-3, -0x1.0b79360000000p-5, 0x1.0498420000000p-6, 0x1.19d4920000000p-4, -0x1.3e24880000000p-3, -0x1.2d5fa20000000p-3, -0x1.07e8060000000p-4, -0x1.70ba180000000p-3, 0x1.08bc040000000p-5, -0x1.12e4080000000p-2, -0x1.3bcdc20000000p-5, -0x1.a523aa0000000p-3, 0x1.0bb99c0000000p-3, -0x1.0ff6b60000000p-3, -0x1.226d2a0000000p-4}
, {-0x1.717fb40000000p-3, -0x1.87b0d00000000p-5, 0x1.8a9dfe0000000p-6, -0x1.bca37e0000000p-4, -0x1.ca5a140000000p-3, -0x1.0111860000000p-2, -0x1.c674de0000000p-6, 0x1.63b6920000000p-3, -0x1.587aa40000000p-4, 0x1.740b2c0000000p-4, 0x1.8121800000000p-4, 0x1.ffd05c0000000p-5, -0x1.2e2d880000000p-2, -0x1.0876140000000p-4, 0x1.5447460000000p-9, 0x1.8c787a0000000p-6, 0x1.4220640000000p-5, -0x1.5ab6f00000000p-3, -0x1.810d100000000p-5, -0x1.7273d20000000p-7, 0x1.4fd2420000000p-7, -0x1.3589920000000p-3, 0x1.5559020000000p-4, -0x1.8dd48e0000000p-4, -0x1.241c7c0000000p-2, 0x1.7825bc0000000p-7, -0x1.492e0a0000000p-3, 0x1.a979600000000p-3, -0x1.8407ac0000000p-4, 0x1.0a06820000000p-3, -0x1.15f6c40000000p-4, -0x1.e475600000000p-7}
, {0x1.b0ffb00000000p-2, 0x1.050cea0000000p-4, -0x1.07e3e20000000p-3, -0x1.61446a0000000p-3, -0x1.3c275c0000000p-3, -0x1.167c820000000p-4, 0x1.12a5a00000000p-3, 0x1.cbda580000000p-4, -0x1.3b02b20000000p-3, -0x1.2aca400000000p-4, 0x1.2a82820000000p-4, 0x1.74ab880000000p-5, 0x1.022b8a0000000p-2, -0x1.b9e1980000000p-4, -0x1.2347fc0000000p-6, -0x1.f5c7260000000p-6, 0x1.6e32900000000p-2, 0x1.032abe0000000p-3, -0x1.f209820000000p-4, -0x1.0f5c9c0000000p-2, -0x1.f92fa00000000p-3, -0x1.22d06a0000000p-3, 0x1.a6a1180000000p-3, -0x1.5b144c0000000p-4, -0x1.01e3e20000000p-2, -0x1.6c748e0000000p-3, 0x1.9abfa40000000p-3, 0x1.09d27e0000000p-2, 0x1.d690ac0000000p-4, -0x1.7299660000000p-4, 0x1.1d52540000000p-2, -0x1.cf700a0000000p-3}
}
, {{-0x1.92443a0000000p-5, 0x1.b7ab9c0000000p-4, 0x1.ebdffa0000000p-7, -0x1.621a960000000p-4, -0x1.9c5bcc0000000p-2, -0x1.4668920000000p-2, 0x1.57e8d20000000p-4, 0x1.aff1ae0000000p-3, -0x1.0a6c480000000p-3, 0x1.f7fe1a0000000p-6, 0x1.20a78e0000000p-3, -0x1.76da1c0000000p-2, -0x1.4788480000000p-5, 0x1.ae1d9a0000000p-5, -0x1.9bacfc0000000p-4, -0x1.6eedcc0000000p-4, 0x1.e8d40e0000000p-4, -0x1.f6c6ce0000000p-5, -0x1.5ffc200000000p-2, 0x1.2077ca0000000p-3, -0x1.270a100000000p-1, -0x1.9695380000000p-3, 0x1.0e36d40000000p-4, 0x1.1cecec0000000p-3, -0x1.1a163c0000000p-3, 0x1.7505120000000p-3, -0x1.e78c360000000p-4, 0x1.c0eb300000000p-2, -0x1.667e340000000p-3, -0x1.0e92280000000p-4, -0x1.e2b1ce0000000p-4, -0x1.b31ba20000000p-10}
, {0x1.49ec720000000p-3, 0x1.5d5c980000000p-3, 0x1.1e75440000000p-5, 0x1.ecfed00000000p-4, -0x1.c987ac0000000p-5, -0x1.d6e44c0000000p-3, 0x1.70f7400000000p-4, 0x1.2b000a0000000p-5, -0x1.26a02a0000000p-7, -0x1.40c31e0000000p-5, 0x1.e683880000000p-6, -0x1.49edc80000000p-3, 0x1.4dfed00000000p-3, 0x1.3260720000000p-6, -0x1.2d37320000000p-7, 0x1.e363fc0000000p-4, 0x1.6c91480000000p-4, 0x1.d9c7520000000p-3, -0x1.5bf2720000000p-3, 0x1.f823a80000000p-7, -0x1.740da40000000p-2, 0x1.4678080000000p-4, 0x1.7a49120000000p-5, 0x1.d13dcc0000000p-3, -0x1.09bcc20000000p-6, -0x1.be6dc20000000p-3, 0x1.ef969a0000000p-8, 0x1.528d2c0000000p-2, -0x1.6395f60000000p-3, 0x1.1784440000000p-4, -0x1.3ad13a0000000p-4, -0x1.bdee320000000p-7}
, {0x1.37a6620000000p-8, 0x1.129c580000000p-3, 0x1.3065360000000p-7, 0x1.adb2e20000000p-3, 0x1.891a1a0000000p-5, -0x1.8593f40000000p-3, 0x1.0d946a0000000p-3, 0x1.e8caf60000000p-8, -0x1.22c7560000000p-7, 0x1.3dc6d40000000p-4, 0x1.dc08da0000000p-9, -0x1.6d9b6e0000000p-3, 0x1.362fa00000000p-6, 0x1.9a3be00000000p-5, 0x1.b999ee0000000p-5, 0x1.3c3f900000000p-6, -0x1.ecd05e0000000p-5, 0x1.30c96a0000000p-8, -0x1.6e60b20000000p-3, 0x1.7fe8240000000p-3, -0x1.be763e0000000p-4, 0x1.91ef660000000p-4, -0x1.1627b80000000p-3, -0x1.7e9dba0000000p-5, 0x1.abaed40000000p-5, -0x1.29c8840000000p-4, -0x1.4cc5a00000000p-4, 0x1.7535220000000p-6, -0x1.672dc80000000p-3, 0x1.ad46900000000p-5, -0x1.ebdac60000000p-4, -0x1.6369ac0000000p-10}
}
, {{-0x1.c9baf40000000p-5, -0x1.5925ea0000000p-2, -0x1.7d48100000000p-5, 0x1.5270840000000p-7, 0x1.0b6f860000000p-2, 0x1.f7eb320000000p-3, 0x1.471b200000000p-5, -0x1.96cce60000000p-5, -0x1.1393880000000p-4, 0x1.6ffeb20000000p-4, -0x1.4f59300000000p-5, 0x1.24b1dc0000000p-4, 0x1.41b37c0000000p-4, 0x1.423d6e0000000p-4, 0x1.4453da0000000p-2, 0x1.5c0dbc0000000p-4, -0x1.d7067e0000000p-4, -0x1.f773da0000000p-4, 0x1.d87f8e0000000p-4, 0x1.bea9c60000000p-5, 0x1.a73d060000000p-3, -0x1.2faabe0000000p-4, 0x1.d762380000000p-5, 0x1.a94f2e0000000p-4, 0x1.0785820000000p-3, 0x1.e560680000000p-3, 0x1.a613ac0000000p-7, -0x1.a8eaaa0000000p-3, 0x1.a06e120000000p-3, 0x1.98d7e80000000p-8, 0x1.2b67160000000p-2, 0x1.8f69c20000000p-4}
, {-0x1.0ea64c0000000p-3, 0x1.2b5f660000000p-5, -0x1.b526980000000p-6, 0x1.742e0c0000000p-5, 0x1.d68ade0000000p-5, -0x1.2a6af80000000p-5, -0x1.edada40000000p-5, 0x1.2020600000000p-4, 0x1.016c580000000p-4, -0x1.6ccb340000000p-6, -0x1.d605940000000p-5, 0x1.0332800000000p-3, -0x1.af01380000000p-3, 0x1.aa1a4e0000000p-5, 0x1.65f4640000000p-4, -0x1.9ea6260000000p-4, 0x1.7403700000000p-5, -0x1.65bdf40000000p-3, 0x1.a4df080000000p-4, 0x1.0a57d00000000p-3, -0x1.55c4f20000000p-5, -0x1.5cbc580000000p-6, -0x1.54b91a0000000p-6, 0x1.7448ac0000000p-3, -0x1.9f682c0000000p-5, -0x1.f7ff9c0000000p-6, -0x1.6663560000000p-5, -0x1.ccadd60000000p-5, 0x1.3768300000000p-5, -0x1.908e4a0000000p-4, -0x1.308c200000000p-5, 0x1.2034460000000p-5}
, {-0x1.cd95400000000p-3, 0x1.a06d380000000p-3, 0x1.1f850a0000000p-3, -0x1.509db80000000p-4, -0x1.4ee9960000000p-3, -0x1.6f1c980000000p-2, 0x1.acbb4a0000000p-5, 0x1.241e160000000p-3, -0x1.e5f0940000000p-5, 0x1.24bc180000000p-3, 0x1.253bfa0000000p-4, 0x1.0558760000000p-3, 0x1.26c59e0000000p-6, -0x1.79a5a80000000p-2, -0x1.cffcca0000000p-3, -0x1.579be20000000p-2, 0x1.b58d060000000p-3, -0x1.e174220000000p-5, 0x1.1c5fa60000000p-3, 0x1.07ad5a0000000p-3, -0x1.31297e0000000p-2, -0x1.9cf3b20000000p-4, -0x1.2f55420000000p-2, 0x1.a6149a0000000p-6, -0x1.47a23a0000000p-4, -0x1.13fcac0000000p-3, -0x1.6c413e0000000p-6, 0x1.b8a97e0000000p-3, -0x1.a70cfe0000000p-4, 0x1.24a1960000000p-7, -0x1.65679a0000000p-2, -0x1.3beb3a0000000p-4}
}
, {{0x1.4ff5e00000000p-5, -0x1.5163900000000p-5, -0x1.56bd2a0000000p-3, -0x1.33f3ea0000000p-3, 0x1.9b60d80000000p-3, 0x1.21a1640000000p-2, -0x1.046e6e0000000p-4, 0x1.06eca80000000p-4, 0x1.3957860000000p-4, -0x1.201ef80000000p-9, 0x1.bd7d5c0000000p-7, 0x1.2df1d20000000p-3, -0x1.3c6d380000000p-6, 0x1.f7c35e0000000p-3, 0x1.ab7cfe0000000p-5, 0x1.ec2b1a0000000p-3, 0x1.7271e80000000p-7, 0x1.d65aac0000000p-4, -0x1.ac0eb80000000p-6, -0x1.3658a80000000p-6, 0x1.d9206a0000000p-5, -0x1.3187ce0000000p-3, 0x1.325ea40000000p-3, -0x1.e5fd500000000p-3, 0x1.dd1de20000000p-7, -0x1.3439500000000p-2, 0x1.dde8a20000000p-4, 0x1.787c220000000p-6, 0x1.efa8920000000p-4, -0x1.91e27a0000000p-2, -0x1.9ed3d20000000p-4, 0x1.5ed4600000000p-3}
, {-0x1.461b1e0000000p-6, -0x1.d77f020000000p-4, -0x1.9888d60000000p-8, -0x1.bdf5660000000p-4, 0x1.38f1da0000000p-4, 0x1.e6fd2e0000000p-4, -0x1.fd8df20000000p-3, -0x1.000d220000000p-3, -0x1.70ad500000000p-4, -0x1.50ecde0000000p-5, 0x1.7635b20000000p-5, -0x1.df249a0000000p-4, 0x1.584ff20000000p-5, -0x1.b898640000000p-6, 0x1.3e590c0000000p-4, 0x1.ffbcce0000000p-6, 0x1.27f3920000000p-5, -0x1.06dffc0000000p-3, -0x1.8845860000000p-4, -0x1.ffbd1e0000000p-3, -0x1.1e2ca00000000p-6, -0x1.8bef620000000p-6, -0x1.7e47ae0000000p-5, -0x1.44dea20000000p-4, -0x1.7251fc0000000p-5, -0x1.57e7ba0000000p-4, -0x1.2a1b700000000p-4, 0x1.17acfa0000000p-4, -0x1.8110280000000p-4, -0x1.68a4420000000p-4, 0x1.00e2b40000000p-5, 0x1.40a8c60000000p-3}
, {-0x1.c165fe0000000p-6, -0x1.349c3c0000000p-3, 0x1.8796ec0000000p-2, 0x1.411eb20000000p-4, -0x1.9da13e0000000p-10, 0x1.365bee0000000p-4, -0x1.988aca0000000p-3, -0x1.0ec6340000000p-3, -0x1.4491da0000000p-3, 0x1.d418ce0000000p-3, -0x1.e6321a0000000p-3, -0x1.069f280000000p-4, 0x1.ce97880000000p-3, 0x1.09aa900000000p-3, -0x1.d5b6f00000000p-5, 0x1.7447ba0000000p-4, 0x1.d9f3b40000000p-4, 0x1.2fea560000000p-7, -0x1.1b3dbc0000000p-3, -0x1.c07d280000000p-4, 0x1.a66c000000000p-4, 0x1.86379a0000000p-4, -0x1.d7d1260000000p-3, -0x1.3c33740000000p-5, -0x1.25d1f40000000p-4, 0x1.e9fff00000000p-5, -0x1.a1bab80000000p-4, -0x1.9d53120000000p-3, -0x1.41d37a0000000p-3, 0x1.a378d20000000p-3, -0x1.b4927a0000000p-4, 0x1.3c29760000000p-5}
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

typedef float batch_normalization_2_output_type[8][64];

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
#define WEIGHTS_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define NUMBER_T float
#define LONG_NUMBER_T float


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

const float batch_normalization_2_bias[64] = {-0x1.c242b20000000p-1, -0x1.23c5f00000000p-2, -0x1.6514740000000p-2, -0x1.3c02260000000p-1, -0x1.d40dc80000000p-2, -0x1.2ad91e0000000p-1, -0x1.8faf940000000p-3, -0x1.1fc2d80000000p-1, -0x1.39850a0000000p-1, -0x1.bb992c0000000p-3, 0x1.c3de800000000p-7, -0x1.ccb4b60000000p-2, -0x1.445a8c0000000p-1, -0x1.c1a6700000000p-2, -0x1.0581d80000000p+0, -0x1.148b0a0000000p-1, -0x1.529ad60000000p-3, -0x1.7fdd820000000p-1, -0x1.9e5f360000000p-2, -0x1.4e19780000000p-1, -0x1.0ce6720000000p-1, -0x1.4329020000000p-1, -0x1.95d53e0000000p-2, -0x1.6e45080000000p-3, -0x1.36a2800000000p-1, -0x1.7ad1d60000000p-2, -0x1.6c3c300000000p-4, -0x1.eb75e80000000p-4, -0x1.d81b7e0000000p-2, -0x1.af33400000000p-3, -0x1.1fcea20000000p-1, -0x1.3ed66c0000000p-3, 0x1.f0e9e00000000p-4, -0x1.a6c7580000000p-2, -0x1.5219b00000000p-1, -0x1.4beca80000000p-2, -0x1.e3f8d60000000p-2, -0x1.451f280000000p-1, -0x1.26b1900000000p-3, -0x1.7807440000000p-2, -0x1.b901d80000000p-2, -0x1.5bb3780000000p-1, -0x1.2c96820000000p-1, -0x1.f13e5a0000000p-1, -0x1.0d92900000000p-1, -0x1.11534c0000000p-1, -0x1.54b4b20000000p-1, -0x1.165a740000000p-3, -0x1.09487c0000000p-2, -0x1.8583700000000p-1, -0x1.640fb00000000p-3, -0x1.8313300000000p-2, -0x1.559fc80000000p-2, -0x1.b6b1a80000000p-4, -0x1.6c0be00000000p-2, -0x1.b588420000000p-3, -0x1.e145fc0000000p-2, -0x1.2371280000000p-2, -0x1.bc1ff00000000p-5, -0x1.0deb700000000p-1, -0x1.81a5220000000p-2, -0x1.53a89a0000000p-1, -0x1.481f900000000p-1, -0x1.fd9e620000000p-3}
;
const float batch_normalization_2_kernel[64] = {0x1.c2b69e0000000p-1, 0x1.b45c2a0000000p+0, 0x1.0b6f400000000p+0, 0x1.14de840000000p+0, 0x1.7342760000000p-1, 0x1.01955c0000000p+0, 0x1.a9421c0000000p-1, 0x1.7fb3a00000000p+0, 0x1.5d05c60000000p-1, 0x1.06e9260000000p+0, 0x1.ac82ea0000000p-1, 0x1.e7b2280000000p-1, 0x1.25a7320000000p-1, 0x1.005cc20000000p+0, 0x1.3fd4c20000000p+0, 0x1.e422d80000000p-1, 0x1.46d6260000000p+0, 0x1.6ab1ca0000000p+0, 0x1.095f2e0000000p+0, 0x1.938f620000000p+1, 0x1.5af1560000000p-1, 0x1.9f47b80000000p-1, 0x1.aadf280000000p+0, 0x1.c127b60000000p-1, 0x1.217d4a0000000p-1, 0x1.3fd5140000000p+0, 0x1.55e4f20000000p-1, 0x1.862ff20000000p-1, 0x1.2984940000000p-1, 0x1.30161a0000000p-1, 0x1.15bc6e0000000p+1, 0x1.087b4a0000000p+1, 0x1.275eb80000000p+0, 0x1.83d89e0000000p-1, 0x1.8c619e0000000p-1, 0x1.0490c40000000p+0, 0x1.4bfe340000000p+0, 0x1.a2d1dc0000000p-1, 0x1.3d116e0000000p+0, 0x1.29bcd80000000p+0, 0x1.f567700000000p-1, 0x1.d3eefa0000000p-1, 0x1.4462ee0000000p-1, 0x1.968faa0000000p-2, 0x1.05701c0000000p+1, 0x1.c84af80000000p-1, 0x1.1c1e300000000p-1, 0x1.0c97780000000p+0, 0x1.80b8ce0000000p-1, 0x1.8d35200000000p+0, 0x1.34a0400000000p-1, 0x1.2652300000000p-1, 0x1.aa16320000000p-1, 0x1.d05cb20000000p-1, 0x1.b71c920000000p-2, 0x1.c737f20000000p-1, 0x1.c9ed440000000p-1, 0x1.f23ee60000000p-1, 0x1.1908080000000p+0, 0x1.b327880000000p-1, 0x1.4747700000000p+0, 0x1.5a84820000000p-1, 0x1.9fe36a0000000p-1, 0x1.c8879a0000000p-1}
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

typedef float max_pooling1d_2_output_type[POOL_LENGTH][INPUT_CHANNELS];

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
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define NUMBER_T float
#define LONG_NUMBER_T float


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

typedef float conv1d_3_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

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
#define WEIGHTS_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define NUMBER_T float
#define LONG_NUMBER_T float


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


#error "Data type unsupported by CMSIS-NN"

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


const float  conv1d_3_bias[CONV_FILTERS] = {0x1.47cdec0000000p-3, -0x1.b54df80000000p-3, 0x1.d516d00000000p-4, -0x1.a57e540000000p-7, 0x1.7abc500000000p-5, 0x1.925e120000000p-4, 0x1.2b6a560000000p-4, 0x1.889e320000000p-4, 0x1.1cd5760000000p-2, -0x1.2a5bc40000000p-6, 0x1.36b8200000000p-4, 0x1.9900220000000p-4, 0x1.2b66140000000p-4, 0x1.08cf340000000p-4, 0x1.ce2af00000000p-5, 0x1.b8dffc0000000p-4, -0x1.1211360000000p-4, 0x1.4a19a80000000p-4, -0x1.0b41b80000000p-4, 0x1.b14d200000000p-4, 0x1.fb31920000000p-3, 0x1.c595740000000p-4, 0x1.3d85d20000000p-4, 0x1.1c8af60000000p-2, 0x1.3baaae0000000p-2, 0x1.542dec0000000p-4, 0x1.4b0d9e0000000p-6, 0x1.5bb8fe0000000p-3, 0x1.14d4140000000p-2, 0x1.296aa00000000p-3, 0x1.03c4e80000000p-3, 0x1.40516c0000000p-4, 0x1.5623a60000000p-3, 0x1.0ee9b00000000p-6, 0x1.6d68240000000p-3, 0x1.4d3f800000000p-3, 0x1.1918b40000000p-3, 0x1.1f56720000000p-3, 0x1.47f6a20000000p-4, 0x1.905a2e0000000p-4, 0x1.e3edcc0000000p-4, -0x1.ed9cb60000000p-4, 0x1.1b08f40000000p-3, 0x1.457da60000000p-3, 0x1.a22f900000000p-4, 0x1.4800b20000000p-4, 0x1.aebc5e0000000p-5, 0x1.5034a60000000p-3, 0x1.3cb1da0000000p-5, 0x1.e83fa00000000p-4, 0x1.6350fe0000000p-3, -0x1.def3020000000p-6, 0x1.b476c20000000p-5, 0x1.92db080000000p-4, 0x1.2008160000000p-5, 0x1.20cc340000000p-3, 0x1.3eef6e0000000p-3, 0x1.fe48340000000p-5, 0x1.2afcf60000000p-5, 0x1.adae9c0000000p-4, 0x1.c420780000000p-4, 0x1.10f9620000000p-4, -0x1.594cae0000000p-4, 0x1.412e5e0000000p-6}
;

const float  conv1d_3_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS] = {{{0x1.111b9c0000000p-4, 0x1.4e5b480000000p-3, 0x1.614f240000000p-2, -0x1.2423a20000000p-4, 0x1.367bd80000000p-5, -0x1.307ba20000000p-4, -0x1.7e1d6a0000000p-3, -0x1.60d46c0000000p-5, 0x1.89b5200000000p-2, 0x1.055ec80000000p-4, 0x1.5f95ea0000000p-5, 0x1.866e6e0000000p-3, -0x1.3164740000000p-8, 0x1.885dba0000000p-5, -0x1.7136720000000p-4, -0x1.5916200000000p-9, -0x1.20b8960000000p-4, -0x1.c6f8c60000000p-3, -0x1.135da40000000p-5, -0x1.76bb200000000p-5, -0x1.0000340000000p-8, 0x1.6e076e0000000p-5, 0x1.acf84a0000000p-4, -0x1.a75cf20000000p-6, 0x1.6a024a0000000p-4, 0x1.5457120000000p-4, 0x1.7d01d00000000p-3, 0x1.00d3de0000000p-6, 0x1.363ed40000000p-8, 0x1.89f2a40000000p-4, -0x1.a929060000000p-4, -0x1.a49fd80000000p-5, -0x1.5c114c0000000p-3, 0x1.6fca8a0000000p-7, 0x1.8e256a0000000p-6, -0x1.6620960000000p-4, 0x1.87dd060000000p-9, -0x1.1004900000000p-4, 0x1.e03be60000000p-3, 0x1.346f180000000p-2, 0x1.2ef9e00000000p-5, -0x1.cc65540000000p-7, 0x1.b240000000000p-4, 0x1.3daa3a0000000p-4, 0x1.90b64e0000000p-3, -0x1.371c9c0000000p-4, -0x1.9380e20000000p-5, 0x1.c6db2e0000000p-6, 0x1.2c68c20000000p-4, -0x1.bcb1ee0000000p-4, 0x1.00234c0000000p-2, 0x1.c615760000000p-3, 0x1.76d55a0000000p-3, 0x1.dcbd820000000p-6, 0x1.3671be0000000p-6, -0x1.3bd03e0000000p-3, -0x1.ec7b9a0000000p-3, 0x1.632df00000000p-3, 0x1.fa6eb20000000p-6, 0x1.8b7d5e0000000p-5, -0x1.7081740000000p-3, 0x1.b2a4a00000000p-4, 0x1.2437680000000p-8, 0x1.71c09a0000000p-3}
, {-0x1.161e080000000p-3, -0x1.af39580000000p-7, 0x1.1b638e0000000p-5, -0x1.31f4ce0000000p-3, 0x1.d0786a0000000p-6, -0x1.0763640000000p-5, 0x1.bbf2740000000p-4, 0x1.87cffe0000000p-6, 0x1.f3ce300000000p-4, -0x1.b7903a0000000p-5, 0x1.be8ba60000000p-4, 0x1.b782e80000000p-4, 0x1.da10f20000000p-4, 0x1.8e8fa80000000p-4, 0x1.9487800000000p-4, -0x1.60f2520000000p-7, 0x1.6f2bfe0000000p-4, -0x1.a7432c0000000p-4, -0x1.8d29560000000p-4, -0x1.bf12a00000000p-4, -0x1.a13be60000000p-4, -0x1.28c8660000000p-4, -0x1.73b3360000000p-5, 0x1.fba59c0000000p-5, -0x1.47548c0000000p-3, 0x1.a9d9100000000p-3, 0x1.c563e60000000p-4, -0x1.6811dc0000000p-4, -0x1.4abdc80000000p-3, -0x1.056e980000000p-4, -0x1.55a5a20000000p-3, -0x1.b15bbc0000000p-4, -0x1.32725a0000000p-6, -0x1.25cb660000000p-4, -0x1.1b63560000000p-3, 0x1.7b87560000000p-4, 0x1.1cc2380000000p-3, 0x1.6df32e0000000p-5, 0x1.b61df60000000p-4, -0x1.6527ee0000000p-7, -0x1.6f32660000000p-6, -0x1.14e9e60000000p-5, -0x1.ba07b00000000p-4, -0x1.ff31600000000p-6, -0x1.bce0860000000p-5, -0x1.a7d6b60000000p-3, 0x1.8055ae0000000p-6, -0x1.423c8a0000000p-4, -0x1.5ffc7e0000000p-3, -0x1.6ba4c60000000p-3, 0x1.eed33c0000000p-3, 0x1.30111c0000000p-4, -0x1.5ee3960000000p-4, -0x1.3d168e0000000p-4, -0x1.080e500000000p-2, -0x1.9a8df00000000p-3, 0x1.a832440000000p-5, 0x1.6075860000000p-5, -0x1.f75c0a0000000p-5, 0x1.3b45100000000p-4, -0x1.52fcf00000000p-5, 0x1.de99a60000000p-6, -0x1.81ad380000000p-5, 0x1.05c0120000000p-9}
, {0x1.b653a80000000p-4, -0x1.b2edf40000000p-3, 0x1.5099e60000000p-3, 0x1.0fb6440000000p-3, 0x1.8260960000000p-5, -0x1.f0924c0000000p-3, -0x1.59de820000000p-5, 0x1.0534380000000p-4, 0x1.da76360000000p-4, -0x1.288a9a0000000p-7, -0x1.6da39c0000000p-5, 0x1.58a6f20000000p-8, 0x1.f0a7200000000p-6, -0x1.5623860000000p-4, -0x1.de304e0000000p-4, -0x1.2339c40000000p-3, -0x1.189d9c0000000p-2, -0x1.4a92a20000000p-3, 0x1.053af40000000p-4, -0x1.2b014a0000000p-3, -0x1.b617ce0000000p-5, 0x1.2a90380000000p-5, 0x1.6c037e0000000p-6, 0x1.1ed4920000000p-5, -0x1.29dfc20000000p-3, 0x1.4dffc60000000p-2, 0x1.4d33100000000p-5, 0x1.760c840000000p-6, -0x1.2c83600000000p-4, -0x1.670ba80000000p-9, -0x1.2f9ba00000000p-4, -0x1.afe3d40000000p-3, -0x1.b1e0e20000000p-7, -0x1.34d93e0000000p-3, -0x1.1bb0de0000000p-4, 0x1.17f9420000000p-3, 0x1.ef484c0000000p-5, -0x1.1d5f2a0000000p-6, -0x1.730ce60000000p-6, -0x1.2442500000000p-4, -0x1.27ffe40000000p-3, -0x1.178a1c0000000p-4, -0x1.9d3f300000000p-5, 0x1.8066500000000p-6, -0x1.094a500000000p-2, 0x1.1180640000000p-4, 0x1.b36dce0000000p-4, 0x1.2bb46a0000000p-5, -0x1.9f87e00000000p-4, -0x1.2998a20000000p-3, 0x1.a45b640000000p-4, -0x1.67ce9a0000000p-8, 0x1.d85c220000000p-5, 0x1.0c61160000000p-5, -0x1.5328740000000p-3, -0x1.159d780000000p-3, 0x1.ee0f600000000p-5, 0x1.1ec2080000000p-2, -0x1.268ef80000000p-3, -0x1.8565f20000000p-5, -0x1.c33df40000000p-4, 0x1.3eaff00000000p-5, 0x1.326fae0000000p-3, -0x1.1875d20000000p-3}
}
, {{-0x1.e399980000000p-4, 0x1.4019180000000p-10, -0x1.fdd89e0000000p-2, 0x1.29dc400000000p-2, 0x1.a79b920000000p-3, -0x1.d853440000000p-3, 0x1.21a95a0000000p-3, -0x1.0faef40000000p-2, -0x1.2cd45e0000000p-2, 0x1.77aabc0000000p-5, -0x1.10c7f00000000p-4, -0x1.0dbda80000000p-2, -0x1.5172680000000p-6, 0x1.d1d3b80000000p-3, -0x1.136f960000000p-4, 0x1.9921980000000p-3, -0x1.87cca80000000p-5, -0x1.7d05e00000000p-3, -0x1.57fd840000000p-3, 0x1.b2ceea0000000p-3, 0x1.0a8e6e0000000p-5, -0x1.3e0b3a0000000p-3, 0x1.5e748e0000000p-3, 0x1.0564ec0000000p-3, 0x1.08dee60000000p-3, -0x1.466e0a0000000p-3, -0x1.e479ca0000000p-4, 0x1.322b500000000p-13, -0x1.deb1720000000p-7, -0x1.a369e80000000p-4, -0x1.45ed4c0000000p-3, 0x1.56cdd00000000p-2, 0x1.4952680000000p-6, -0x1.b5b9dc0000000p-3, -0x1.43093c0000000p-3, -0x1.cb10500000000p-2, -0x1.74432c0000000p-3, -0x1.7a6bce0000000p-3, 0x1.dcd0220000000p-3, -0x1.06a2580000000p-3, 0x1.9175d60000000p-3, -0x1.c5f74a0000000p-6, -0x1.8208ea0000000p-4, 0x1.8103b60000000p-5, 0x1.e94b960000000p-4, -0x1.8d93880000000p-9, 0x1.7729640000000p-3, -0x1.1414140000000p-1, -0x1.0b1a520000000p-4, 0x1.f23f820000000p-4, -0x1.45948a0000000p-2, -0x1.25b5e20000000p-3, -0x1.900cd60000000p-2, -0x1.2465c80000000p-4, 0x1.64ff3c0000000p-6, 0x1.43fb060000000p-4, 0x1.46e7560000000p-5, -0x1.667c000000000p-3, 0x1.9158a60000000p-4, 0x1.8bb6020000000p-2, 0x1.4599500000000p-6, -0x1.24c61e0000000p-3, -0x1.f03b120000000p-6, 0x1.501a340000000p-7}
, {0x1.a7303a0000000p-6, -0x1.52e0a40000000p-3, -0x1.bbdb760000000p-6, 0x1.30b7380000000p-4, 0x1.c5b2820000000p-4, -0x1.4aa4a20000000p-4, -0x1.8a45100000000p-8, 0x1.ad79800000000p-3, 0x1.f9bd000000000p-4, -0x1.b1e2860000000p-3, -0x1.317ca20000000p-3, 0x1.7c98de0000000p-3, 0x1.4ef0920000000p-3, 0x1.134cc80000000p-7, 0x1.69ba920000000p-3, 0x1.8483d80000000p-5, -0x1.15aad00000000p-3, 0x1.a3cb720000000p-3, -0x1.35c61a0000000p-3, 0x1.93187a0000000p-4, 0x1.9d2bda0000000p-5, 0x1.4706460000000p-5, 0x1.a7e5900000000p-4, 0x1.45370c0000000p-6, -0x1.87ad4a0000000p-7, 0x1.8a58540000000p-3, 0x1.5aa0b60000000p-4, -0x1.d37bd60000000p-3, -0x1.a4ecb20000000p-3, -0x1.4c930e0000000p-3, -0x1.97f51a0000000p-6, -0x1.3f97160000000p-5, 0x1.9b6bbe0000000p-6, 0x1.34e7780000000p-4, -0x1.bc6ab00000000p-7, -0x1.056fe00000000p-5, 0x1.e802d60000000p-3, -0x1.aba89a0000000p-4, -0x1.2beb2c0000000p-3, -0x1.34b2de0000000p-3, 0x1.1a63d40000000p-4, 0x1.db9a040000000p-5, 0x1.69e9940000000p-4, -0x1.466c3a0000000p-4, -0x1.a361ae0000000p-7, 0x1.06e9de0000000p-3, 0x1.4f5a980000000p-5, 0x1.3677ee0000000p-3, 0x1.e53c640000000p-5, 0x1.659ec00000000p-3, 0x1.2abc2c0000000p-7, -0x1.0d9cb80000000p-4, 0x1.1cb87c0000000p-3, -0x1.6215560000000p-4, -0x1.443f1a0000000p-3, 0x1.9341b20000000p-3, 0x1.1b4b800000000p-4, 0x1.b7519c0000000p-3, -0x1.8daf140000000p-7, 0x1.133d680000000p-3, 0x1.9a48400000000p-5, 0x1.c5afe20000000p-4, -0x1.b434260000000p-9, -0x1.3dc43a0000000p-3}
, {0x1.119c8c0000000p-3, 0x1.b6c3ae0000000p-3, 0x1.36ffb20000000p-5, -0x1.0ec3920000000p-4, 0x1.f3d70a0000000p-3, -0x1.53a0de0000000p-3, 0x1.fc2d9a0000000p-4, 0x1.1df5420000000p-4, -0x1.7faeea0000000p-2, -0x1.0d9dca0000000p-6, 0x1.68d3c20000000p-3, -0x1.390a560000000p-6, 0x1.3dd3100000000p-8, -0x1.6e20f80000000p-4, 0x1.a3cda60000000p-4, 0x1.b2a4800000000p-10, 0x1.29d9b20000000p-4, 0x1.0e05ba0000000p-2, -0x1.5495ca0000000p-2, 0x1.14cb420000000p-6, -0x1.14d1240000000p-4, -0x1.9e68780000000p-5, -0x1.87a8c80000000p-3, -0x1.a0716e0000000p-7, -0x1.38e1480000000p-3, -0x1.d271600000000p-5, 0x1.a4792c0000000p-7, -0x1.24da860000000p-7, 0x1.0a3cc00000000p-7, -0x1.979e100000000p-5, 0x1.dad01e0000000p-3, 0x1.6f5e2a0000000p-2, 0x1.a0981c0000000p-6, -0x1.65b5960000000p-4, -0x1.8606be0000000p-3, 0x1.051fbc0000000p-5, 0x1.3e0d380000000p-2, -0x1.0296140000000p-1, 0x1.6865b60000000p-3, -0x1.389bc40000000p-3, 0x1.65ae9c0000000p-2, 0x1.4322640000000p-4, -0x1.32e80c0000000p-5, 0x1.ed1e940000000p-4, -0x1.57940c0000000p-6, 0x1.04bc860000000p-3, -0x1.1c43460000000p-3, -0x1.a0ba920000000p-4, -0x1.32280a0000000p-3, 0x1.494bb60000000p-4, 0x1.1a59640000000p-4, 0x1.bb28ae0000000p-3, -0x1.3a83fa0000000p-3, 0x1.172b9a0000000p-3, -0x1.aded380000000p-3, 0x1.5177ec0000000p-2, -0x1.f51bda0000000p-7, -0x1.5517160000000p-5, -0x1.24e5220000000p-3, 0x1.a9b3dc0000000p-5, 0x1.b490300000000p-3, 0x1.bb420e0000000p-5, 0x1.0eb23e0000000p-5, 0x1.420c820000000p-5}
}
, {{0x1.14fdce0000000p-4, 0x1.6684340000000p-4, 0x1.53740c0000000p-8, -0x1.b0008e0000000p-4, -0x1.f3018a0000000p-5, 0x1.3cc2340000000p-3, 0x1.61107e0000000p-5, -0x1.81ee860000000p-7, -0x1.7b337e0000000p-4, 0x1.5dbc3e0000000p-4, 0x1.6fe17a0000000p-3, 0x1.27ac680000000p-4, 0x1.37f5ac0000000p-4, 0x1.02a1880000000p-5, -0x1.c495280000000p-3, 0x1.3954f20000000p-4, 0x1.5416dc0000000p-4, -0x1.adba220000000p-3, -0x1.6e8af80000000p-5, -0x1.a1e9500000000p-7, -0x1.29141c0000000p-4, 0x1.9ec8200000000p-4, -0x1.dcb24e0000000p-3, 0x1.034f4a0000000p-3, -0x1.35429e0000000p-5, 0x1.4998920000000p-5, 0x1.d661220000000p-4, -0x1.c6b43a0000000p-4, 0x1.993f880000000p-4, -0x1.ccc35a0000000p-6, -0x1.8b11e80000000p-5, 0x1.45d4ea0000000p-5, -0x1.13acfc0000000p-4, -0x1.34b3540000000p-5, 0x1.7b372c0000000p-4, 0x1.1d02d40000000p-3, -0x1.f1137e0000000p-5, 0x1.a6bf020000000p-8, 0x1.0424980000000p-3, 0x1.075e020000000p-2, 0x1.cf8bbe0000000p-3, 0x1.da4f480000000p-4, 0x1.1b600e0000000p-4, -0x1.3a634e0000000p-3, -0x1.c2c88c0000000p-4, 0x1.c9a27c0000000p-4, 0x1.bf4ac80000000p-3, -0x1.05072a0000000p-5, 0x1.6f881a0000000p-4, 0x1.b964600000000p-8, 0x1.f48e2a0000000p-4, 0x1.a419020000000p-7, 0x1.a948bc0000000p-6, 0x1.ec2e2a0000000p-5, -0x1.b94f340000000p-4, 0x1.d15d520000000p-5, 0x1.a973700000000p-5, -0x1.3cfce20000000p-6, 0x1.8168ba0000000p-4, -0x1.4be3ac0000000p-4, -0x1.e363de0000000p-4, 0x1.8e7f040000000p-3, 0x1.fb70e40000000p-6, -0x1.983d320000000p-5}
, {-0x1.a2a1340000000p-3, 0x1.1114ce0000000p-5, 0x1.10a8b80000000p-3, 0x1.e36d3e0000000p-14, 0x1.6ebf480000000p-4, -0x1.7653940000000p-5, 0x1.bf598a0000000p-6, -0x1.01dc000000000p-11, 0x1.d8961c0000000p-9, 0x1.039a660000000p-3, -0x1.3381ea0000000p-7, 0x1.7e9a6c0000000p-5, 0x1.3a2c360000000p-9, 0x1.462a440000000p-4, 0x1.1ece940000000p-6, 0x1.07dece0000000p-6, 0x1.cff39a0000000p-4, -0x1.07201e0000000p-3, -0x1.6be0640000000p-6, -0x1.5cc1960000000p-5, 0x1.74acd00000000p-4, -0x1.6d51c40000000p-4, 0x1.03d7f00000000p-2, 0x1.0d4c760000000p-4, 0x1.4c2b6c0000000p-3, -0x1.6f40280000000p-7, 0x1.d775c60000000p-4, 0x1.2de7240000000p-4, 0x1.c4ee1e0000000p-5, 0x1.53147e0000000p-5, -0x1.2a52260000000p-3, 0x1.936f1a0000000p-4, 0x1.2b463e0000000p-3, 0x1.1dd94e0000000p-4, -0x1.13ce260000000p-4, -0x1.cc5c660000000p-4, 0x1.6c43200000000p-4, 0x1.830bdc0000000p-4, 0x1.7018b40000000p-3, -0x1.92a9c60000000p-5, 0x1.0381620000000p-6, 0x1.a4961e0000000p-4, 0x1.b14fda0000000p-4, 0x1.276c6e0000000p-5, 0x1.06ea940000000p-6, 0x1.35215a0000000p-3, 0x1.15fa800000000p-4, 0x1.db61b20000000p-6, -0x1.0b70a60000000p-5, -0x1.1982760000000p-3, 0x1.92e4c40000000p-3, 0x1.7fef940000000p-4, -0x1.8b5e620000000p-6, 0x1.9c5f2a0000000p-3, -0x1.10ab6e0000000p-4, 0x1.ded5960000000p-4, -0x1.19cbd60000000p-4, 0x1.fcaf0c0000000p-5, 0x1.5355120000000p-4, 0x1.b152b00000000p-3, -0x1.1a12fa0000000p-3, 0x1.e8225e0000000p-5, 0x1.a862060000000p-3, 0x1.38012a0000000p-3}
, {-0x1.4d517e0000000p-3, 0x1.1545ba0000000p-5, 0x1.6a9f340000000p-6, -0x1.b622cc0000000p-3, -0x1.655b160000000p-7, 0x1.205a2e0000000p-3, -0x1.20f59a0000000p-3, 0x1.57952c0000000p-4, -0x1.2cc1c00000000p-5, -0x1.304d4c0000000p-3, 0x1.16965c0000000p-3, -0x1.727ea00000000p-5, -0x1.bdd42a0000000p-5, -0x1.7841600000000p-3, 0x1.596ebc0000000p-5, 0x1.cda1b40000000p-8, 0x1.2e554e0000000p-3, 0x1.98d14a0000000p-4, 0x1.e4dec20000000p-4, -0x1.6290980000000p-3, -0x1.dbbd620000000p-7, -0x1.7d9d460000000p-5, -0x1.fe8c4c0000000p-5, -0x1.bd8af80000000p-4, 0x1.a6f7aa0000000p-5, -0x1.16602c0000000p-5, -0x1.c9b8ce0000000p-9, -0x1.f7f5b60000000p-4, 0x1.5561320000000p-4, 0x1.817b440000000p-3, -0x1.e75f560000000p-4, 0x1.9b44760000000p-4, 0x1.b6eab20000000p-4, 0x1.7502440000000p-5, 0x1.a98dbc0000000p-6, 0x1.16e6860000000p-2, 0x1.98b32e0000000p-4, -0x1.ee5da00000000p-5, -0x1.e18a4a0000000p-4, 0x1.7369840000000p-3, -0x1.55d3680000000p-4, -0x1.e28a880000000p-5, -0x1.0d6e000000000p-5, 0x1.8fd55c0000000p-6, -0x1.65cd3c0000000p-2, -0x1.bd4ca00000000p-4, 0x1.5d21f60000000p-4, -0x1.0e155c0000000p-5, 0x1.026ba00000000p-3, -0x1.e33fbc0000000p-4, 0x1.47c8520000000p-5, 0x1.163a2a0000000p-5, 0x1.2ccc100000000p-3, 0x1.612b020000000p-5, 0x1.6bd9160000000p-6, -0x1.cac82e0000000p-6, 0x1.943a8a0000000p-4, 0x1.1abf920000000p-5, -0x1.d2ceca0000000p-9, -0x1.1ca2380000000p-4, -0x1.ee93b40000000p-4, 0x1.6d5dfe0000000p-7, 0x1.c18eee0000000p-4, -0x1.cd01cc0000000p-5}
}
, {{-0x1.6752480000000p-3, -0x1.6c5a740000000p-2, 0x1.f5b84a0000000p-3, 0x1.26d5860000000p-7, -0x1.112c500000000p-3, -0x1.51da260000000p-4, 0x1.e4825a0000000p-6, 0x1.74f9da0000000p-4, 0x1.4946220000000p-3, 0x1.9291900000000p-3, -0x1.e9c89e0000000p-4, 0x1.f696420000000p-3, -0x1.206e720000000p-5, -0x1.1b35c00000000p-2, 0x1.243bda0000000p-2, -0x1.53f4d80000000p-4, -0x1.643c140000000p-4, -0x1.4029b00000000p-4, -0x1.e272080000000p-3, -0x1.b926200000000p-5, -0x1.7e653c0000000p-4, -0x1.0d5db80000000p-3, 0x1.2e826a0000000p-2, -0x1.5a53c40000000p-4, -0x1.f049540000000p-8, 0x1.635cfc0000000p-5, 0x1.a170c20000000p-4, -0x1.c70bf40000000p-3, -0x1.b9adf40000000p-3, 0x1.7aef360000000p-4, 0x1.63c1800000000p-4, -0x1.d258dc0000000p-2, 0x1.d8e8ac0000000p-3, 0x1.febc9a0000000p-5, -0x1.0376ae0000000p-5, -0x1.06ff320000000p-4, 0x1.2fdeea0000000p-4, 0x1.411aac0000000p-3, -0x1.da268e0000000p-3, 0x1.8017940000000p-6, -0x1.b2e5880000000p-2, -0x1.078fb60000000p-4, 0x1.95c8f60000000p-7, -0x1.6fa22e0000000p-5, -0x1.d284480000000p-10, 0x1.1681760000000p-4, -0x1.4ed1ba0000000p-2, 0x1.ef62ec0000000p-3, 0x1.a69fec0000000p-6, 0x1.ef445c0000000p-5, 0x1.09a4840000000p-7, -0x1.5781e40000000p-5, 0x1.e9cf780000000p-6, -0x1.6a82f60000000p-3, -0x1.45f18e0000000p-3, -0x1.ffa9a40000000p-8, -0x1.d3f5dc0000000p-4, 0x1.9f50420000000p-3, -0x1.cfdd000000000p-4, 0x1.d7f96a0000000p-4, 0x1.ae36b40000000p-6, -0x1.533cb20000000p-6, -0x1.0cd9720000000p-3, -0x1.1b04fa0000000p-8}
, {0x1.388a800000000p-4, -0x1.43baa20000000p-5, 0x1.ffecb80000000p-4, -0x1.fa3a8a0000000p-5, 0x1.0a20fa0000000p-4, 0x1.3321f60000000p-4, 0x1.ec0a6a0000000p-5, 0x1.2c4fc80000000p-3, -0x1.d10a4e0000000p-4, -0x1.0119e00000000p-3, 0x1.ef24820000000p-5, 0x1.2dc0480000000p-3, -0x1.781aee0000000p-4, -0x1.1274660000000p-2, 0x1.4fc9120000000p-3, 0x1.91ed5a0000000p-5, -0x1.f0c7740000000p-6, 0x1.6303240000000p-7, -0x1.6796160000000p-4, 0x1.fe88f20000000p-4, -0x1.0484020000000p-2, 0x1.38b4d00000000p-3, -0x1.f133160000000p-4, 0x1.8398e60000000p-6, 0x1.a802260000000p-5, 0x1.26ca800000000p-3, 0x1.42ffe20000000p-7, 0x1.3cc4600000000p-4, 0x1.3ca4440000000p-7, 0x1.567ad60000000p-6, 0x1.e2f26c0000000p-4, 0x1.6d94b60000000p-4, 0x1.d2a3360000000p-4, 0x1.d9a0ac0000000p-5, 0x1.522b060000000p-4, 0x1.d4760e0000000p-3, 0x1.8fc57a0000000p-3, 0x1.95ca380000000p-4, 0x1.976c3a0000000p-8, -0x1.93ae9c0000000p-4, 0x1.ec64020000000p-4, 0x1.d59dac0000000p-5, -0x1.2ad71c0000000p-4, 0x1.80100c0000000p-7, -0x1.839fe00000000p-4, -0x1.f9ac500000000p-5, -0x1.a1d3820000000p-6, -0x1.ad1b4e0000000p-5, -0x1.0b160a0000000p-6, 0x1.7e525e0000000p-2, -0x1.1b40560000000p-5, -0x1.d690ac0000000p-4, 0x1.8399560000000p-9, -0x1.34daba0000000p-4, -0x1.de2e480000000p-4, 0x1.6c75cc0000000p-4, -0x1.7132260000000p-7, -0x1.2862720000000p-5, 0x1.2c736e0000000p-3, 0x1.3c4a140000000p-4, 0x1.6108440000000p-2, 0x1.3f37c60000000p-4, 0x1.2b90e20000000p-4, 0x1.5571b80000000p-7}
, {0x1.5989ee0000000p-4, -0x1.8188ac0000000p-4, 0x1.ba2bfc0000000p-4, 0x1.f22f8c0000000p-5, -0x1.85b0ca0000000p-4, 0x1.c1ccc60000000p-6, 0x1.c57a380000000p-5, 0x1.a029f40000000p-4, 0x1.c512fc0000000p-4, 0x1.7e7bbe0000000p-3, -0x1.c11ff80000000p-6, 0x1.196a860000000p-6, -0x1.8e63fc0000000p-3, -0x1.cd7b500000000p-5, 0x1.2ae4060000000p-8, 0x1.a5a3340000000p-5, 0x1.0867620000000p-6, -0x1.3de41e0000000p-3, 0x1.b088ba0000000p-4, -0x1.fbbb760000000p-5, 0x1.3561440000000p-3, -0x1.0465e60000000p-4, 0x1.47d83a0000000p-5, 0x1.ee651c0000000p-4, -0x1.7f16d00000000p-8, -0x1.a1a1f20000000p-5, 0x1.50e89e0000000p-5, -0x1.5049d20000000p-4, -0x1.99f61e0000000p-5, -0x1.b858980000000p-3, 0x1.12df020000000p-3, -0x1.1694d60000000p-3, -0x1.0248c40000000p-3, 0x1.2482540000000p-4, -0x1.60b1980000000p-9, 0x1.8b2eaa0000000p-4, -0x1.16730e0000000p-3, 0x1.c158b40000000p-4, 0x1.9639cc0000000p-4, 0x1.e989fe0000000p-6, -0x1.1d9bae0000000p-4, 0x1.2c77840000000p-5, -0x1.2e75ce0000000p-3, -0x1.a6ee920000000p-7, -0x1.c124ec0000000p-4, -0x1.557e460000000p-5, 0x1.f9fd1a0000000p-4, 0x1.854f040000000p-3, -0x1.7a93b00000000p-3, -0x1.531c080000000p-3, 0x1.47ed100000000p-4, 0x1.7350e80000000p-5, -0x1.b5fe2e0000000p-5, -0x1.3d809e0000000p-3, 0x1.8d41c60000000p-4, -0x1.3951d80000000p-2, 0x1.c4f34a0000000p-5, 0x1.1a2cda0000000p-3, -0x1.70c51e0000000p-4, -0x1.5efe540000000p-3, 0x1.256d6e0000000p-4, 0x1.06ed400000000p-5, -0x1.b8a8140000000p-4, -0x1.b6d79a0000000p-9}
}
, {{-0x1.5742aa0000000p-3, -0x1.b9b8400000000p-3, 0x1.8a4b8c0000000p-5, 0x1.a139ee0000000p-4, -0x1.7785ca0000000p-6, -0x1.962bca0000000p-6, 0x1.48bb3a0000000p-4, 0x1.ad9e740000000p-5, -0x1.1f17840000000p-5, -0x1.6d19ec0000000p-3, -0x1.ef03e00000000p-4, 0x1.b2bf8c0000000p-4, 0x1.f2b8d80000000p-7, 0x1.8959060000000p-5, 0x1.54ce640000000p-4, 0x1.3726a60000000p-4, -0x1.6ca5c20000000p-6, 0x1.1143d40000000p-4, -0x1.cf87940000000p-6, 0x1.af77460000000p-4, 0x1.6bffe60000000p-4, 0x1.d4eb8e0000000p-8, 0x1.18aef00000000p-4, -0x1.2245ba0000000p-3, -0x1.b0a9500000000p-5, 0x1.265e280000000p-4, 0x1.bae3b40000000p-4, -0x1.cb8fe40000000p-5, -0x1.fcbd440000000p-5, -0x1.9ac9b80000000p-4, 0x1.3558c60000000p-5, -0x1.9a052e0000000p-7, -0x1.d59f000000000p-6, -0x1.52bc300000000p-3, -0x1.b8f6300000000p-3, -0x1.075ef20000000p-3, -0x1.18697e0000000p-7, 0x1.031b1c0000000p-3, 0x1.021cdc0000000p-6, -0x1.708da20000000p-3, 0x1.c5b55a0000000p-3, 0x1.485b060000000p-5, -0x1.8633c40000000p-5, 0x1.af4b900000000p-7, 0x1.6c829e0000000p-3, 0x1.e6b28a0000000p-6, -0x1.0044620000000p-3, -0x1.148f4e0000000p-5, -0x1.3bba640000000p-3, 0x1.dc2c440000000p-6, 0x1.1e29fa0000000p-3, -0x1.47e8d60000000p-4, -0x1.18f4500000000p-3, -0x1.69ec620000000p-3, -0x1.5f7b560000000p-4, -0x1.c768860000000p-6, 0x1.58ba100000000p-4, 0x1.c60dd20000000p-5, -0x1.9655860000000p-5, 0x1.99f4880000000p-5, 0x1.9575000000000p-4, -0x1.79806c0000000p-3, -0x1.3c15a60000000p-3, 0x1.76914c0000000p-3}
, {0x1.3607fa0000000p-3, -0x1.9ab0e60000000p-3, 0x1.53aac40000000p-5, -0x1.24d6f20000000p-7, -0x1.df11460000000p-4, 0x1.e1938a0000000p-5, 0x1.03b11a0000000p-6, -0x1.2dffd00000000p-6, 0x1.4e44f80000000p-4, 0x1.bb46ee0000000p-6, -0x1.87b1240000000p-4, 0x1.625af60000000p-3, 0x1.ecb6c00000000p-6, 0x1.74a9220000000p-6, -0x1.fecbf80000000p-5, -0x1.65f35a0000000p-4, -0x1.dd43be0000000p-4, 0x1.4901960000000p-3, -0x1.b272fa0000000p-4, 0x1.04ae020000000p-2, -0x1.3ded520000000p-4, 0x1.0f566c0000000p-4, 0x1.38a6a20000000p-4, -0x1.2763d60000000p-3, 0x1.8a664e0000000p-4, 0x1.fd09280000000p-4, -0x1.f48b200000000p-10, -0x1.8520d80000000p-4, 0x1.38ad400000000p-4, -0x1.88d5ea0000000p-4, 0x1.b1e5d60000000p-3, -0x1.03fe940000000p-4, 0x1.4f70040000000p-3, -0x1.c8dab40000000p-4, -0x1.1d2a360000000p-4, -0x1.0b84e20000000p-3, -0x1.e07f740000000p-6, -0x1.5189dc0000000p-4, -0x1.17ca940000000p-3, 0x1.e7634e0000000p-5, -0x1.7118700000000p-3, -0x1.268b0c0000000p-4, -0x1.0576720000000p-4, -0x1.5c774c0000000p-5, 0x1.2d913e0000000p-4, -0x1.7ed19e0000000p-7, -0x1.13bd640000000p-3, -0x1.28420e0000000p-4, 0x1.1dc24e0000000p-6, 0x1.6974620000000p-4, -0x1.2fa63e0000000p-4, 0x1.85041a0000000p-7, -0x1.cb35b40000000p-4, -0x1.b6c0980000000p-6, -0x1.8b9a160000000p-7, -0x1.0dde940000000p-3, -0x1.1647400000000p-3, -0x1.ed72b40000000p-6, -0x1.08115e0000000p-4, -0x1.2fa32a0000000p-3, 0x1.10c2f00000000p-4, 0x1.c0c6b60000000p-5, -0x1.1db5180000000p-3, -0x1.4214ac0000000p-4}
, {0x1.94e63a0000000p-4, -0x1.f831960000000p-4, 0x1.46a5180000000p-4, 0x1.67b04c0000000p-4, -0x1.0dca260000000p-4, -0x1.503e400000000p-4, 0x1.db72fe0000000p-5, 0x1.434b780000000p-5, 0x1.13681a0000000p-5, -0x1.3f83600000000p-4, -0x1.0510860000000p-3, 0x1.37f89c0000000p-7, 0x1.45452a0000000p-3, 0x1.ff84f40000000p-7, 0x1.cd00760000000p-7, 0x1.c914360000000p-6, -0x1.444b4c0000000p-3, -0x1.8812660000000p-5, 0x1.9b1d7c0000000p-8, 0x1.c17a060000000p-4, -0x1.adab620000000p-5, -0x1.4d547c0000000p-5, -0x1.6a1e6c0000000p-5, -0x1.aaaa3e0000000p-4, 0x1.3d2cd20000000p-4, 0x1.75af1e0000000p-3, -0x1.a3d6740000000p-7, 0x1.1bf2640000000p-6, 0x1.34d8060000000p-4, 0x1.c2acb20000000p-6, 0x1.8f94680000000p-4, -0x1.d0e3c80000000p-4, -0x1.a9a4d00000000p-7, -0x1.f4e5a00000000p-8, -0x1.b32fde0000000p-7, -0x1.b60e9a0000000p-6, -0x1.2ac1d60000000p-3, -0x1.1d98ea0000000p-3, -0x1.dd73440000000p-6, 0x1.7d8e980000000p-5, -0x1.5ba7ae0000000p-3, -0x1.9034140000000p-6, 0x1.c1f7a60000000p-4, 0x1.edf8c00000000p-5, 0x1.d4e2ae0000000p-6, 0x1.ad5a140000000p-4, -0x1.a1c7c80000000p-6, 0x1.109c020000000p-4, -0x1.be84b20000000p-4, 0x1.335da40000000p-10, 0x1.de87000000000p-5, -0x1.0f9a700000000p-6, -0x1.effe340000000p-9, -0x1.c304200000000p-3, 0x1.e4d2fe0000000p-8, -0x1.f4f2ae0000000p-4, -0x1.f3d9900000000p-5, 0x1.1e5c3e0000000p-3, -0x1.c59bac0000000p-4, 0x1.d442420000000p-4, -0x1.8a5e080000000p-3, -0x1.10d3ca0000000p-3, -0x1.482a200000000p-5, 0x1.1e561e0000000p-4}
}
, {{0x1.be5dce0000000p-5, 0x1.b2b8a80000000p-4, -0x1.aabd460000000p-5, -0x1.96883a0000000p-4, -0x1.0c96d00000000p-4, -0x1.7ac05a0000000p-5, 0x1.0c5fdc0000000p-7, 0x1.11e8b00000000p-4, -0x1.43cb140000000p-3, 0x1.4149ae0000000p-3, 0x1.0041400000000p-2, 0x1.5e792a0000000p-5, 0x1.224bca0000000p-3, -0x1.c172fc0000000p-3, -0x1.4baa160000000p-4, -0x1.b300bc0000000p-4, -0x1.8c6e860000000p-10, 0x1.4a80440000000p-4, 0x1.3523520000000p-4, -0x1.ae720e0000000p-5, -0x1.83745c0000000p-6, 0x1.beb2a40000000p-5, 0x1.a9a4440000000p-9, -0x1.3ea09e0000000p-3, -0x1.7f00a40000000p-3, 0x1.62b7d40000000p-4, -0x1.6e0f8c0000000p-6, -0x1.a582120000000p-8, 0x1.17a85c0000000p-3, -0x1.9492c00000000p-5, 0x1.33363a0000000p-2, -0x1.b567540000000p-4, -0x1.5a5aa20000000p-3, -0x1.a249ee0000000p-5, 0x1.f3178e0000000p-6, -0x1.3364460000000p-4, -0x1.54e4e20000000p-5, 0x1.143a5a0000000p-5, -0x1.f190780000000p-5, 0x1.035abe0000000p-4, 0x1.ad51300000000p-11, -0x1.7c09660000000p-5, -0x1.20cc880000000p-3, -0x1.59589a0000000p-6, -0x1.a708a00000000p-4, -0x1.0830560000000p-6, -0x1.b0af600000000p-3, -0x1.68844a0000000p-4, 0x1.0bd8340000000p-3, -0x1.6544740000000p-5, 0x1.3d96ca0000000p-4, 0x1.d256b80000000p-3, 0x1.7e8cc40000000p-4, -0x1.25c62a0000000p-4, -0x1.f12cce0000000p-4, -0x1.6748540000000p-2, 0x1.b867d60000000p-5, 0x1.68c6d20000000p-5, -0x1.4035fa0000000p-3, -0x1.c5c2320000000p-3, -0x1.77cf5e0000000p-6, -0x1.47613e0000000p-3, -0x1.d99e3a0000000p-4, -0x1.ed66ce0000000p-6}
, {-0x1.21a0da0000000p-5, -0x1.b555ca0000000p-5, 0x1.173fac0000000p-4, 0x1.71df940000000p-6, 0x1.842a8e0000000p-8, -0x1.0f63420000000p-5, 0x1.16e9660000000p-8, -0x1.dac0e00000000p-6, 0x1.1b19460000000p-3, 0x1.31d61e0000000p-3, 0x1.a331dc0000000p-6, 0x1.82dafc0000000p-3, -0x1.fa56740000000p-4, 0x1.bdc2080000000p-6, 0x1.e8c38c0000000p-4, -0x1.7be2180000000p-4, -0x1.d181c40000000p-4, 0x1.0577aa0000000p-4, -0x1.0685ce0000000p-3, -0x1.f8b4ee0000000p-5, -0x1.df45340000000p-4, -0x1.1f146c0000000p-3, 0x1.1ba6fc0000000p-3, -0x1.6487580000000p-4, 0x1.635b860000000p-5, 0x1.b689920000000p-6, 0x1.3077960000000p-4, 0x1.de19d40000000p-4, -0x1.a2cac80000000p-4, -0x1.855c600000000p-5, 0x1.ebcb620000000p-3, -0x1.0e07020000000p-3, -0x1.0123200000000p-8, -0x1.9bb18a0000000p-4, -0x1.33ba020000000p-4, -0x1.3acbdc0000000p-4, -0x1.c32e080000000p-5, -0x1.8a0f2e0000000p-4, -0x1.7c26fe0000000p-5, -0x1.1e281a0000000p-4, -0x1.0944880000000p-2, -0x1.39d7d20000000p-3, 0x1.4c39f20000000p-5, -0x1.880eec0000000p-5, 0x1.00b33a0000000p-6, 0x1.0011ea0000000p-4, -0x1.a0991c0000000p-5, 0x1.0b10320000000p-5, -0x1.ec276c0000000p-6, 0x1.e836580000000p-8, -0x1.f25e940000000p-8, -0x1.5ae8520000000p-5, 0x1.dc2bf00000000p-5, -0x1.a7e6c40000000p-4, 0x1.8c9e960000000p-9, -0x1.3b8e800000000p-4, -0x1.f7a5020000000p-4, 0x1.cdb8000000000p-4, -0x1.6a98260000000p-3, -0x1.b33fd20000000p-6, -0x1.af716e0000000p-7, -0x1.aefc000000000p-6, 0x1.a6e2460000000p-5, -0x1.549d960000000p-5}
, {0x1.b67c180000000p-3, -0x1.9ef0d20000000p-3, 0x1.0690780000000p-4, 0x1.68bc280000000p-3, 0x1.84f22a0000000p-4, 0x1.2b4a000000000p-7, 0x1.1e8baa0000000p-10, 0x1.2924dc0000000p-4, -0x1.d2bda40000000p-5, -0x1.5191c80000000p-6, -0x1.dc97820000000p-6, 0x1.b587ca0000000p-3, -0x1.068de60000000p-3, -0x1.fd45280000000p-5, 0x1.85a6420000000p-5, -0x1.2f8dd40000000p-9, 0x1.a572340000000p-9, 0x1.d241060000000p-5, -0x1.2b01da0000000p-2, 0x1.b4f67e0000000p-5, -0x1.522afc0000000p-3, -0x1.e2dd6e0000000p-3, 0x1.aec4ae0000000p-5, 0x1.cf75720000000p-6, -0x1.82da120000000p-7, -0x1.71376a0000000p-3, -0x1.d96cac0000000p-4, 0x1.8cf1d60000000p-6, -0x1.c727f20000000p-3, 0x1.4d04de0000000p-4, 0x1.7182360000000p-3, -0x1.153d8a0000000p-4, 0x1.acbd2c0000000p-3, 0x1.00e04a0000000p-4, -0x1.e13ec60000000p-4, -0x1.d38a380000000p-7, 0x1.102dca0000000p-4, 0x1.98da460000000p-4, -0x1.0574440000000p-4, -0x1.e45f120000000p-4, -0x1.39e9c20000000p-6, 0x1.dbd4100000000p-6, -0x1.3a8a4a0000000p-7, 0x1.a801500000000p-5, 0x1.af20300000000p-3, 0x1.06b5a20000000p-6, -0x1.0a19d40000000p-12, 0x1.e157840000000p-5, 0x1.95bcba0000000p-4, 0x1.6b98380000000p-4, -0x1.32122a0000000p-6, -0x1.d7557a0000000p-5, -0x1.7a68b60000000p-6, 0x1.1ba1120000000p-3, -0x1.d68a200000000p-3, 0x1.e6a50a0000000p-3, -0x1.243a300000000p-7, -0x1.bb32a00000000p-5, 0x1.4655fe0000000p-3, 0x1.ea20dc0000000p-3, 0x1.29529e0000000p-3, 0x1.2f97ce0000000p-4, 0x1.4d94a00000000p-4, -0x1.c68dc20000000p-4}
}
, {{-0x1.64eb680000000p-3, 0x1.e7b2860000000p-3, 0x1.b00c660000000p-4, -0x1.da84580000000p-4, -0x1.8c93dc0000000p-4, 0x1.59c7980000000p-7, 0x1.76d4080000000p-4, 0x1.4065600000000p-4, -0x1.c29bb00000000p-5, 0x1.6e52120000000p-3, 0x1.33598a0000000p-3, 0x1.6747b60000000p-4, -0x1.b979e20000000p-7, -0x1.85e5fc0000000p-5, 0x1.6cf74a0000000p-3, 0x1.dc5c7c0000000p-3, 0x1.33630e0000000p-4, 0x1.3fc2b40000000p-4, -0x1.d2804c0000000p-11, -0x1.8d510c0000000p-5, 0x1.6f0c100000000p-5, -0x1.20a7820000000p-4, -0x1.c59de40000000p-4, 0x1.feabd40000000p-5, -0x1.5e029c0000000p-2, 0x1.84edd80000000p-4, 0x1.6957620000000p-6, -0x1.c7f12c0000000p-6, -0x1.e5d0ac0000000p-4, 0x1.6962680000000p-4, 0x1.9a36d40000000p-3, 0x1.4354140000000p-4, 0x1.7ac81a0000000p-4, -0x1.158f320000000p-3, -0x1.9097aa0000000p-4, 0x1.1168280000000p-3, 0x1.9892880000000p-6, -0x1.0c44420000000p-3, -0x1.c19c9a0000000p-10, -0x1.bf6ede0000000p-6, -0x1.b708500000000p-6, -0x1.67804a0000000p-4, -0x1.aac19c0000000p-3, -0x1.864b7c0000000p-4, 0x1.3442800000000p-4, 0x1.6e03620000000p-5, -0x1.88fc440000000p-4, -0x1.11a5aa0000000p-2, -0x1.922c980000000p-7, -0x1.b34f4c0000000p-4, -0x1.ded3cc0000000p-6, -0x1.3c015a0000000p-6, -0x1.0f6a360000000p-5, 0x1.9fa7c40000000p-3, -0x1.c632c00000000p-3, -0x1.2216c00000000p-3, 0x1.41762e0000000p-4, 0x1.1312f00000000p-4, 0x1.0a1a100000000p-4, -0x1.7917e00000000p-5, 0x1.1062d00000000p-8, -0x1.eadf900000000p-5, -0x1.98a8000000000p-4, 0x1.3b95dc0000000p-4}
, {-0x1.59f9f80000000p-4, 0x1.eae6480000000p-5, 0x1.5b56360000000p-5, -0x1.45d7980000000p-4, -0x1.7d4aa00000000p-5, 0x1.d30fa40000000p-4, -0x1.cc30720000000p-5, 0x1.3755340000000p-4, -0x1.7575f60000000p-3, 0x1.6958300000000p-5, -0x1.4f2c6e0000000p-4, 0x1.14dc7c0000000p-7, 0x1.5b70720000000p-5, 0x1.de567c0000000p-6, 0x1.e8bc9c0000000p-10, 0x1.a66ef60000000p-3, 0x1.f4a0fa0000000p-5, 0x1.2d46a40000000p-4, 0x1.1de7440000000p-4, -0x1.12328c0000000p-4, -0x1.71605c0000000p-6, 0x1.d688000000000p-4, -0x1.343d780000000p-5, -0x1.56e76a0000000p-6, -0x1.c91f0a0000000p-3, -0x1.e8b4fe0000000p-5, 0x1.1850260000000p-4, -0x1.340f0a0000000p-4, -0x1.70b60e0000000p-10, -0x1.84f4bc0000000p-7, 0x1.7107c20000000p-5, 0x1.e65c2c0000000p-3, 0x1.6948a40000000p-3, 0x1.d30c960000000p-4, -0x1.7d4d4e0000000p-4, 0x1.5112e00000000p-4, 0x1.ae744e0000000p-5, 0x1.45825a0000000p-3, 0x1.a3c6e20000000p-4, 0x1.5ee98c0000000p-4, -0x1.b690a60000000p-3, -0x1.c755460000000p-5, -0x1.75d41c0000000p-5, -0x1.c22e440000000p-3, 0x1.9ad1420000000p-5, -0x1.3fe6300000000p-6, -0x1.4f40cc0000000p-5, -0x1.150f2e0000000p-4, 0x1.219b060000000p-4, -0x1.f623fe0000000p-4, 0x1.383c9a0000000p-4, 0x1.b1b43a0000000p-5, -0x1.d9b6e40000000p-6, 0x1.0c10f40000000p-3, -0x1.5800040000000p-3, 0x1.bb5a920000000p-4, 0x1.f499080000000p-4, -0x1.65013e0000000p-3, 0x1.3950300000000p-3, 0x1.9069f60000000p-5, 0x1.3b197c0000000p-6, 0x1.0953ca0000000p-5, 0x1.dac60a0000000p-8, -0x1.b353ee0000000p-4}
, {-0x1.94170c0000000p-3, 0x1.e12fac0000000p-5, -0x1.3f9d1c0000000p-3, -0x1.5df6ba0000000p-2, -0x1.0a1e9c0000000p-2, -0x1.f763a40000000p-8, -0x1.1fea360000000p-5, 0x1.0f95bc0000000p-7, -0x1.884e1e0000000p-5, -0x1.60de800000000p-3, -0x1.0410cc0000000p-2, -0x1.18cc980000000p-4, -0x1.86abee0000000p-5, 0x1.7afa040000000p-5, -0x1.f19ae40000000p-4, 0x1.3949bc0000000p-2, 0x1.62aba00000000p-6, -0x1.3451960000000p-4, 0x1.0b28c00000000p-2, -0x1.93fe9c0000000p-3, 0x1.c0643e0000000p-4, 0x1.85680c0000000p-3, -0x1.9d897c0000000p-3, -0x1.bc44d80000000p-3, 0x1.7b6e080000000p-4, -0x1.2957520000000p-3, -0x1.e401540000000p-9, -0x1.fb6ab60000000p-3, -0x1.357d2c0000000p-6, 0x1.52ce900000000p-4, -0x1.b9e17e0000000p-3, 0x1.30e66c0000000p-3, -0x1.8e85aa0000000p-6, -0x1.1363080000000p-4, 0x1.05fae40000000p-3, 0x1.e316180000000p-5, -0x1.0737d60000000p-2, 0x1.64f39a0000000p-2, -0x1.0952620000000p-2, 0x1.ae0b840000000p-3, -0x1.54aa940000000p-3, 0x1.634ff40000000p-6, -0x1.874ae20000000p-2, -0x1.24cab00000000p-3, -0x1.7033f20000000p-3, -0x1.e47b400000000p-4, 0x1.1efe2e0000000p-3, -0x1.3245ca0000000p-5, 0x1.5b666a0000000p-4, -0x1.9a13a80000000p-3, 0x1.09cd7e0000000p-5, -0x1.070a220000000p-3, -0x1.968f280000000p-7, -0x1.09dd3e0000000p-5, 0x1.3c453a0000000p-7, -0x1.bf596e0000000p-4, 0x1.be0a520000000p-4, -0x1.c1b7dc0000000p-8, 0x1.d821d00000000p-5, -0x1.ed77c20000000p-6, -0x1.e7eb820000000p-8, -0x1.bddf880000000p-3, -0x1.7575900000000p-3, -0x1.3cdfce0000000p-3}
}
, {{-0x1.89976a0000000p-4, -0x1.1fe2840000000p-5, 0x1.3a66cc0000000p-4, 0x1.e796500000000p-4, -0x1.08c2de0000000p-3, -0x1.5093760000000p-3, 0x1.4915960000000p-3, -0x1.80b0b20000000p-6, 0x1.579e9e0000000p-2, 0x1.192d440000000p-2, 0x1.c494680000000p-3, 0x1.106b1c0000000p-2, 0x1.75b32e0000000p-3, -0x1.dcc03e0000000p-7, -0x1.f924180000000p-6, 0x1.7715180000000p-4, 0x1.87ccfa0000000p-3, 0x1.b01b7c0000000p-9, -0x1.4d33800000000p-4, 0x1.5b79b60000000p-8, 0x1.ace9a20000000p-3, -0x1.3263660000000p-3, -0x1.c5245c0000000p-3, -0x1.b723be0000000p-5, -0x1.8056c00000000p-4, 0x1.7e32020000000p-6, 0x1.ed89640000000p-5, -0x1.fa03a40000000p-7, 0x1.4bdd320000000p-3, 0x1.fcc3fa0000000p-4, -0x1.4cc6e60000000p-4, 0x1.acc31c0000000p-4, -0x1.1215720000000p-4, -0x1.3c8e8c0000000p-4, -0x1.5732e60000000p-3, -0x1.ea5c5e0000000p-4, 0x1.c833640000000p-3, 0x1.13a7ca0000000p-7, -0x1.f9df260000000p-5, 0x1.2c66f80000000p-6, -0x1.03ddda0000000p-9, 0x1.3241240000000p-3, 0x1.db6fd60000000p-4, 0x1.d532620000000p-4, -0x1.2db6cc0000000p-3, 0x1.6e9a340000000p-3, 0x1.21ebca0000000p-4, 0x1.2bcd700000000p-3, -0x1.2fcaee0000000p-4, -0x1.7a08f20000000p-3, 0x1.4b54860000000p-5, 0x1.bc089c0000000p-6, -0x1.a711940000000p-5, 0x1.196c3e0000000p-3, 0x1.9935200000000p-5, 0x1.b3ba5c0000000p-3, -0x1.1138a40000000p-4, 0x1.29fd520000000p-2, -0x1.378e160000000p-4, -0x1.3ab48e0000000p-3, 0x1.d44fdc0000000p-5, 0x1.0377f80000000p-3, 0x1.0cb61a0000000p-4, -0x1.132b1e0000000p-7}
, {-0x1.4ee3a80000000p-4, 0x1.0e398c0000000p-4, 0x1.f48f780000000p-4, -0x1.22b0c60000000p-4, 0x1.af479e0000000p-5, 0x1.b5bd440000000p-3, -0x1.436a2e0000000p-4, 0x1.219c460000000p-5, 0x1.c6cfbe0000000p-6, 0x1.e8e2080000000p-3, 0x1.6cd6940000000p-6, 0x1.6659ca0000000p-3, 0x1.7dd0420000000p-5, 0x1.f2a67e0000000p-5, 0x1.4ad2720000000p-5, 0x1.7257720000000p-3, 0x1.67343a0000000p-7, 0x1.a68bb60000000p-7, -0x1.28e8e00000000p-7, -0x1.a2f1920000000p-5, -0x1.8f95760000000p-6, -0x1.8830080000000p-4, 0x1.2ba0e40000000p-6, 0x1.8e4de80000000p-3, -0x1.644d800000000p-4, -0x1.ee161c0000000p-5, 0x1.8d9d400000000p-5, -0x1.c30b1a0000000p-3, -0x1.f0e1d00000000p-4, -0x1.5f491e0000000p-5, -0x1.47a22a0000000p-3, 0x1.64d3360000000p-8, 0x1.ec5b0a0000000p-3, -0x1.15451e0000000p-6, -0x1.1c97c60000000p-3, -0x1.7f2a8a0000000p-6, 0x1.7369440000000p-3, 0x1.0d64ee0000000p-4, 0x1.21a4280000000p-6, -0x1.468f600000000p-5, 0x1.b6de580000000p-4, -0x1.ab83760000000p-8, -0x1.d3ffec0000000p-5, -0x1.3ac3440000000p-6, -0x1.81a2840000000p-5, 0x1.4156540000000p-3, 0x1.74bc760000000p-5, 0x1.f6842e0000000p-5, -0x1.ba8b4e0000000p-7, -0x1.37d0800000000p-3, 0x1.4706bc0000000p-5, 0x1.b5660a0000000p-9, -0x1.84ee4e0000000p-8, -0x1.401b080000000p-5, -0x1.91af360000000p-4, 0x1.9da67c0000000p-4, 0x1.b9ff280000000p-6, 0x1.6545020000000p-6, 0x1.c98fda0000000p-3, 0x1.07c7620000000p-2, -0x1.d2b9020000000p-4, 0x1.196a440000000p-3, 0x1.d4b92e0000000p-6, 0x1.3043dc0000000p-4}
, {-0x1.e5b4ac0000000p-3, 0x1.1e559a0000000p-2, 0x1.658fee0000000p-4, -0x1.771cea0000000p-4, -0x1.b997d20000000p-3, 0x1.2d2ee80000000p-2, -0x1.e0a5940000000p-4, 0x1.7e296a0000000p-4, 0x1.237bf00000000p-3, 0x1.73df380000000p-4, -0x1.65e3960000000p-5, 0x1.d31dfe0000000p-4, 0x1.714da20000000p-4, -0x1.a18ab40000000p-5, 0x1.ee6f020000000p-7, 0x1.e5f68a0000000p-5, 0x1.309a960000000p-7, -0x1.03fce20000000p-8, 0x1.104bd80000000p-3, -0x1.2ec97e0000000p-3, 0x1.794f720000000p-3, 0x1.ac3b940000000p-6, -0x1.0b254a0000000p-5, 0x1.3a30260000000p-3, -0x1.4db1bc0000000p-8, -0x1.acfeb40000000p-10, 0x1.e4fa300000000p-4, -0x1.a0e3060000000p-5, 0x1.9901fa0000000p-5, 0x1.8e1dde0000000p-4, -0x1.e85c640000000p-3, 0x1.5000140000000p-4, -0x1.501a100000000p-4, 0x1.1a5dde0000000p-3, 0x1.223ab40000000p-3, 0x1.1126900000000p-4, -0x1.7245c20000000p-6, 0x1.3161880000000p-3, 0x1.8dbd780000000p-9, 0x1.c01dea0000000p-4, 0x1.98e9900000000p-5, -0x1.d452ae0000000p-4, 0x1.051bf60000000p-5, -0x1.109ab80000000p-3, 0x1.1b7de80000000p-6, 0x1.1176e00000000p-5, -0x1.545dbc0000000p-5, 0x1.48edde0000000p-3, 0x1.2834d40000000p-4, -0x1.c3ca480000000p-3, 0x1.86d6380000000p-6, 0x1.7df8840000000p-6, 0x1.9bc8680000000p-4, -0x1.6d7f920000000p-5, 0x1.0a23b40000000p-2, -0x1.4c4a040000000p-3, 0x1.6748ba0000000p-6, 0x1.a3f9020000000p-3, 0x1.7a2fa40000000p-6, -0x1.4f983e0000000p-4, -0x1.7569b20000000p-3, -0x1.1829fc0000000p-3, -0x1.1d27e00000000p-4, -0x1.bcef8c0000000p-6}
}
, {{0x1.caf3360000000p-3, 0x1.5e223e0000000p-4, 0x1.125c280000000p-6, -0x1.d1ac100000000p-5, -0x1.7f2e3c0000000p-4, -0x1.e04ba00000000p-4, -0x1.99e33c0000000p-4, 0x1.9a65be0000000p-5, -0x1.0dbb5e0000000p-4, 0x1.2da4580000000p-3, 0x1.a55cc00000000p-4, 0x1.913eba0000000p-4, -0x1.42efc40000000p-4, -0x1.43ef740000000p-5, 0x1.2c12900000000p-4, -0x1.23aff80000000p-3, -0x1.9d7af60000000p-4, 0x1.0e09c60000000p-9, 0x1.5f07b80000000p-5, 0x1.3fe7620000000p-3, -0x1.5d96860000000p-3, 0x1.a3588c0000000p-5, 0x1.42b34a0000000p-6, 0x1.a505440000000p-3, -0x1.e4a1dc0000000p-5, 0x1.81c3ba0000000p-4, 0x1.39b9800000000p-5, 0x1.bbb4ea0000000p-6, 0x1.0fcf9e0000000p-6, -0x1.6f2f8e0000000p-7, 0x1.1a0f1c0000000p-4, -0x1.bf5cd60000000p-5, -0x1.19424a0000000p-4, 0x1.e584420000000p-6, -0x1.21b6ae0000000p-8, 0x1.89de340000000p-4, 0x1.9d37c40000000p-4, 0x1.1888080000000p-4, -0x1.bf377a0000000p-5, 0x1.8052860000000p-3, 0x1.492df40000000p-4, -0x1.166f180000000p-2, -0x1.d83e220000000p-6, -0x1.5d28bc0000000p-3, 0x1.d1ea800000000p-7, -0x1.3044b40000000p-2, -0x1.f29aa40000000p-5, 0x1.38adb00000000p-4, -0x1.84ccb40000000p-5, 0x1.3e40120000000p-4, 0x1.554ec80000000p-3, 0x1.1c98ec0000000p-5, -0x1.e6a3b20000000p-4, -0x1.7e18280000000p-3, -0x1.b8cd820000000p-5, -0x1.86a7620000000p-7, 0x1.0b125a0000000p-6, -0x1.bfaeea0000000p-5, 0x1.07fffc0000000p-2, -0x1.ec6c4e0000000p-10, 0x1.72b9f00000000p-4, 0x1.390c680000000p-2, 0x1.4d51540000000p-3, 0x1.2f7f9c0000000p-5}
, {0x1.fb6ce40000000p-4, -0x1.1d0c600000000p-6, 0x1.6e2f7a0000000p-4, -0x1.2b9f5e0000000p-3, 0x1.0118bc0000000p-4, 0x1.fc9ec20000000p-6, 0x1.807bee0000000p-6, 0x1.1682540000000p-4, 0x1.3079200000000p-5, -0x1.da0b440000000p-5, 0x1.fd31780000000p-4, -0x1.20a2160000000p-3, 0x1.a3916a0000000p-6, -0x1.add89c0000000p-4, -0x1.00a4460000000p-3, -0x1.bd658c0000000p-4, -0x1.4c4e6c0000000p-5, 0x1.38ef6e0000000p-6, 0x1.d86a300000000p-7, 0x1.18fa8c0000000p-8, -0x1.22b16e0000000p-5, 0x1.6bc2200000000p-4, 0x1.7b54ea0000000p-9, 0x1.5485ee0000000p-4, 0x1.badb560000000p-4, -0x1.7851320000000p-5, -0x1.76c0a80000000p-6, 0x1.89db780000000p-5, -0x1.64a7d00000000p-3, 0x1.12c8540000000p-3, 0x1.ed2be80000000p-6, -0x1.9b98ea0000000p-4, -0x1.9032200000000p-4, -0x1.59635a0000000p-4, 0x1.2c1fdc0000000p-5, 0x1.24696c0000000p-4, -0x1.6d44120000000p-3, -0x1.1fab240000000p-4, -0x1.b60e400000000p-9, -0x1.5400960000000p-8, 0x1.1410aa0000000p-5, -0x1.5501400000000p-5, -0x1.22ef640000000p-4, -0x1.de3e060000000p-6, -0x1.615f280000000p-3, -0x1.0fdc1e0000000p-4, -0x1.9f5a720000000p-5, -0x1.41cc4a0000000p-4, 0x1.d784760000000p-6, 0x1.e8551a0000000p-3, -0x1.11c1fa0000000p-3, 0x1.76c3680000000p-7, 0x1.bc4cb20000000p-4, -0x1.6824cc0000000p-4, -0x1.d152440000000p-4, 0x1.46dd6c0000000p-4, -0x1.0425380000000p-4, 0x1.99435e0000000p-5, 0x1.52fd0c0000000p-4, -0x1.9088760000000p-2, 0x1.aabdea0000000p-4, 0x1.c11cf40000000p-7, 0x1.145c940000000p-4, 0x1.e3ca140000000p-5}
, {0x1.83af620000000p-5, -0x1.30cbcc0000000p-3, 0x1.83f2a80000000p-3, 0x1.595d0c0000000p-3, 0x1.ad5dae0000000p-3, 0x1.997d640000000p-5, 0x1.11c0520000000p-6, 0x1.300b060000000p-3, 0x1.ea4b5a0000000p-5, 0x1.e0a3280000000p-5, 0x1.05473a0000000p-3, 0x1.b15db20000000p-7, -0x1.1adf960000000p-5, 0x1.d539ae0000000p-3, 0x1.c814e80000000p-7, -0x1.1280880000000p-2, 0x1.4c913e0000000p-7, 0x1.b314840000000p-6, -0x1.cccbd60000000p-7, 0x1.5df34a0000000p-3, -0x1.db06d80000000p-4, -0x1.30e40e0000000p-2, 0x1.2a8d5c0000000p-4, -0x1.a930580000000p-4, -0x1.6288da0000000p-6, 0x1.68ff760000000p-3, -0x1.eb1c0a0000000p-7, 0x1.0435580000000p-4, -0x1.b3aab60000000p-6, -0x1.f2bca00000000p-6, 0x1.f043ce0000000p-4, -0x1.f245a00000000p-3, 0x1.21f11a0000000p-4, 0x1.27f2ca0000000p-4, -0x1.26cc020000000p-4, -0x1.76d8540000000p-4, -0x1.2330a00000000p-4, -0x1.6f2dae0000000p-4, 0x1.1760580000000p-2, -0x1.3ea63a0000000p-4, 0x1.5c57f40000000p-4, -0x1.b04e8c0000000p-3, 0x1.11f0600000000p-2, -0x1.ba4e640000000p-9, 0x1.77fba00000000p-3, 0x1.9bb5ae0000000p-8, -0x1.3c0b280000000p-3, -0x1.330ac20000000p-5, -0x1.2ec0a20000000p-6, 0x1.573a1e0000000p-2, -0x1.6e32f60000000p-9, 0x1.0264c60000000p-2, -0x1.0956860000000p-3, 0x1.7fe3e20000000p-4, -0x1.6469160000000p-4, 0x1.12644a0000000p-2, -0x1.ace5760000000p-4, 0x1.9340f00000000p-4, 0x1.5ca82a0000000p-4, -0x1.0403da0000000p-4, 0x1.61d4c80000000p-3, 0x1.81b80a0000000p-2, 0x1.bdf04e0000000p-6, 0x1.73b0de0000000p-3}
}
, {{0x1.4113480000000p-4, 0x1.ebb70e0000000p-8, -0x1.04d5680000000p-5, -0x1.a732200000000p-4, 0x1.0dfa340000000p-4, 0x1.2364520000000p-3, -0x1.761b060000000p-4, 0x1.8dd2900000000p-4, -0x1.fd9cc60000000p-3, -0x1.18eef00000000p-4, -0x1.4fbd940000000p-6, -0x1.b14d4c0000000p-4, -0x1.e649d20000000p-4, 0x1.6f273e0000000p-3, -0x1.0d1f6a0000000p-5, -0x1.ec0db20000000p-3, 0x1.0b43a40000000p-5, -0x1.edb3860000000p-6, 0x1.7000380000000p-4, 0x1.2a0a500000000p-6, 0x1.84aa640000000p-5, -0x1.3a60fc0000000p-5, -0x1.62b1ba0000000p-5, 0x1.c2f9fe0000000p-4, -0x1.2039920000000p-2, 0x1.d808340000000p-5, -0x1.2b579e0000000p-9, 0x1.88e03c0000000p-4, -0x1.8b573a0000000p-3, -0x1.1b56a00000000p-5, 0x1.21326a0000000p-3, 0x1.7d05ce0000000p-3, -0x1.44f63e0000000p-3, -0x1.2935a80000000p-4, -0x1.5c688e0000000p-4, -0x1.acfe120000000p-5, 0x1.15fa0a0000000p-3, -0x1.beb5740000000p-5, 0x1.2a60b60000000p-2, 0x1.e7e5680000000p-5, 0x1.9d887e0000000p-3, 0x1.5ba2780000000p-6, -0x1.5274ea0000000p-3, 0x1.cedae20000000p-4, 0x1.235a600000000p-3, -0x1.5b3f2c0000000p-3, -0x1.d0aa240000000p-5, -0x1.99be680000000p-5, -0x1.ac4a060000000p-4, 0x1.c0a09c0000000p-6, -0x1.d0a4f00000000p-5, 0x1.ee33e40000000p-6, -0x1.52c8860000000p-3, -0x1.e52a600000000p-9, -0x1.4df6400000000p-4, 0x1.54ff240000000p-6, -0x1.52f6020000000p-5, -0x1.b1bf9e0000000p-6, -0x1.7b09680000000p-9, -0x1.556dfa0000000p-3, -0x1.586d460000000p-4, -0x1.68e24c0000000p-3, 0x1.859a520000000p-7, -0x1.3680ae0000000p-3}
, {0x1.0d8e880000000p-2, -0x1.10f8940000000p-4, 0x1.a85d2e0000000p-7, 0x1.48f6ec0000000p-5, 0x1.a99c500000000p-5, 0x1.a6a0620000000p-4, -0x1.e77c9a0000000p-4, 0x1.8386aa0000000p-3, 0x1.a6b11c0000000p-3, -0x1.404c8a0000000p-4, -0x1.546b8c0000000p-6, -0x1.fc671a0000000p-5, -0x1.3fb7ca0000000p-4, -0x1.3c2a060000000p-3, 0x1.fb05480000000p-3, -0x1.da54840000000p-6, 0x1.18b0640000000p-4, -0x1.99401e0000000p-5, -0x1.96b1680000000p-4, -0x1.3020180000000p-6, 0x1.064a980000000p-3, -0x1.2ebb500000000p-5, 0x1.a4bfc60000000p-5, -0x1.215e260000000p-4, -0x1.44bd640000000p-6, -0x1.5632620000000p-5, 0x1.3715d00000000p-6, -0x1.f73e860000000p-7, -0x1.cf5b920000000p-4, -0x1.0258ec0000000p-5, 0x1.57ec8c0000000p-4, 0x1.4fa6cc0000000p-6, -0x1.15def00000000p-3, 0x1.21018e0000000p-4, 0x1.3697c80000000p-7, -0x1.585ba80000000p-4, -0x1.1896300000000p-4, 0x1.7478400000000p-4, -0x1.75d5660000000p-5, 0x1.4c0b940000000p-3, 0x1.6bbdee0000000p-5, 0x1.6664e00000000p-6, -0x1.c6c0f40000000p-5, -0x1.084d880000000p-4, 0x1.453b880000000p-3, 0x1.7ec8960000000p-4, -0x1.46463e0000000p-4, 0x1.1f1e7c0000000p-3, 0x1.559f100000000p-6, 0x1.2a3c320000000p-4, -0x1.0baf1c0000000p-5, -0x1.f7e00e0000000p-9, 0x1.49c25c0000000p-5, -0x1.5a63b20000000p-4, -0x1.269f6a0000000p-3, 0x1.df37540000000p-5, 0x1.e3b0780000000p-5, 0x1.8221ce0000000p-3, -0x1.0014020000000p-5, -0x1.25a1bc0000000p-3, 0x1.96f9620000000p-4, -0x1.37c7ba0000000p-5, -0x1.192adc0000000p-3, -0x1.1779f80000000p-3}
, {0x1.666f2e0000000p-9, -0x1.f42e160000000p-8, -0x1.76d16c0000000p-3, -0x1.2447880000000p-2, -0x1.775c440000000p-4, 0x1.e7c2f80000000p-5, -0x1.2c54700000000p-4, 0x1.c835d60000000p-5, -0x1.533c680000000p-3, -0x1.800b500000000p-4, -0x1.35a0320000000p-4, -0x1.38d6920000000p-3, -0x1.65313a0000000p-10, 0x1.d46b5e0000000p-7, 0x1.0e6d9c0000000p-2, -0x1.79ff720000000p-6, 0x1.1683920000000p-4, 0x1.df82ba0000000p-4, -0x1.060f4e0000000p-3, 0x1.d392b40000000p-5, -0x1.0e73780000000p-4, 0x1.bb907e0000000p-6, -0x1.079c960000000p-3, -0x1.0c1b400000000p-4, -0x1.7b4fd00000000p-5, 0x1.ec62500000000p-3, -0x1.5169540000000p-3, -0x1.8982740000000p-3, 0x1.8344a80000000p-4, -0x1.329d320000000p-3, 0x1.d0e3ca0000000p-4, 0x1.4d59b60000000p-6, -0x1.8a7a7a0000000p-7, -0x1.de91fc0000000p-4, -0x1.024d440000000p-2, 0x1.b42f4a0000000p-3, 0x1.111b120000000p-3, 0x1.869a0a0000000p-8, -0x1.9d5c960000000p-3, 0x1.a7ab040000000p-3, 0x1.b1f3540000000p-4, -0x1.c45b4a0000000p-4, -0x1.a8a41e0000000p-5, 0x1.02acaa0000000p-5, -0x1.02a59a0000000p-2, -0x1.0a63160000000p-4, 0x1.378d6e0000000p-6, -0x1.7e19ca0000000p-4, 0x1.ab28620000000p-4, -0x1.fcdf540000000p-6, -0x1.f610660000000p-4, -0x1.34222c0000000p-4, -0x1.9c21320000000p-3, 0x1.10a9520000000p-3, -0x1.6932de0000000p-3, 0x1.13a93a0000000p-2, -0x1.09af540000000p-5, -0x1.55b64e0000000p-4, -0x1.b471f20000000p-4, -0x1.db448a0000000p-3, 0x1.424db80000000p-5, -0x1.00e25c0000000p-3, -0x1.71d0da0000000p-3, -0x1.9d63900000000p-4}
}
, {{-0x1.542b720000000p-4, -0x1.d9aa440000000p-3, 0x1.29be3a0000000p-4, 0x1.6f24320000000p-4, -0x1.2d04460000000p-4, -0x1.1da35e0000000p-3, 0x1.4055d60000000p-6, -0x1.e194600000000p-5, 0x1.6a36f00000000p-4, 0x1.4e0a3a0000000p-4, -0x1.63297e0000000p-4, 0x1.c9e0be0000000p-6, 0x1.967cd60000000p-5, 0x1.d261fc0000000p-2, -0x1.92e2a40000000p-4, 0x1.a1b9040000000p-7, -0x1.03e0880000000p-7, 0x1.2d5d460000000p-5, 0x1.1fb7fc0000000p-3, -0x1.ffc4de0000000p-6, -0x1.28d2040000000p-7, -0x1.309bc60000000p-4, 0x1.154fde0000000p-2, -0x1.7d08280000000p-3, 0x1.1a38fe0000000p-4, -0x1.a6cfbc0000000p-6, 0x1.d18e560000000p-5, -0x1.d7e03c0000000p-4, -0x1.43d1fc0000000p-3, -0x1.f6d7120000000p-4, 0x1.d6f2100000000p-5, -0x1.fb17fc0000000p-4, 0x1.800d6c0000000p-2, -0x1.96efde0000000p-3, -0x1.99142c0000000p-3, -0x1.c501640000000p-5, -0x1.2ad9140000000p-6, -0x1.1f8caa0000000p-4, 0x1.aee2c80000000p-4, -0x1.da97040000000p-3, 0x1.947b9c0000000p-5, 0x1.9d0b980000000p-5, 0x1.f8cdba0000000p-7, 0x1.1934d40000000p-4, -0x1.07014e0000000p-4, -0x1.2a6a940000000p-7, -0x1.0870a40000000p-5, 0x1.6a82c80000000p-7, 0x1.7f35880000000p-5, 0x1.75c4380000000p-4, 0x1.f838b20000000p-6, -0x1.8b91de0000000p-6, -0x1.67671e0000000p-4, 0x1.0cbb5e0000000p-4, 0x1.c189b40000000p-5, -0x1.b87d5e0000000p-5, -0x1.8447300000000p-4, 0x1.a9d55a0000000p-8, -0x1.6eca900000000p-4, 0x1.d604b80000000p-3, -0x1.b6ef400000000p-3, -0x1.3ed0160000000p-4, -0x1.2a59660000000p-3, 0x1.854e920000000p-3}
, {0x1.1f7f080000000p-4, -0x1.5807f60000000p-3, 0x1.9fc4f60000000p-4, -0x1.1b00f40000000p-3, -0x1.ac50280000000p-5, -0x1.500eac0000000p-4, 0x1.741f880000000p-5, 0x1.b3616c0000000p-4, 0x1.5cfadc0000000p-3, -0x1.c211160000000p-4, 0x1.c705440000000p-4, -0x1.ef75180000000p-6, -0x1.e5664a0000000p-5, -0x1.2d34300000000p-3, 0x1.3c02080000000p-5, -0x1.4cd3aa0000000p-4, -0x1.bd95b40000000p-6, -0x1.24d8aa0000000p-6, -0x1.8621b00000000p-3, 0x1.e9a1ee0000000p-4, -0x1.6503a20000000p-5, -0x1.1b9b080000000p-3, -0x1.28f52e0000000p-5, -0x1.af830a0000000p-4, -0x1.16d7700000000p-5, 0x1.1dfcd40000000p-5, 0x1.6bbd4c0000000p-14, 0x1.9ad4bc0000000p-5, 0x1.8f21e60000000p-9, -0x1.6b9b680000000p-5, -0x1.03a0b00000000p-2, -0x1.1996940000000p-3, 0x1.7a89a80000000p-6, 0x1.3bfc1a0000000p-4, 0x1.76b6880000000p-5, 0x1.ce23b00000000p-6, 0x1.a0ecb60000000p-4, -0x1.a838840000000p-4, -0x1.4074520000000p-3, 0x1.9a14340000000p-6, 0x1.a93df60000000p-4, -0x1.47e9f40000000p-7, 0x1.c450da0000000p-3, -0x1.9050d40000000p-6, -0x1.2b0a700000000p-7, 0x1.98377e0000000p-7, 0x1.09b2640000000p-6, -0x1.10b5f00000000p-4, -0x1.3cc3460000000p-4, -0x1.45cd020000000p-6, -0x1.9b0db00000000p-3, 0x1.feed720000000p-6, 0x1.03bd120000000p-3, -0x1.444ad60000000p-5, -0x1.159e280000000p-5, 0x1.a855560000000p-5, -0x1.9823600000000p-4, 0x1.5334c80000000p-5, -0x1.25c0000000000p-4, 0x1.5f79820000000p-5, 0x1.a4a54e0000000p-5, 0x1.c4a2ca0000000p-5, 0x1.1706260000000p-3, -0x1.90f2240000000p-4}
, {0x1.b4881a0000000p-4, 0x1.c9b17e0000000p-7, 0x1.d8e1aa0000000p-5, 0x1.19c1d00000000p-3, -0x1.97b1b80000000p-3, -0x1.a210ea0000000p-4, 0x1.36eb180000000p-5, 0x1.00655c0000000p-2, 0x1.d2e5700000000p-3, -0x1.2c46320000000p-3, 0x1.66c98a0000000p-3, -0x1.b9c6480000000p-4, -0x1.d051aa0000000p-8, -0x1.49e1d00000000p-4, -0x1.eb5d9a0000000p-6, 0x1.2b40980000000p-5, -0x1.d1cd440000000p-3, -0x1.edddf00000000p-4, 0x1.5cb48a0000000p-7, 0x1.74b79c0000000p-3, -0x1.ef6bd40000000p-5, -0x1.71d9600000000p-5, 0x1.35c92a0000000p-3, 0x1.a71dfa0000000p-4, -0x1.c01bfe0000000p-3, 0x1.9c3c420000000p-3, -0x1.c4a5c60000000p-5, 0x1.3650d20000000p-3, -0x1.2990100000000p-2, -0x1.a9fe320000000p-3, 0x1.751ad00000000p-4, -0x1.71fa1c0000000p-5, 0x1.61ee9e0000000p-3, 0x1.acb6940000000p-7, 0x1.51f9c80000000p-4, 0x1.0d16cc0000000p-4, -0x1.4ebc300000000p-12, -0x1.ecc0280000000p-5, 0x1.34bbb00000000p-8, -0x1.3b24a20000000p-5, 0x1.7edd140000000p-3, -0x1.f62f040000000p-5, 0x1.f198de0000000p-8, -0x1.4c2f5e0000000p-3, -0x1.11b14c0000000p-5, 0x1.1f2f9e0000000p-7, -0x1.a448ec0000000p-3, 0x1.969cec0000000p-5, -0x1.2142780000000p-2, 0x1.4f71260000000p-3, 0x1.5c9abe0000000p-6, -0x1.bf94ec0000000p-4, -0x1.1b2a100000000p-6, 0x1.23222c0000000p-3, -0x1.7204160000000p-3, -0x1.f4d8260000000p-3, -0x1.f5928e0000000p-6, 0x1.f0c7520000000p-3, -0x1.0bf5dc0000000p-3, -0x1.fdb83a0000000p-4, -0x1.034f180000000p-3, 0x1.a43e1e0000000p-9, -0x1.f5ae740000000p-6, -0x1.c31b9c0000000p-6}
}
, {{-0x1.a5c98a0000000p-6, -0x1.44e7780000000p-3, -0x1.e1f07c0000000p-4, -0x1.a8bfc80000000p-6, 0x1.77caa20000000p-6, -0x1.d1cc7e0000000p-3, -0x1.93d6f80000000p-4, -0x1.e564a20000000p-3, 0x1.74e76c0000000p-5, -0x1.2c92680000000p-6, -0x1.8764640000000p-4, 0x1.d8ea320000000p-5, -0x1.ce3b8a0000000p-5, 0x1.4e92be0000000p-4, 0x1.1b63ee0000000p-3, 0x1.e4a5640000000p-7, -0x1.9654f20000000p-4, 0x1.1d08880000000p-2, -0x1.330ca20000000p-4, -0x1.4012ae0000000p-3, -0x1.9c66ee0000000p-4, -0x1.2d81100000000p-3, 0x1.91479c0000000p-3, -0x1.ec25b80000000p-8, -0x1.dac7280000000p-13, -0x1.018a640000000p-3, 0x1.3a8e640000000p-7, -0x1.3f05f80000000p-3, 0x1.9cefcc0000000p-6, -0x1.34e8c00000000p-6, 0x1.ce54440000000p-4, -0x1.93bc8e0000000p-3, 0x1.fdf0f00000000p-3, 0x1.3f782c0000000p-4, -0x1.51fe2c0000000p-3, -0x1.2ba1720000000p-3, 0x1.6b89b20000000p-5, 0x1.00dd620000000p-8, 0x1.1f6b620000000p-6, -0x1.c655b60000000p-5, -0x1.fd26920000000p-8, 0x1.6652520000000p-8, -0x1.846d1c0000000p-3, -0x1.b2ee840000000p-4, 0x1.a1a5f60000000p-5, -0x1.15a2d60000000p-3, -0x1.9b48f60000000p-3, 0x1.a497e80000000p-4, -0x1.b18fa00000000p-4, 0x1.5151780000000p-7, -0x1.5921300000000p-9, -0x1.2a45620000000p-4, -0x1.f7028e0000000p-3, -0x1.040a5a0000000p-4, 0x1.27defa0000000p-4, -0x1.3a3e640000000p-4, -0x1.0b9ac80000000p-7, 0x1.4e26580000000p-6, -0x1.45f5bc0000000p-3, -0x1.90d9f20000000p-4, -0x1.14c72a0000000p-4, -0x1.f8e74e0000000p-3, -0x1.4f038e0000000p-2, -0x1.a767020000000p-6}
, {0x1.9dbd480000000p-5, -0x1.6165260000000p-4, -0x1.4298480000000p-6, -0x1.6145880000000p-5, 0x1.d831f80000000p-4, 0x1.d1f7340000000p-5, 0x1.93b7620000000p-6, -0x1.34e8780000000p-3, 0x1.9a0d320000000p-5, -0x1.ea005a0000000p-5, -0x1.27b50a0000000p-5, 0x1.ae35400000000p-4, 0x1.f7d34c0000000p-6, -0x1.3cce300000000p-5, 0x1.4eea200000000p-5, 0x1.6dfe1c0000000p-3, -0x1.3f685c0000000p-8, 0x1.36d9a60000000p-4, 0x1.7520d20000000p-5, 0x1.a07c8e0000000p-4, 0x1.1876300000000p-3, 0x1.819de00000000p-6, 0x1.ed28a00000000p-6, 0x1.727d820000000p-5, -0x1.80a4dc0000000p-4, 0x1.e71d3a0000000p-3, -0x1.573dd40000000p-4, -0x1.3dede20000000p-3, -0x1.6953a20000000p-6, -0x1.4628b60000000p-5, 0x1.22017e0000000p-3, 0x1.ca603c0000000p-6, -0x1.1422060000000p-4, 0x1.b023920000000p-7, -0x1.f5accc0000000p-5, 0x1.9672560000000p-7, 0x1.03542c0000000p-3, 0x1.de4ec60000000p-10, 0x1.8d9e8c0000000p-3, -0x1.20e16c0000000p-5, 0x1.05f7e80000000p-3, -0x1.592f000000000p-5, 0x1.122d6a0000000p-9, -0x1.0d39360000000p-6, 0x1.e2db520000000p-5, -0x1.5d217a0000000p-8, 0x1.07d4fa0000000p-3, -0x1.16d5800000000p-3, -0x1.0d051e0000000p-6, 0x1.31826e0000000p-5, -0x1.1f44e80000000p-4, -0x1.02cc120000000p-5, -0x1.02d89a0000000p-3, 0x1.5aacd80000000p-3, -0x1.be76ac0000000p-6, 0x1.5660520000000p-6, -0x1.16f17e0000000p-3, -0x1.9c0fa80000000p-6, -0x1.099bbc0000000p-6, -0x1.caf9300000000p-8, -0x1.60622e0000000p-4, 0x1.39ef840000000p-4, -0x1.04a9600000000p-2, 0x1.58559c0000000p-6}
, {0x1.51bf160000000p-5, -0x1.5c047a0000000p-6, -0x1.2f71fa0000000p-3, -0x1.fb1cc20000000p-4, 0x1.88a1e00000000p-6, 0x1.5760180000000p-3, -0x1.4df1c20000000p-4, -0x1.cbe0d40000000p-5, 0x1.a4808e0000000p-4, -0x1.5480600000000p-8, -0x1.f9e5240000000p-6, 0x1.36f6520000000p-4, -0x1.69edc60000000p-5, -0x1.ad2a5e0000000p-7, -0x1.d0c5ca0000000p-5, 0x1.42e8580000000p-4, -0x1.d6fe260000000p-3, 0x1.7f87480000000p-4, -0x1.dc80ae0000000p-5, 0x1.0d5ade0000000p-7, 0x1.a0db060000000p-5, 0x1.fae5c00000000p-7, -0x1.d2697a0000000p-4, 0x1.80afc00000000p-7, 0x1.55f4f60000000p-4, 0x1.0808f60000000p-3, -0x1.13c5780000000p-3, -0x1.a75a260000000p-3, 0x1.21ca460000000p-6, 0x1.b5f9420000000p-8, 0x1.2f96000000000p-4, -0x1.37c6180000000p-5, 0x1.46064e0000000p-3, 0x1.d98d6e0000000p-6, 0x1.e5fb7e0000000p-4, 0x1.b9a8f00000000p-6, 0x1.4757b80000000p-4, -0x1.18fc600000000p-5, -0x1.1891ec0000000p-3, 0x1.e35ae00000000p-7, 0x1.2c0c400000000p-2, -0x1.8d7a5a0000000p-4, -0x1.6850560000000p-3, -0x1.ab61200000000p-3, 0x1.2304ec0000000p-4, 0x1.1d76680000000p-4, -0x1.f10e600000000p-5, -0x1.7dda800000000p-4, 0x1.b20d4a0000000p-5, 0x1.a1c0d80000000p-6, 0x1.430ae80000000p-6, -0x1.185fd00000000p-5, -0x1.dc1b780000000p-5, 0x1.594b520000000p-4, -0x1.54a3820000000p-5, 0x1.fb43880000000p-4, 0x1.3289e40000000p-6, 0x1.15ac9e0000000p-5, 0x1.0ffeee0000000p-7, -0x1.a390920000000p-4, 0x1.d3ea1a0000000p-5, 0x1.9ad9220000000p-5, -0x1.146e220000000p-2, 0x1.16855a0000000p-4}
}
, {{-0x1.81be0c0000000p-6, -0x1.36cdb40000000p-3, 0x1.adc37c0000000p-4, 0x1.1df64e0000000p-2, 0x1.07a3ca0000000p-4, -0x1.da80660000000p-4, 0x1.11ab240000000p-4, -0x1.6e0dbe0000000p-4, 0x1.ff0cb20000000p-4, 0x1.b815140000000p-4, -0x1.0123120000000p-3, 0x1.5ae4d20000000p-4, -0x1.cac9c00000000p-4, 0x1.ec48880000000p-3, -0x1.6cbf7a0000000p-8, -0x1.c9b2120000000p-4, -0x1.6faee20000000p-7, -0x1.69e0240000000p-4, -0x1.542e0e0000000p-6, -0x1.cdc27a0000000p-4, -0x1.82ebf80000000p-3, -0x1.375f7a0000000p-2, 0x1.776f8e0000000p-3, 0x1.7b48de0000000p-6, 0x1.621d740000000p-2, -0x1.23a5100000000p-3, -0x1.1903a00000000p-3, 0x1.09c1920000000p-3, -0x1.4a9c740000000p-2, 0x1.cd05e00000000p-9, -0x1.06e7620000000p-5, -0x1.531eb20000000p-3, 0x1.8db7220000000p-3, -0x1.379fbe0000000p-3, -0x1.389d120000000p-2, -0x1.2ccb040000000p-2, 0x1.391b020000000p-5, -0x1.0748140000000p-4, 0x1.d57e7c0000000p-3, -0x1.556bae0000000p-3, -0x1.cd987e0000000p-3, 0x1.09f2420000000p-4, 0x1.29c1c60000000p-3, 0x1.9e26260000000p-8, 0x1.89cd480000000p-3, 0x1.7337500000000p-3, -0x1.8ea0480000000p-4, 0x1.3d326a0000000p-3, 0x1.7de6280000000p-5, -0x1.0129760000000p-3, 0x1.0811840000000p-7, -0x1.062f520000000p-3, -0x1.0f84200000000p-3, 0x1.cb840a0000000p-6, 0x1.891d940000000p-4, -0x1.9987720000000p-3, -0x1.9705e20000000p-3, -0x1.ddd7940000000p-5, 0x1.6c63760000000p-7, 0x1.6b56de0000000p-2, -0x1.eea4ae0000000p-3, -0x1.9b08a40000000p-4, 0x1.5c2a5a0000000p-3, 0x1.740cbc0000000p-4}
, {-0x1.7f7dea0000000p-6, 0x1.387fba0000000p-4, -0x1.3d45400000000p-3, -0x1.384fb20000000p-5, 0x1.2115e40000000p-4, -0x1.1300640000000p-4, -0x1.78011a0000000p-9, 0x1.233c180000000p-4, -0x1.e4e87c0000000p-5, -0x1.a3dcde0000000p-7, 0x1.9656c00000000p-6, 0x1.ba4a860000000p-3, -0x1.b278ec0000000p-4, -0x1.c1ca420000000p-6, 0x1.a4c5820000000p-7, -0x1.9f7ff80000000p-6, 0x1.b3b6300000000p-11, -0x1.51de680000000p-9, 0x1.1979f40000000p-3, 0x1.1520f40000000p-4, -0x1.3d5ac80000000p-3, -0x1.ad6e840000000p-4, -0x1.05d8360000000p-3, 0x1.2b7d100000000p-4, -0x1.7aae500000000p-3, 0x1.9731a20000000p-3, -0x1.9a6cf80000000p-3, 0x1.2e70720000000p-9, -0x1.c42b1a0000000p-6, -0x1.5aea6a0000000p-4, -0x1.baac1a0000000p-4, 0x1.0044600000000p-4, -0x1.1160e80000000p-3, -0x1.572c340000000p-5, -0x1.aaabc20000000p-4, -0x1.e23faa0000000p-4, 0x1.0be0fa0000000p-6, -0x1.6863dc0000000p-8, -0x1.c404b40000000p-4, -0x1.1ecdcc0000000p-5, -0x1.5acabc0000000p-6, -0x1.2c5c460000000p-4, 0x1.59dc080000000p-4, -0x1.45df540000000p-5, 0x1.65f7740000000p-5, -0x1.e45a280000000p-4, 0x1.0ff96c0000000p-5, -0x1.f955980000000p-8, -0x1.e3eba80000000p-4, 0x1.e151f40000000p-4, 0x1.5e783e0000000p-4, 0x1.3ceeb20000000p-9, -0x1.f19c640000000p-6, -0x1.be09980000000p-4, -0x1.9071e80000000p-5, -0x1.ce44f20000000p-5, -0x1.f7546a0000000p-5, 0x1.f2c0220000000p-4, -0x1.284b600000000p-6, 0x1.3821760000000p-3, -0x1.9f43620000000p-3, 0x1.9fa5cc0000000p-3, -0x1.b349e60000000p-10, -0x1.2054b00000000p-5}
, {0x1.79a7000000000p-3, -0x1.aa63a40000000p-4, -0x1.24af160000000p-5, 0x1.54622e0000000p-4, -0x1.d00f380000000p-4, -0x1.aa0b920000000p-3, -0x1.1475f80000000p-3, 0x1.2b14140000000p-2, 0x1.5e74860000000p-5, 0x1.0c357a0000000p-2, 0x1.63eb8e0000000p-3, 0x1.a0e3d00000000p-2, -0x1.dcdd1e0000000p-4, -0x1.0840fa0000000p-7, 0x1.ddb3560000000p-12, -0x1.e7ff0a0000000p-4, -0x1.03dfd60000000p-4, -0x1.9ea3340000000p-5, 0x1.de63460000000p-5, 0x1.c522180000000p-7, 0x1.d6b9400000000p-7, -0x1.737e5c0000000p-7, 0x1.35e3b60000000p-5, 0x1.c676c80000000p-4, -0x1.684d740000000p-2, 0x1.8674da0000000p-5, -0x1.8bca460000000p-4, 0x1.e943080000000p-4, -0x1.50a7a80000000p-2, -0x1.025baa0000000p-2, 0x1.197d620000000p-3, -0x1.ab9a960000000p-6, 0x1.c9a5800000000p-7, -0x1.f41a5e0000000p-4, -0x1.7890d20000000p-3, -0x1.87a3260000000p-3, 0x1.5b3efe0000000p-7, -0x1.2cd45c0000000p-2, -0x1.b1c0580000000p-4, -0x1.9082c80000000p-4, -0x1.a1b2c80000000p-3, 0x1.004db40000000p-4, -0x1.069f620000000p-3, -0x1.4734500000000p-4, 0x1.fd08a40000000p-4, 0x1.93004a0000000p-5, -0x1.8f28cc0000000p-5, 0x1.4ae5e40000000p-5, -0x1.a6aa4c0000000p-5, -0x1.cc1dee0000000p-4, 0x1.e46c7c0000000p-3, 0x1.22143a0000000p-3, -0x1.3f2e780000000p-4, -0x1.1e9cec0000000p-3, -0x1.24a9280000000p-6, -0x1.dfcf820000000p-8, -0x1.a17ade0000000p-4, 0x1.a2724c0000000p-4, -0x1.2fd9160000000p-3, 0x1.85f2300000000p-4, -0x1.4cc93c0000000p-4, 0x1.c430e60000000p-3, -0x1.a5d83e0000000p-3, -0x1.1524bc0000000p-6}
}
, {{-0x1.133f4c0000000p-4, -0x1.3b33740000000p-6, 0x1.2606040000000p-2, 0x1.178e200000000p-4, 0x1.c061ac0000000p-4, -0x1.bf04460000000p-5, -0x1.fa89880000000p-5, 0x1.6f2dd20000000p-3, 0x1.c4cb680000000p-3, 0x1.e66e4c0000000p-5, -0x1.28f1760000000p-3, -0x1.6b10400000000p-10, 0x1.619d760000000p-3, -0x1.cac2c60000000p-5, 0x1.1008ae0000000p-4, 0x1.4769500000000p-6, -0x1.ef992a0000000p-3, -0x1.384c2e0000000p-3, -0x1.f89f680000000p-3, -0x1.9753080000000p-4, 0x1.1361f20000000p-6, 0x1.acbb2a0000000p-5, 0x1.1fb7980000000p-3, -0x1.2ef2440000000p-2, 0x1.0e95520000000p-4, 0x1.74bbbe0000000p-2, 0x1.208d9a0000000p-3, 0x1.c915a80000000p-5, -0x1.12e5440000000p-2, 0x1.9264f20000000p-4, -0x1.0b1cb20000000p-3, -0x1.8ab4080000000p-2, 0x1.5ab6a40000000p-3, 0x1.79b95a0000000p-3, -0x1.24fbbc0000000p-3, 0x1.d070760000000p-3, -0x1.1911320000000p-4, -0x1.02e1a00000000p-6, 0x1.6e4e2e0000000p-5, 0x1.9ea4d20000000p-7, -0x1.3e21480000000p-1, -0x1.5054ea0000000p-4, -0x1.e69b940000000p-6, -0x1.d8c2840000000p-7, 0x1.5e14860000000p-2, 0x1.aaab020000000p-5, -0x1.848b5c0000000p-2, 0x1.131bca0000000p-2, -0x1.1212c80000000p-2, -0x1.2803780000000p-3, 0x1.8dda280000000p-5, -0x1.64595a0000000p-4, 0x1.85aa220000000p-3, -0x1.0fa1940000000p-3, 0x1.ba25a20000000p-5, -0x1.8b33360000000p-3, -0x1.75e6e80000000p-2, 0x1.b1d2fe0000000p-2, 0x1.5c57e20000000p-4, 0x1.c5aa740000000p-5, -0x1.10173e0000000p-3, -0x1.2fcb560000000p-3, 0x1.830c020000000p-7, 0x1.2ca78e0000000p-3}
, {0x1.39bc2e0000000p-3, 0x1.d1d4d60000000p-4, 0x1.38b9120000000p-4, 0x1.fbddda0000000p-9, -0x1.f5bfd60000000p-4, 0x1.2144e00000000p-3, 0x1.2d498c0000000p-5, 0x1.3d262a0000000p-3, -0x1.5760b20000000p-3, 0x1.141d1e0000000p-4, 0x1.29429c0000000p-7, -0x1.2494f60000000p-4, -0x1.6f9d480000000p-3, -0x1.19b8960000000p-4, 0x1.55643a0000000p-4, -0x1.a417e80000000p-4, 0x1.61f56a0000000p-5, -0x1.643edc0000000p-4, -0x1.dc784c0000000p-6, 0x1.78dc640000000p-4, 0x1.74e94c0000000p-5, 0x1.9514400000000p-4, -0x1.4111940000000p-4, 0x1.59354a0000000p-4, -0x1.e877ce0000000p-4, -0x1.bd9b180000000p-5, -0x1.292e5c0000000p-5, -0x1.c78d120000000p-5, 0x1.4e6d260000000p-7, 0x1.31286a0000000p-6, -0x1.1ee9c40000000p-3, -0x1.4937ac0000000p-4, 0x1.ed83ca0000000p-5, -0x1.8f053c0000000p-4, 0x1.2082620000000p-4, 0x1.073eac0000000p-5, 0x1.4f07080000000p-3, 0x1.8778a00000000p-4, -0x1.5cda980000000p-2, -0x1.5b90da0000000p-11, -0x1.8da4b40000000p-3, -0x1.7fe1360000000p-4, 0x1.7054f80000000p-4, -0x1.2ef5e20000000p-3, 0x1.87c3a20000000p-6, -0x1.d511980000000p-4, 0x1.5c91ee0000000p-6, 0x1.a5a2f20000000p-4, 0x1.6afe1a0000000p-3, -0x1.2aa70c0000000p-3, 0x1.4ca9a00000000p-4, -0x1.4a02020000000p-5, -0x1.d380bc0000000p-8, -0x1.e72a500000000p-5, -0x1.8bec780000000p-6, -0x1.cf36640000000p-3, -0x1.6ccbd40000000p-4, 0x1.429ccc0000000p-5, 0x1.c0c4840000000p-4, 0x1.0364260000000p-3, -0x1.d437760000000p-6, -0x1.c9d4c60000000p-5, -0x1.6ea4200000000p-4, 0x1.419a1a0000000p-6}
, {0x1.1a42400000000p-8, -0x1.0967620000000p-3, -0x1.4378320000000p-4, -0x1.a222680000000p-3, -0x1.d4fb560000000p-5, -0x1.2636b20000000p-4, -0x1.ff2e9c0000000p-5, 0x1.207cbe0000000p-2, 0x1.00c68e0000000p-3, -0x1.0b05920000000p-2, 0x1.0601880000000p-5, 0x1.49a4d20000000p-5, 0x1.03cd9c0000000p-8, 0x1.7d760a0000000p-3, -0x1.018e040000000p-5, -0x1.add5080000000p-5, -0x1.2de6b00000000p-3, 0x1.643c6a0000000p-6, 0x1.ede6f20000000p-4, -0x1.93aabc0000000p-3, -0x1.73292e0000000p-3, 0x1.d8dec40000000p-4, -0x1.557a2e0000000p-4, 0x1.e1953e0000000p-4, -0x1.dfe5680000000p-7, -0x1.d00a1a0000000p-5, -0x1.d3c5920000000p-5, -0x1.03a3640000000p-4, 0x1.6c03fe0000000p-4, -0x1.a2a8660000000p-5, 0x1.b0c2580000000p-11, -0x1.1ec5980000000p-4, -0x1.116f4e0000000p-2, -0x1.6ed6be0000000p-3, 0x1.49973c0000000p-5, 0x1.d492a40000000p-3, -0x1.b3a7c20000000p-3, 0x1.6e5ca80000000p-2, 0x1.367e100000000p-3, 0x1.6fb3be0000000p-3, -0x1.6594380000000p-2, 0x1.ffbcc20000000p-4, -0x1.d939a80000000p-3, 0x1.c013560000000p-3, 0x1.9db76a0000000p-5, -0x1.4c45ae0000000p-3, 0x1.8b276e0000000p-2, -0x1.a7de080000000p-5, 0x1.3c28960000000p-3, -0x1.7283100000000p-3, 0x1.1314040000000p-6, -0x1.cbd8500000000p-5, -0x1.2050940000000p-3, -0x1.2f2f700000000p-4, -0x1.2c04440000000p-4, -0x1.cea7b60000000p-3, 0x1.70277c0000000p-3, -0x1.01964c0000000p-4, 0x1.68bca60000000p-5, -0x1.e8b0400000000p-3, -0x1.71ff6a0000000p-4, -0x1.309af80000000p-6, -0x1.92fcf80000000p-4, 0x1.2b0c720000000p-5}
}
, {{0x1.2a28ba0000000p-3, -0x1.20f6680000000p-3, -0x1.229e8c0000000p-4, -0x1.d446de0000000p-4, -0x1.77643a0000000p-6, -0x1.b00bea0000000p-4, 0x1.de0c540000000p-6, -0x1.0510480000000p-6, 0x1.4928fa0000000p-5, -0x1.d5d5ba0000000p-5, 0x1.167b540000000p-6, -0x1.6aa89e0000000p-5, -0x1.1c076a0000000p-2, -0x1.4fba600000000p-3, 0x1.d1ab7e0000000p-5, 0x1.1b41460000000p-7, -0x1.bbacde0000000p-6, 0x1.8d2d040000000p-4, -0x1.1fb28c0000000p-3, 0x1.2c16fa0000000p-4, -0x1.a1d0160000000p-5, -0x1.6fa8f60000000p-5, -0x1.2e5d480000000p-3, -0x1.4c93820000000p-4, -0x1.e81c900000000p-4, -0x1.0271680000000p-5, -0x1.9effa40000000p-3, -0x1.e1a8c20000000p-3, 0x1.48ccbc0000000p-4, -0x1.cb85dc0000000p-3, 0x1.da7b860000000p-3, -0x1.79d2280000000p-5, -0x1.0d80240000000p-3, -0x1.4618ec0000000p-6, 0x1.b777d00000000p-3, 0x1.b04afc0000000p-9, -0x1.6bbba20000000p-6, 0x1.0995bc0000000p-4, -0x1.5312e40000000p-3, 0x1.c1bffe0000000p-7, 0x1.e824100000000p-5, 0x1.971c1e0000000p-4, -0x1.62202a0000000p-4, -0x1.99513a0000000p-3, -0x1.3bba120000000p-2, -0x1.02029c0000000p-4, 0x1.6070b80000000p-4, 0x1.17ef980000000p-5, 0x1.0673960000000p-3, -0x1.7273320000000p-4, -0x1.b67f9e0000000p-3, 0x1.2ee4460000000p-4, 0x1.010c380000000p-7, -0x1.1793880000000p-4, -0x1.2d75800000000p-5, -0x1.386b4a0000000p-5, -0x1.6ebe940000000p-4, -0x1.f6fda40000000p-7, -0x1.1049300000000p-4, -0x1.00b4360000000p-2, -0x1.cce5560000000p-4, 0x1.5a445e0000000p-3, -0x1.48aac60000000p-3, -0x1.cc07a60000000p-4}
, {-0x1.3a07de0000000p-6, 0x1.11f4b00000000p-6, 0x1.1b58520000000p-5, 0x1.0560c80000000p-4, 0x1.646bfa0000000p-3, 0x1.6ed42a0000000p-7, 0x1.4d86100000000p-7, 0x1.2dd68c0000000p-5, -0x1.70c3840000000p-5, -0x1.9d4e1a0000000p-4, 0x1.3c455a0000000p-3, -0x1.2ff3c80000000p-5, -0x1.bbd3b80000000p-3, 0x1.c8f9a00000000p-3, -0x1.aa17520000000p-3, -0x1.8698ac0000000p-3, -0x1.85418a0000000p-4, 0x1.a578c80000000p-3, -0x1.ad1fd60000000p-4, -0x1.ceb5da0000000p-6, 0x1.9ee4f60000000p-4, -0x1.50e62c0000000p-5, -0x1.40bb4a0000000p-7, 0x1.3710060000000p-4, 0x1.bef23c0000000p-5, -0x1.3376840000000p-3, -0x1.556df60000000p-3, -0x1.292fde0000000p-6, -0x1.eb48000000000p-5, -0x1.07b3780000000p-5, 0x1.9ae6fe0000000p-7, -0x1.104eee0000000p-5, 0x1.50df180000000p-3, -0x1.02a79e0000000p-3, -0x1.8ff8980000000p-4, -0x1.65ea000000000p-4, 0x1.71ecae0000000p-4, -0x1.4e8f900000000p-5, -0x1.d04a020000000p-6, 0x1.3807ec0000000p-6, 0x1.600bb00000000p-3, 0x1.48bb0c0000000p-9, 0x1.4db8900000000p-5, -0x1.1a04b80000000p-4, -0x1.6ce31a0000000p-5, -0x1.25292c0000000p-4, -0x1.589e640000000p-6, -0x1.8cb8d00000000p-5, -0x1.14713c0000000p-5, 0x1.ff242c0000000p-5, -0x1.8fc6340000000p-3, -0x1.4cfb5e0000000p-6, -0x1.dde7080000000p-4, 0x1.abef940000000p-5, 0x1.48f6480000000p-3, -0x1.c3855a0000000p-7, -0x1.ba29c20000000p-5, -0x1.99d2460000000p-5, -0x1.fe94420000000p-6, -0x1.4b1c6a0000000p-3, -0x1.b4df4a0000000p-4, -0x1.050fba0000000p-4, -0x1.d0ff120000000p-5, 0x1.7313620000000p-5}
, {0x1.d195b00000000p-4, 0x1.7964fc0000000p-5, -0x1.0308840000000p-6, 0x1.1b60e80000000p-2, 0x1.5c58a40000000p-5, -0x1.12bb2c0000000p-3, 0x1.1ddaae0000000p-4, 0x1.a1c5ec0000000p-5, 0x1.db1cae0000000p-14, 0x1.a662d40000000p-5, 0x1.4f41c40000000p-4, -0x1.1938a20000000p-6, -0x1.9f97e00000000p-3, 0x1.a2a25e0000000p-4, 0x1.c1dcdc0000000p-5, -0x1.fdd9240000000p-3, -0x1.4e64900000000p-4, -0x1.48e2ce0000000p-4, 0x1.6d0d2c0000000p-3, 0x1.430ede0000000p-4, -0x1.b356440000000p-4, -0x1.5bb7e60000000p-3, 0x1.a23dac0000000p-3, 0x1.7a66180000000p-3, 0x1.4205740000000p-8, 0x1.79ab8c0000000p-5, -0x1.63b95c0000000p-6, 0x1.2333480000000p-2, -0x1.c922f80000000p-4, -0x1.7ccecc0000000p-3, -0x1.59b8a00000000p-5, 0x1.25a69c0000000p-7, -0x1.64297e0000000p-5, 0x1.5992480000000p-3, -0x1.d115e40000000p-6, -0x1.ac4c2a0000000p-4, 0x1.1b39760000000p-2, -0x1.60aa420000000p-2, 0x1.c28e420000000p-3, -0x1.4bd8be0000000p-5, 0x1.3efe1c0000000p-3, 0x1.734d320000000p-3, 0x1.63c98c0000000p-3, -0x1.70e21c0000000p-5, -0x1.5373f80000000p-5, -0x1.15a49a0000000p-2, -0x1.0868760000000p-4, -0x1.73d45a0000000p-7, -0x1.6a2d040000000p-3, 0x1.d5dd380000000p-7, -0x1.dce1a60000000p-7, -0x1.4836160000000p-5, -0x1.1ff4100000000p-4, -0x1.f656360000000p-3, -0x1.1453b60000000p-4, -0x1.05fb800000000p-3, -0x1.1c017a0000000p-2, -0x1.d114ec0000000p-4, 0x1.71b9380000000p-5, 0x1.570b460000000p-5, -0x1.719e960000000p-5, 0x1.d52b040000000p-8, 0x1.a98b6c0000000p-3, -0x1.92dc500000000p-5}
}
, {{0x1.4796080000000p-3, 0x1.0bbf1a0000000p-3, 0x1.9c8f680000000p-3, -0x1.914c260000000p-6, -0x1.583c0e0000000p-5, 0x1.c298d40000000p-3, 0x1.4f2a120000000p-6, 0x1.c604880000000p-5, 0x1.62e33c0000000p-5, 0x1.b9292c0000000p-5, 0x1.95fa4e0000000p-6, 0x1.b920540000000p-5, 0x1.a227d20000000p-4, -0x1.de0b1e0000000p-5, 0x1.0ea27e0000000p-7, -0x1.e378fa0000000p-5, 0x1.a7eda00000000p-5, 0x1.67616c0000000p-5, -0x1.507ce00000000p-3, -0x1.19f5f60000000p-3, -0x1.6244980000000p-6, 0x1.287eda0000000p-4, -0x1.d5b4460000000p-5, 0x1.9bbe8e0000000p-4, 0x1.2a5ee60000000p-5, 0x1.e5195e0000000p-4, 0x1.5269f60000000p-3, -0x1.05bd1e0000000p-5, 0x1.e01dba0000000p-5, 0x1.d7ba1a0000000p-4, 0x1.d491b80000000p-8, 0x1.1879e00000000p-3, -0x1.4bde8c0000000p-4, 0x1.f1a3660000000p-5, 0x1.42dbc20000000p-7, 0x1.2e08420000000p-6, -0x1.8e4e480000000p-5, 0x1.077e700000000p-3, 0x1.e0b6980000000p-4, 0x1.67880c0000000p-4, 0x1.d001ac0000000p-4, 0x1.ed83140000000p-8, 0x1.d710220000000p-6, -0x1.4d76080000000p-5, 0x1.a6f3820000000p-6, -0x1.1c83b60000000p-5, 0x1.7f91e20000000p-4, -0x1.81fc460000000p-3, 0x1.57bfc00000000p-3, -0x1.0a01f40000000p-4, 0x1.f42a140000000p-5, 0x1.d076ae0000000p-6, 0x1.94ff680000000p-5, 0x1.01c0420000000p-3, 0x1.e83b940000000p-3, 0x1.2ef6b20000000p-4, 0x1.c5c9cc0000000p-5, 0x1.55afd40000000p-5, 0x1.b2998c0000000p-8, -0x1.82b0180000000p-5, -0x1.f1fb0a0000000p-4, -0x1.67c5f20000000p-4, -0x1.8a0b400000000p-4, 0x1.d3b5860000000p-5}
, {-0x1.ff526a0000000p-4, 0x1.cbdc0e0000000p-4, 0x1.58c0420000000p-6, 0x1.3ca05e0000000p-4, -0x1.5e66b60000000p-8, 0x1.1429f20000000p-4, 0x1.4f78f20000000p-3, 0x1.22964c0000000p-3, 0x1.f06d5c0000000p-6, 0x1.751f1e0000000p-3, 0x1.0e008a0000000p-4, -0x1.4718760000000p-4, -0x1.1b397a0000000p-4, 0x1.e20d900000000p-6, 0x1.f9f1b00000000p-6, 0x1.75066a0000000p-4, 0x1.d320040000000p-3, -0x1.a0ffbc0000000p-5, -0x1.732cec0000000p-4, -0x1.04025e0000000p-6, -0x1.17ce0e0000000p-5, 0x1.62e2dc0000000p-5, 0x1.7752f60000000p-3, 0x1.a3bdc80000000p-3, -0x1.754c540000000p-4, 0x1.10210c0000000p-4, 0x1.c2a9620000000p-5, 0x1.89b1520000000p-5, 0x1.23595c0000000p-3, -0x1.066b340000000p-4, -0x1.6d8c4a0000000p-3, 0x1.0f8a920000000p-5, 0x1.3c71d20000000p-12, -0x1.80eb7c0000000p-7, -0x1.ff0f320000000p-8, -0x1.85ec9e0000000p-4, -0x1.d8108c0000000p-5, 0x1.da8e5a0000000p-4, -0x1.70bd300000000p-6, 0x1.585f860000000p-5, 0x1.e4bc360000000p-4, 0x1.63d8a20000000p-5, -0x1.4dfae20000000p-6, -0x1.c6682c0000000p-8, -0x1.01d9ec0000000p-3, 0x1.f004fc0000000p-5, 0x1.2468360000000p-3, -0x1.5e15560000000p-4, 0x1.afabba0000000p-5, -0x1.37e8940000000p-3, 0x1.8884a20000000p-3, 0x1.d932bc0000000p-7, 0x1.3d2cd20000000p-3, 0x1.ef0f740000000p-4, 0x1.808ccc0000000p-7, 0x1.77bf0e0000000p-7, 0x1.5270160000000p-5, 0x1.35255a0000000p-4, 0x1.6665160000000p-4, 0x1.314e800000000p-3, -0x1.1615ee0000000p-3, -0x1.45b8040000000p-3, 0x1.8066c80000000p-3, -0x1.13aaba0000000p-4}
, {0x1.6a88d80000000p-5, -0x1.b99e3a0000000p-7, -0x1.52ef2c0000000p-3, -0x1.4dd19a0000000p-4, 0x1.1874260000000p-4, -0x1.2919f20000000p-4, 0x1.9056a80000000p-6, -0x1.8009bc0000000p-7, -0x1.86c6d40000000p-8, 0x1.cf14e20000000p-4, 0x1.b469400000000p-8, 0x1.489cd60000000p-3, 0x1.9628be0000000p-5, 0x1.6c400a0000000p-4, -0x1.7ada8c0000000p-8, -0x1.89cc500000000p-5, 0x1.9e7ad40000000p-3, 0x1.03f14a0000000p-5, 0x1.3979d60000000p-4, 0x1.e5cc460000000p-5, -0x1.45460c0000000p-3, 0x1.c18dd40000000p-4, -0x1.076ebc0000000p-4, 0x1.25152c0000000p-4, 0x1.c0beec0000000p-10, -0x1.21f4800000000p-4, 0x1.777c860000000p-4, 0x1.4f593c0000000p-4, 0x1.037d980000000p-3, -0x1.fa88520000000p-4, -0x1.3bebc80000000p-3, 0x1.26cfee0000000p-6, 0x1.2b9f0c0000000p-3, 0x1.c8d85e0000000p-4, 0x1.0827720000000p-4, 0x1.922d3c0000000p-5, 0x1.6c7d000000000p-4, 0x1.7ddd8e0000000p-4, 0x1.65b6d60000000p-3, -0x1.6147a80000000p-6, -0x1.fbaae20000000p-7, 0x1.345a240000000p-3, 0x1.81ab000000000p-5, -0x1.15d0120000000p-4, -0x1.1a72e00000000p-5, -0x1.1460160000000p-5, 0x1.23630a0000000p-3, -0x1.681a2a0000000p-4, 0x1.621af60000000p-4, 0x1.7a44e80000000p-4, 0x1.2eced40000000p-3, -0x1.87a79a0000000p-5, -0x1.26de1c0000000p-4, 0x1.5ee16e0000000p-6, 0x1.9af17c0000000p-5, 0x1.7ffc780000000p-3, 0x1.b4ba1e0000000p-3, -0x1.6391560000000p-4, 0x1.b6dc300000000p-4, 0x1.5469500000000p-3, 0x1.4dcc220000000p-6, 0x1.700a760000000p-6, 0x1.5f0bc60000000p-3, 0x1.6318fe0000000p-3}
}
, {{-0x1.fa74720000000p-5, -0x1.2ca8680000000p-3, -0x1.9496840000000p-3, 0x1.86d3aa0000000p-12, -0x1.8377be0000000p-5, -0x1.0d5a000000000p-3, 0x1.accfde0000000p-6, -0x1.4d17ac0000000p-3, -0x1.1a5d6e0000000p-2, 0x1.9345200000000p-5, -0x1.5ba0c60000000p-3, 0x1.ab962e0000000p-3, 0x1.92b17e0000000p-6, 0x1.4d249e0000000p-2, 0x1.2892660000000p-6, 0x1.3dbc620000000p-2, 0x1.b92b260000000p-5, 0x1.a410ac0000000p-6, 0x1.563b740000000p-4, 0x1.df9d780000000p-4, 0x1.174bd20000000p-5, -0x1.0e1b640000000p-2, 0x1.0f5af60000000p-3, 0x1.c5ce100000000p-9, -0x1.7c477e0000000p-6, 0x1.28e1420000000p-3, -0x1.1cdeaa0000000p-3, -0x1.e23dde0000000p-5, -0x1.f252360000000p-4, -0x1.474eda0000000p-3, -0x1.0793660000000p-7, 0x1.a4ff700000000p-5, 0x1.5f82b60000000p-5, -0x1.7625300000000p-3, -0x1.5e462c0000000p-3, -0x1.397c000000000p-4, 0x1.8372400000000p-5, 0x1.637ca20000000p-6, -0x1.34e1d40000000p-4, -0x1.8c64060000000p-2, -0x1.9f35000000000p-6, -0x1.2914560000000p-4, -0x1.0525860000000p-2, -0x1.74a8080000000p-4, -0x1.cd0db80000000p-5, 0x1.14dda80000000p-6, 0x1.6b640c0000000p-5, -0x1.1027780000000p-2, -0x1.18b99c0000000p-3, 0x1.81438e0000000p-6, -0x1.6df5920000000p-2, -0x1.c3a2a80000000p-3, -0x1.fd2d4a0000000p-3, -0x1.6973820000000p-3, -0x1.e421ea0000000p-4, -0x1.f88e180000000p-5, 0x1.0493940000000p-5, -0x1.442e740000000p-7, -0x1.2eacc60000000p-3, 0x1.9a506e0000000p-5, 0x1.ad6adc0000000p-6, -0x1.947fe20000000p-5, -0x1.cdf0340000000p-4, 0x1.135f1a0000000p-5}
, {0x1.3df3600000000p-4, -0x1.452d7a0000000p-3, -0x1.8f5f2e0000000p-5, 0x1.a36f020000000p-5, -0x1.30b0b00000000p-3, 0x1.6c7f780000000p-3, -0x1.2ffd0a0000000p-5, -0x1.22783c0000000p-3, 0x1.f880ae0000000p-6, -0x1.e7e1ba0000000p-4, -0x1.38a6d40000000p-4, 0x1.529e660000000p-7, 0x1.2a9e060000000p-4, -0x1.3aebfc0000000p-5, 0x1.64efee0000000p-4, -0x1.6793100000000p-6, -0x1.2458600000000p-4, 0x1.a9d07c0000000p-4, -0x1.9371c20000000p-4, 0x1.521a960000000p-4, 0x1.51e4b60000000p-3, -0x1.dbf4760000000p-6, 0x1.0479b60000000p-4, -0x1.71ef6a0000000p-3, -0x1.91755a0000000p-4, 0x1.c661220000000p-3, 0x1.1dff9c0000000p-5, -0x1.5666800000000p-10, -0x1.5a80400000000p-6, -0x1.5406e60000000p-8, 0x1.32b4be0000000p-8, -0x1.82759e0000000p-3, 0x1.f858b40000000p-5, -0x1.0034ac0000000p-4, 0x1.eaa8980000000p-5, -0x1.57eb420000000p-5, 0x1.68f6c20000000p-3, -0x1.6ca3940000000p-6, 0x1.6e5b4a0000000p-6, 0x1.6e03f40000000p-5, -0x1.1e93080000000p-5, -0x1.09b8600000000p-4, 0x1.72a7860000000p-6, 0x1.095ff00000000p-4, 0x1.1a5e320000000p-5, 0x1.ab47d60000000p-9, -0x1.10ec880000000p-3, 0x1.5eb0c00000000p-5, 0x1.61a6a00000000p-5, 0x1.91bd180000000p-7, 0x1.7785d60000000p-4, -0x1.1feb2e0000000p-5, -0x1.f6ee5e0000000p-5, -0x1.74ac160000000p-3, -0x1.b7a72e0000000p-9, 0x1.67b8620000000p-5, -0x1.3ea5fe0000000p-3, 0x1.ea59e60000000p-4, -0x1.bd26f20000000p-4, 0x1.81170c0000000p-3, 0x1.029fea0000000p-2, -0x1.2424520000000p-4, -0x1.e8be4c0000000p-4, 0x1.06c5f40000000p-4}
, {-0x1.e87f360000000p-5, -0x1.9727860000000p-9, -0x1.9f9afc0000000p-6, 0x1.34fe660000000p-3, -0x1.d0f7900000000p-4, -0x1.a4ef020000000p-3, -0x1.89dbf80000000p-6, 0x1.853fac0000000p-4, 0x1.9de65c0000000p-4, -0x1.b3502c0000000p-4, 0x1.55d6f80000000p-3, 0x1.49861c0000000p-5, 0x1.5e69f00000000p-8, -0x1.e7a13a0000000p-4, -0x1.23f8620000000p-4, 0x1.028f0e0000000p-4, 0x1.116c240000000p-7, -0x1.8e2fa20000000p-7, -0x1.59364a0000000p-3, 0x1.bcb37e0000000p-3, -0x1.22508c0000000p-3, -0x1.6b09800000000p-4, -0x1.09b6340000000p-10, -0x1.0427ea0000000p-6, -0x1.601c160000000p-3, 0x1.99588a0000000p-3, 0x1.80cef60000000p-4, -0x1.37c2da0000000p-4, 0x1.74319c0000000p-6, -0x1.cb5a620000000p-4, 0x1.50149a0000000p-3, 0x1.cd190e0000000p-3, 0x1.05c49e0000000p-3, -0x1.c1f6d00000000p-4, 0x1.1f85f80000000p-6, -0x1.2955b20000000p-5, 0x1.f88dfe0000000p-4, -0x1.2625bc0000000p-2, -0x1.197a240000000p-2, -0x1.92b5320000000p-5, -0x1.1724ee0000000p-3, 0x1.ca519c0000000p-3, 0x1.8c24a60000000p-4, -0x1.b943520000000p-4, -0x1.e098600000000p-6, 0x1.436cac0000000p-4, -0x1.76f2ae0000000p-4, -0x1.ff41fe0000000p-6, -0x1.220c320000000p-4, -0x1.a276180000000p-5, 0x1.bbe56c0000000p-6, -0x1.4c19ae0000000p-6, 0x1.2b1fb80000000p-3, -0x1.00c2940000000p-3, -0x1.2072320000000p-4, 0x1.b8701e0000000p-4, -0x1.1213b60000000p-2, 0x1.6bb2e80000000p-3, -0x1.3470460000000p-3, 0x1.2a69440000000p-3, 0x1.1939be0000000p-2, 0x1.5d9d4e0000000p-5, -0x1.d2a1ec0000000p-6, -0x1.102b6a0000000p-4}
}
, {{-0x1.d4bbb40000000p-4, 0x1.65e3b80000000p-3, -0x1.3335020000000p-2, -0x1.eb42760000000p-5, 0x1.30c42c0000000p-4, 0x1.6a8d9a0000000p-4, -0x1.314fa60000000p-3, -0x1.2d239e0000000p-3, -0x1.1124180000000p-2, -0x1.797eec0000000p-4, 0x1.59f9c00000000p-3, 0x1.c3242c0000000p-3, -0x1.4730100000000p-2, -0x1.dcf0c20000000p-6, -0x1.5a7c680000000p-3, -0x1.9f3b6a0000000p-5, -0x1.5f89d20000000p-4, -0x1.5973fe0000000p-5, 0x1.268e420000000p-6, 0x1.64de9e0000000p-3, -0x1.a9f98e0000000p-3, -0x1.04d55e0000000p-3, 0x1.0d5ba20000000p-2, 0x1.18e92a0000000p-3, -0x1.4b56000000000p-4, -0x1.1d02e00000000p-4, -0x1.3744100000000p-5, -0x1.0114b00000000p-5, -0x1.5765980000000p-4, -0x1.745e920000000p-3, 0x1.0909f60000000p-4, 0x1.6ac9260000000p-4, 0x1.14dac80000000p-3, -0x1.25fd880000000p-3, -0x1.3217820000000p-3, -0x1.0bb8ea0000000p-8, -0x1.7b2ab40000000p-5, -0x1.06a4a40000000p-6, 0x1.0c5d040000000p-2, -0x1.a92fd00000000p-3, 0x1.3c573e0000000p-3, -0x1.b2910a0000000p-4, -0x1.ea596e0000000p-4, -0x1.8ca5e20000000p-3, 0x1.af94500000000p-5, -0x1.3a181a0000000p-3, 0x1.7d9e9c0000000p-4, -0x1.1f22680000000p-2, -0x1.d6668e0000000p-5, -0x1.f3e1440000000p-5, -0x1.9e0ffe0000000p-4, -0x1.0c0bea0000000p-2, -0x1.1974180000000p-2, -0x1.1e42a60000000p-3, -0x1.c9e4ee0000000p-10, -0x1.0068de0000000p-8, -0x1.446aaa0000000p-7, -0x1.4bc2440000000p-2, 0x1.9fce4e0000000p-3, 0x1.8c55640000000p-3, 0x1.edf4180000000p-6, 0x1.d461de0000000p-6, -0x1.7a63fa0000000p-9, 0x1.6a785e0000000p-3}
, {-0x1.845e920000000p-6, 0x1.1bf3d60000000p-4, -0x1.554e1c0000000p-3, 0x1.2af3660000000p-4, 0x1.fe00420000000p-4, -0x1.0817aa0000000p-5, -0x1.7831600000000p-3, 0x1.fd91540000000p-4, -0x1.e4efb20000000p-3, 0x1.6cb1cc0000000p-5, -0x1.645acc0000000p-4, 0x1.119e440000000p-3, -0x1.35ad320000000p-3, -0x1.c7e28e0000000p-4, 0x1.63182a0000000p-4, -0x1.d83ca00000000p-5, -0x1.99cfce0000000p-3, -0x1.53b1280000000p-7, 0x1.7917060000000p-6, 0x1.c28a860000000p-6, -0x1.d872020000000p-5, 0x1.2925e20000000p-4, 0x1.1c110a0000000p-4, -0x1.3433fe0000000p-3, 0x1.3cfcd00000000p-5, -0x1.c6b29e0000000p-4, 0x1.7d5f460000000p-5, 0x1.26163c0000000p-9, -0x1.e081ae0000000p-4, -0x1.0cc2e00000000p-4, 0x1.35945c0000000p-4, 0x1.3bcfae0000000p-4, 0x1.8dc7940000000p-6, 0x1.25b2080000000p-4, -0x1.785d420000000p-4, -0x1.02118a0000000p-6, 0x1.c6d0340000000p-5, -0x1.fabb5a0000000p-4, 0x1.0d2f740000000p-4, -0x1.1e523e0000000p-3, -0x1.98558c0000000p-3, -0x1.100bca0000000p-5, -0x1.e61f260000000p-5, -0x1.3a9c4a0000000p-3, -0x1.83a9be0000000p-5, -0x1.8930640000000p-5, 0x1.595d400000000p-7, -0x1.7e1f1a0000000p-5, -0x1.8749c40000000p-3, 0x1.d286a20000000p-4, 0x1.0b3c920000000p-3, -0x1.11a9920000000p-7, -0x1.f1e6e60000000p-4, -0x1.c18dfc0000000p-5, 0x1.07efb20000000p-4, -0x1.bb1da60000000p-5, -0x1.b9a7a20000000p-7, 0x1.1570ea0000000p-6, 0x1.7c760c0000000p-4, 0x1.6c7b060000000p-4, 0x1.da36ca0000000p-5, 0x1.1415de0000000p-3, -0x1.5e68060000000p-9, 0x1.1615000000000p-4}
, {-0x1.1023360000000p-5, 0x1.5dc3520000000p-4, 0x1.9bd6fc0000000p-4, -0x1.93b9d20000000p-5, -0x1.333d0c0000000p-3, -0x1.ba12d60000000p-4, -0x1.661c500000000p-3, 0x1.029dce0000000p-3, 0x1.9b04120000000p-3, 0x1.e20e7e0000000p-3, -0x1.1efef80000000p-5, 0x1.4342740000000p-3, 0x1.3da48c0000000p-5, -0x1.65ecfa0000000p-4, 0x1.bb9f380000000p-8, -0x1.3d74c20000000p-8, -0x1.b539b00000000p-3, -0x1.f610de0000000p-6, 0x1.836f440000000p-3, -0x1.0393c60000000p-8, 0x1.0ef0120000000p-3, 0x1.e0b3700000000p-4, -0x1.334f2a0000000p-4, 0x1.feb01e0000000p-4, -0x1.0ad8ca0000000p-5, -0x1.4cc2d60000000p-5, 0x1.9722660000000p-3, 0x1.00c8c00000000p-6, -0x1.4d1d5c0000000p-3, -0x1.0ccb1c0000000p-4, 0x1.cb271e0000000p-7, -0x1.32f9b00000000p-2, -0x1.f3f5ca0000000p-3, 0x1.a7ad280000000p-3, 0x1.656a5a0000000p-3, -0x1.1063160000000p-4, -0x1.1ea3ca0000000p-2, -0x1.808d9a0000000p-3, -0x1.1c231a0000000p-5, 0x1.2d1fde0000000p-4, -0x1.d2545a0000000p-2, -0x1.78965e0000000p-4, -0x1.059a5e0000000p-4, -0x1.ad8dee0000000p-3, 0x1.be268e0000000p-6, -0x1.1b35940000000p-4, -0x1.0be0fa0000000p-9, -0x1.0cce380000000p-8, 0x1.3ef3f20000000p-4, -0x1.1cc83a0000000p-4, 0x1.19c99c0000000p-3, 0x1.ce2f540000000p-4, 0x1.7361180000000p-4, -0x1.57fdfc0000000p-2, 0x1.4e54180000000p-3, -0x1.686eba0000000p-2, -0x1.6c15600000000p-4, 0x1.6a652c0000000p-3, -0x1.27c7760000000p-3, -0x1.157c5e0000000p-8, -0x1.2cb7e00000000p-2, 0x1.77b7e60000000p-3, -0x1.fa4b020000000p-4, -0x1.6940b00000000p-3}
}
, {{0x1.98e2f20000000p-4, 0x1.e885800000000p-12, -0x1.57eafe0000000p-6, -0x1.4d68440000000p-3, 0x1.dbd8ea0000000p-5, -0x1.0c840a0000000p-5, -0x1.ce5a2c0000000p-4, -0x1.4070140000000p-8, -0x1.7071d40000000p-3, 0x1.086cac0000000p-4, -0x1.4414f40000000p-5, 0x1.814ec20000000p-6, -0x1.2352220000000p-3, -0x1.3007840000000p-3, -0x1.60b56e0000000p-8, -0x1.1a4df00000000p-3, 0x1.24d4b80000000p-3, 0x1.a96ee80000000p-5, 0x1.40fcea0000000p-4, 0x1.f45e460000000p-4, -0x1.a9e2420000000p-5, -0x1.e311bc0000000p-4, -0x1.a21d340000000p-5, -0x1.c1a92e0000000p-4, -0x1.df8e8a0000000p-5, 0x1.6108520000000p-6, 0x1.856f4a0000000p-7, -0x1.0250e00000000p-3, -0x1.6e022a0000000p-5, -0x1.25beb60000000p-3, 0x1.793c800000000p-6, 0x1.f854340000000p-3, -0x1.8775f80000000p-5, -0x1.108bf40000000p-6, 0x1.2d2c720000000p-3, 0x1.20477e0000000p-8, 0x1.b373280000000p-4, 0x1.c8816c0000000p-10, -0x1.3023dc0000000p-3, -0x1.043cd20000000p-4, 0x1.bea35a0000000p-7, -0x1.1d7e040000000p-3, -0x1.459fc80000000p-5, -0x1.0fb1280000000p-3, -0x1.115ff80000000p-7, -0x1.8f4d4e0000000p-3, -0x1.11f0de0000000p-3, -0x1.0164540000000p-2, 0x1.3e6edc0000000p-4, 0x1.bf632a0000000p-6, -0x1.bb5ea40000000p-5, -0x1.7ceb8e0000000p-5, -0x1.0eef160000000p-4, 0x1.d436340000000p-7, 0x1.9c94740000000p-6, 0x1.a544520000000p-4, -0x1.31b7040000000p-4, -0x1.f815fc0000000p-6, -0x1.71109a0000000p-4, -0x1.42acac0000000p-3, 0x1.18cbc00000000p-3, -0x1.e4dbd20000000p-5, -0x1.071b4c0000000p-3, 0x1.8842dc0000000p-7}
, {0x1.2e8a0e0000000p-4, 0x1.a5b3fc0000000p-8, 0x1.f8606e0000000p-5, 0x1.0f18aa0000000p-3, -0x1.42f1dc0000000p-5, -0x1.653e8a0000000p-7, -0x1.6be9820000000p-3, 0x1.28e8d20000000p-5, 0x1.422d720000000p-8, -0x1.b770280000000p-7, 0x1.60b1460000000p-4, -0x1.25733c0000000p-11, -0x1.19edbc0000000p-4, -0x1.46721e0000000p-3, 0x1.331cbe0000000p-4, 0x1.93a7b60000000p-4, -0x1.7374ba0000000p-5, -0x1.0b52520000000p-3, -0x1.3902640000000p-5, 0x1.51990a0000000p-5, -0x1.d52fce0000000p-7, -0x1.ced2960000000p-5, -0x1.cb50d80000000p-5, -0x1.18ad140000000p-5, 0x1.86b24c0000000p-6, 0x1.cedc420000000p-6, -0x1.fbe6120000000p-4, -0x1.0fbd540000000p-3, 0x1.59636e0000000p-6, -0x1.c1574e0000000p-7, -0x1.f174620000000p-7, -0x1.b5384c0000000p-5, -0x1.ce1d580000000p-5, -0x1.a8723e0000000p-4, 0x1.44bc2c0000000p-5, 0x1.b6416a0000000p-6, -0x1.82500a0000000p-3, 0x1.ac1d500000000p-4, 0x1.22b3d40000000p-4, 0x1.85850a0000000p-4, -0x1.4e71180000000p-4, 0x1.2a53de0000000p-5, 0x1.796a5e0000000p-12, 0x1.2d437e0000000p-6, -0x1.36724a0000000p-3, -0x1.b70c4a0000000p-4, 0x1.3c5eb60000000p-12, -0x1.fc3e760000000p-7, -0x1.8ec28c0000000p-7, -0x1.e549d40000000p-6, 0x1.093a9e0000000p-3, -0x1.4a559e0000000p-4, 0x1.507de80000000p-5, 0x1.1f3bf40000000p-5, -0x1.c882ac0000000p-6, -0x1.15c92c0000000p-6, -0x1.1f59dc0000000p-3, -0x1.dbca220000000p-9, 0x1.fc12600000000p-5, -0x1.4b98900000000p-3, 0x1.c8d9340000000p-7, -0x1.557c7c0000000p-6, -0x1.0ad2b40000000p-4, -0x1.3177ca0000000p-4}
, {0x1.40428e0000000p-4, 0x1.1dfd800000000p-6, -0x1.3c23e00000000p-3, -0x1.1d9d4c0000000p-4, 0x1.0cb91c0000000p-5, -0x1.cdda360000000p-5, -0x1.fbdcfc0000000p-4, 0x1.49192e0000000p-3, 0x1.82b67c0000000p-4, 0x1.7ef6c40000000p-4, 0x1.a3c2f20000000p-8, 0x1.1468ba0000000p-5, 0x1.460b9a0000000p-4, 0x1.db5c100000000p-4, -0x1.01e0ee0000000p-3, 0x1.1985ba0000000p-3, -0x1.f45ed40000000p-9, -0x1.b32c100000000p-10, -0x1.28e4b80000000p-6, 0x1.17893e0000000p-3, -0x1.0f11200000000p-6, 0x1.a88aba0000000p-5, -0x1.0bbc0c0000000p-4, 0x1.36cde60000000p-5, -0x1.e38c240000000p-8, 0x1.ebc53e0000000p-6, 0x1.58f9700000000p-5, -0x1.ac8c320000000p-4, 0x1.a329c40000000p-6, -0x1.5a7bfe0000000p-5, -0x1.ac991c0000000p-5, 0x1.dc99940000000p-5, 0x1.160dc00000000p-2, 0x1.5d09220000000p-4, 0x1.9aa3f60000000p-6, -0x1.7c936c0000000p-4, -0x1.1fd31e0000000p-5, 0x1.9e67940000000p-3, -0x1.1a61900000000p-5, -0x1.7cc3e20000000p-4, 0x1.af00940000000p-3, 0x1.6cfe940000000p-3, -0x1.f60c2c0000000p-5, 0x1.9907300000000p-4, -0x1.ba3ba20000000p-4, -0x1.e32ea20000000p-7, -0x1.5684340000000p-9, -0x1.0134a00000000p-3, 0x1.59c98e0000000p-3, 0x1.734e140000000p-2, -0x1.e1bf900000000p-5, 0x1.483e4a0000000p-3, -0x1.1d96a80000000p-3, 0x1.fc9a380000000p-4, 0x1.a296180000000p-9, 0x1.f5d16c0000000p-4, 0x1.dd9cce0000000p-5, 0x1.db04540000000p-5, 0x1.6ed83e0000000p-3, -0x1.f8ebc60000000p-9, 0x1.9e312e0000000p-2, -0x1.9227360000000p-5, -0x1.8546640000000p-6, -0x1.2856b80000000p-4}
}
, {{0x1.7b8e4e0000000p-6, 0x1.7f09860000000p-3, -0x1.9bd60e0000000p-6, -0x1.af39300000000p-7, -0x1.28391c0000000p-5, 0x1.97ac740000000p-3, 0x1.5e9dee0000000p-3, 0x1.0730fa0000000p-6, -0x1.1eb6de0000000p-6, 0x1.5e96880000000p-4, 0x1.3054fa0000000p-4, -0x1.1b38a40000000p-3, 0x1.46179c0000000p-5, 0x1.33edb20000000p-3, -0x1.9736520000000p-5, 0x1.e1768a0000000p-4, 0x1.2c78ec0000000p-3, -0x1.5ccc820000000p-4, 0x1.fdb9f00000000p-5, -0x1.78701a0000000p-7, 0x1.e8efac0000000p-3, 0x1.8a9e7a0000000p-6, 0x1.9f6bb60000000p-4, 0x1.25fcce0000000p-7, 0x1.d5700c0000000p-5, 0x1.6061e00000000p-6, -0x1.265b800000000p-7, 0x1.ba812e0000000p-4, 0x1.9fdfd40000000p-3, 0x1.22ff520000000p-4, -0x1.84e5fc0000000p-5, 0x1.079b7a0000000p-3, -0x1.16f4800000000p-5, 0x1.46432a0000000p-4, 0x1.0032340000000p-3, -0x1.bd834c0000000p-4, -0x1.2213bc0000000p-3, 0x1.4656d40000000p-5, 0x1.0daf000000000p-2, 0x1.1dd1b00000000p-3, 0x1.f34df40000000p-5, 0x1.214ae00000000p-4, 0x1.d036240000000p-4, 0x1.a788120000000p-4, -0x1.2b18160000000p-6, 0x1.79b9d40000000p-5, -0x1.1d68520000000p-4, -0x1.53d2da0000000p-4, 0x1.ac62300000000p-5, 0x1.78647e0000000p-5, 0x1.0c70a40000000p-5, 0x1.2d92ea0000000p-4, 0x1.68b4860000000p-4, -0x1.b053120000000p-7, 0x1.a8528a0000000p-4, -0x1.4810680000000p-4, 0x1.f8abde0000000p-4, 0x1.cca47a0000000p-6, 0x1.84f8700000000p-4, 0x1.7336fe0000000p-5, -0x1.8126120000000p-4, 0x1.65d7fa0000000p-5, 0x1.85ff180000000p-3, 0x1.6fd6ea0000000p-4}
, {-0x1.c91c120000000p-5, 0x1.d0eb4c0000000p-3, 0x1.48df520000000p-4, 0x1.ad2aa80000000p-5, -0x1.8e8d860000000p-6, 0x1.0d59900000000p-5, 0x1.9c41b00000000p-6, 0x1.5f0e8c0000000p-4, -0x1.1ad83c0000000p-6, 0x1.c765e00000000p-4, 0x1.af21680000000p-4, 0x1.a764ce0000000p-5, 0x1.1e87ea0000000p-7, 0x1.7119b60000000p-4, 0x1.2d33620000000p-5, -0x1.69432c0000000p-5, -0x1.5940e00000000p-6, -0x1.c732e20000000p-3, 0x1.eb1b980000000p-4, -0x1.4aa3180000000p-7, -0x1.aedb4e0000000p-4, -0x1.5092fc0000000p-4, 0x1.f271540000000p-4, 0x1.19aa660000000p-2, 0x1.9075ae0000000p-5, -0x1.0dad3a0000000p-4, 0x1.2fe4140000000p-4, 0x1.802fbc0000000p-4, -0x1.fcaca40000000p-6, -0x1.fc022e0000000p-5, 0x1.0dc3bc0000000p-7, 0x1.3940d40000000p-4, -0x1.3fbca40000000p-5, 0x1.2cd2860000000p-4, 0x1.4ccbd60000000p-7, -0x1.88ee160000000p-6, -0x1.b7b7d40000000p-4, 0x1.151f100000000p-6, 0x1.0c86b00000000p-2, 0x1.2da0ea0000000p-4, 0x1.8aaffe0000000p-3, 0x1.224b480000000p-3, 0x1.6f42640000000p-6, -0x1.34cd680000000p-9, 0x1.27777e0000000p-6, 0x1.d5474c0000000p-8, 0x1.1b47340000000p-4, -0x1.29e9860000000p-4, 0x1.22a1be0000000p-4, 0x1.79fbc20000000p-6, -0x1.85904a0000000p-7, -0x1.7cc2f60000000p-5, -0x1.660b880000000p-4, 0x1.1d0fb20000000p-4, 0x1.27f16a0000000p-3, 0x1.0b9a460000000p-5, 0x1.cd69420000000p-4, 0x1.1006420000000p-4, 0x1.1b697c0000000p-3, 0x1.5ca51c0000000p-9, -0x1.30d9020000000p-3, 0x1.9d6eea0000000p-6, 0x1.84db860000000p-4, -0x1.bebf440000000p-4}
, {0x1.1388f60000000p-5, 0x1.cd26de0000000p-4, 0x1.07fe7a0000000p-5, 0x1.796a5e0000000p-3, -0x1.7ed71c0000000p-8, 0x1.bff7760000000p-5, 0x1.618e8e0000000p-4, 0x1.583bb00000000p-3, -0x1.33d8100000000p-6, 0x1.786ea80000000p-3, 0x1.471f180000000p-4, 0x1.a395060000000p-4, -0x1.452d2c0000000p-3, 0x1.8a4c220000000p-5, -0x1.f60cfc0000000p-4, -0x1.47c9be0000000p-4, 0x1.610cd20000000p-5, -0x1.7476880000000p-3, 0x1.56ebbc0000000p-8, 0x1.eabaa80000000p-5, 0x1.70dd6a0000000p-8, -0x1.29a6440000000p-3, 0x1.a200e00000000p-4, 0x1.61cbfa0000000p-3, 0x1.6446440000000p-10, -0x1.a06d900000000p-4, 0x1.3e09180000000p-4, 0x1.1521580000000p-3, -0x1.9ba8da0000000p-3, -0x1.8688aa0000000p-3, 0x1.6281b40000000p-3, 0x1.f637640000000p-4, 0x1.dfd5620000000p-7, -0x1.468b120000000p-4, -0x1.54e50e0000000p-3, -0x1.5f30740000000p-3, -0x1.94a3b60000000p-5, -0x1.3093ca0000000p-4, 0x1.88b49e0000000p-3, -0x1.9d18c80000000p-4, 0x1.4fe4cc0000000p-4, 0x1.0a81cc0000000p-2, -0x1.e3d4460000000p-10, 0x1.01b28a0000000p-3, 0x1.0b625e0000000p-3, -0x1.01fae00000000p-3, 0x1.28622a0000000p-6, -0x1.f06d680000000p-8, -0x1.0bf95a0000000p-3, -0x1.8dee200000000p-5, 0x1.c119080000000p-4, -0x1.51ff2a0000000p-4, -0x1.27e5540000000p-5, 0x1.7aaf840000000p-4, -0x1.07c4280000000p-5, 0x1.ae36e60000000p-4, -0x1.38d5a20000000p-7, 0x1.2e637e0000000p-6, 0x1.9b28a40000000p-4, 0x1.12b0980000000p-2, -0x1.35a5c00000000p-4, 0x1.9588060000000p-4, 0x1.ae9f2a0000000p-3, -0x1.aa03200000000p-12}
}
, {{0x1.be2af00000000p-3, -0x1.353cae0000000p-5, 0x1.2c182e0000000p-4, -0x1.3757ec0000000p-6, -0x1.29658a0000000p-4, -0x1.5473920000000p-3, -0x1.04e8340000000p-4, 0x1.41b3fc0000000p-7, 0x1.4a94660000000p-4, -0x1.8a43a40000000p-4, 0x1.7c5fe40000000p-3, 0x1.29082e0000000p-2, -0x1.5f6d200000000p-3, -0x1.08bc840000000p-2, 0x1.8b0cf60000000p-4, 0x1.0ef0ec0000000p-3, 0x1.7834440000000p-4, 0x1.71d76a0000000p-7, -0x1.1f84ae0000000p-5, 0x1.d0fc2c0000000p-6, -0x1.ee6a840000000p-4, 0x1.6e47ee0000000p-4, -0x1.6d49e40000000p-3, -0x1.515b380000000p-7, -0x1.21b9760000000p-3, 0x1.29e4380000000p-6, 0x1.00fb4a0000000p-4, -0x1.1ecc000000000p-5, 0x1.9d79500000000p-4, -0x1.ce8e8e0000000p-3, 0x1.fc280e0000000p-3, -0x1.dfa8ca0000000p-4, -0x1.1a0b720000000p-4, 0x1.2b182e0000000p-3, -0x1.115cec0000000p-5, 0x1.117f260000000p-5, 0x1.1ba6a80000000p-4, -0x1.68d2920000000p-3, -0x1.2550080000000p-2, 0x1.08b2920000000p-4, -0x1.c1d4c40000000p-6, 0x1.5f15500000000p-6, -0x1.8619980000000p-4, -0x1.ef074c0000000p-3, -0x1.db6f080000000p-5, -0x1.2189a40000000p-5, -0x1.a8c5fa0000000p-7, 0x1.187fa00000000p-3, 0x1.44b39c0000000p-3, -0x1.23a0ce0000000p-9, 0x1.7176700000000p-4, 0x1.682a800000000p-4, 0x1.073ef00000000p-3, -0x1.0892ae0000000p-7, -0x1.0765c00000000p-2, -0x1.f3c3420000000p-8, 0x1.9378dc0000000p-6, 0x1.39faee0000000p-3, 0x1.601faa0000000p-5, -0x1.0916da0000000p-2, 0x1.7e4bc00000000p-3, 0x1.46f9560000000p-1, -0x1.03ae800000000p-4, -0x1.53fb900000000p-2}
, {-0x1.25ff660000000p-3, -0x1.f778a20000000p-4, -0x1.1475a80000000p-7, -0x1.1498980000000p-3, 0x1.e870a60000000p-4, 0x1.130b2a0000000p-4, -0x1.40da680000000p-3, -0x1.dd2eda0000000p-5, -0x1.b7c18e0000000p-4, 0x1.3e3c7e0000000p-7, -0x1.45870a0000000p-4, 0x1.93e9800000000p-3, -0x1.e90eb00000000p-5, 0x1.ba5dd60000000p-5, 0x1.12cce40000000p-6, -0x1.db97360000000p-5, -0x1.3c7f840000000p-3, 0x1.dd67ae0000000p-5, -0x1.a1b3400000000p-7, 0x1.8854b00000000p-3, -0x1.64fc580000000p-8, -0x1.a405ce0000000p-7, -0x1.9e75780000000p-6, -0x1.1b90c20000000p-2, 0x1.8006540000000p-4, 0x1.8fb8f20000000p-4, -0x1.64b3ec0000000p-5, -0x1.4181c80000000p-3, -0x1.c832c40000000p-5, 0x1.0192600000000p-3, 0x1.4e748a0000000p-5, -0x1.3cb6bc0000000p-5, 0x1.8bddba0000000p-5, -0x1.9513240000000p-4, -0x1.dfdfc20000000p-4, 0x1.2e01320000000p-4, 0x1.0dd0240000000p-3, 0x1.01bb860000000p-4, -0x1.22aa640000000p-5, -0x1.cf2fdc0000000p-5, 0x1.c620820000000p-6, -0x1.f3af480000000p-3, -0x1.6c82800000000p-4, 0x1.22c1880000000p-5, -0x1.3e06fc0000000p-7, -0x1.08d4d80000000p-6, -0x1.022b520000000p-3, -0x1.2a16a20000000p-3, 0x1.299f360000000p-5, -0x1.0d87820000000p-4, 0x1.9ee1380000000p-6, 0x1.a59ee40000000p-6, -0x1.c992ba0000000p-3, 0x1.4719bc0000000p-4, -0x1.21e0aa0000000p-3, 0x1.5b93020000000p-4, 0x1.39e5de0000000p-5, 0x1.65bdec0000000p-9, -0x1.3cba440000000p-5, -0x1.1f49d40000000p-3, 0x1.7ac3040000000p-5, -0x1.728ace0000000p-4, -0x1.92ba060000000p-3, 0x1.864c020000000p-5}
, {-0x1.84532a0000000p-4, -0x1.e7bf460000000p-4, 0x1.db7c5a0000000p-7, -0x1.bb7b8a0000000p-6, -0x1.8c84600000000p-6, 0x1.a6c8a40000000p-3, -0x1.7715d00000000p-4, 0x1.41761c0000000p-9, -0x1.7163c60000000p-4, 0x1.667fea0000000p-4, 0x1.de8e9e0000000p-4, 0x1.36e6760000000p-2, -0x1.2480c40000000p-5, -0x1.3391de0000000p-3, -0x1.cdce6e0000000p-3, -0x1.ca99ec0000000p-5, -0x1.4078540000000p-5, 0x1.4f316a0000000p-3, -0x1.60af620000000p-8, 0x1.f2a8880000000p-7, 0x1.27be440000000p-4, 0x1.2139520000000p-3, -0x1.28f0620000000p-4, 0x1.290db40000000p-3, 0x1.69129e0000000p-4, 0x1.c6f6360000000p-7, 0x1.02abce0000000p-4, -0x1.75e55c0000000p-10, -0x1.b254820000000p-7, 0x1.af87e40000000p-3, 0x1.6f94ea0000000p-5, 0x1.0cf3680000000p-5, 0x1.0e9eac0000000p-3, 0x1.826a2a0000000p-5, 0x1.7a86c20000000p-4, 0x1.4f2c1e0000000p-3, -0x1.0a519a0000000p-5, 0x1.1d24a80000000p-3, 0x1.453bf00000000p-6, 0x1.3dbc8c0000000p-7, 0x1.265f100000000p-3, -0x1.6626820000000p-4, 0x1.5a37240000000p-5, -0x1.5fe9f40000000p-5, 0x1.4033fc0000000p-5, -0x1.fce20c0000000p-4, 0x1.79bcca0000000p-5, 0x1.0f568c0000000p-9, -0x1.74cd5c0000000p-4, -0x1.caaddc0000000p-4, 0x1.0e33ce0000000p-3, -0x1.0823c60000000p-2, 0x1.d943840000000p-5, 0x1.b332880000000p-3, 0x1.1127100000000p-3, -0x1.5e4df20000000p-5, 0x1.b246f00000000p-9, 0x1.6550c40000000p-5, 0x1.820abe0000000p-4, 0x1.05c46c0000000p-2, -0x1.7a0d4e0000000p-4, -0x1.ad40160000000p-5, 0x1.ea7e280000000p-5, 0x1.4709da0000000p-5}
}
, {{0x1.1672300000000p-7, -0x1.74719a0000000p-6, 0x1.9bdbd40000000p-6, 0x1.ec00b40000000p-5, 0x1.1ab2420000000p-4, 0x1.4bc35a0000000p-4, 0x1.9e30f20000000p-6, -0x1.39f6c80000000p-3, -0x1.980a0e0000000p-4, 0x1.cbbee80000000p-3, -0x1.2a8b960000000p-4, 0x1.48c4c60000000p-3, 0x1.57660e0000000p-3, 0x1.1510d60000000p-3, -0x1.9f5eac0000000p-4, -0x1.860cb20000000p-4, 0x1.e8b1380000000p-5, -0x1.ab1de40000000p-5, -0x1.0e633e0000000p-5, 0x1.db555e0000000p-6, -0x1.0570b80000000p-9, 0x1.0e7bac0000000p-5, 0x1.c3af700000000p-4, 0x1.b74bc60000000p-3, -0x1.c4b9f20000000p-3, 0x1.4cc4c00000000p-3, -0x1.1529e40000000p-4, 0x1.2640760000000p-4, -0x1.33ec9a0000000p-5, -0x1.24137a0000000p-3, 0x1.cb1c3c0000000p-5, 0x1.f84ef40000000p-5, 0x1.cebb3e0000000p-8, -0x1.e131900000000p-4, -0x1.12f7480000000p-3, -0x1.04ba7c0000000p-3, -0x1.cf2d940000000p-5, 0x1.48b03e0000000p-4, 0x1.56e6980000000p-4, 0x1.5fefde0000000p-7, 0x1.9f290a0000000p-3, -0x1.7fb29a0000000p-4, -0x1.61d80a0000000p-3, 0x1.9506ca0000000p-4, -0x1.d5acaa0000000p-3, -0x1.9adb900000000p-5, -0x1.b8715a0000000p-6, -0x1.c2536e0000000p-4, -0x1.e0e8a80000000p-4, -0x1.089af80000000p-3, -0x1.1ce7dc0000000p-3, 0x1.f363960000000p-4, -0x1.a3fed80000000p-5, 0x1.0c3d460000000p-5, 0x1.d5c7840000000p-8, -0x1.cf3b520000000p-3, -0x1.66b8320000000p-4, 0x1.92b7a60000000p-4, -0x1.aa4b860000000p-7, 0x1.6238ce0000000p-6, 0x1.f012560000000p-5, -0x1.e29f4a0000000p-8, -0x1.3e18ca0000000p-3, -0x1.0659ea0000000p-4}
, {-0x1.2044ea0000000p-3, -0x1.7123bc0000000p-7, 0x1.1526ac0000000p-3, -0x1.9c08c80000000p-4, -0x1.15bfb60000000p-4, 0x1.324e9a0000000p-3, 0x1.9c41d60000000p-7, 0x1.1d54160000000p-2, -0x1.939ec60000000p-4, 0x1.505be40000000p-4, -0x1.8f65ba0000000p-4, -0x1.5e01740000000p-10, -0x1.aa1fa80000000p-7, 0x1.7b96a80000000p-4, 0x1.b59b160000000p-6, -0x1.c68bc00000000p-4, 0x1.c4ac520000000p-6, -0x1.4f91160000000p-4, 0x1.f9da5a0000000p-5, -0x1.ef14b60000000p-6, 0x1.e532b60000000p-11, -0x1.2d3f0a0000000p-5, 0x1.cb29320000000p-5, -0x1.27a5280000000p-8, -0x1.b197260000000p-7, 0x1.f1e9dc0000000p-5, 0x1.2b98960000000p-4, -0x1.28326e0000000p-7, -0x1.c5c1a60000000p-4, 0x1.66df840000000p-4, -0x1.dcfc5c0000000p-6, 0x1.810fac0000000p-3, 0x1.6ff2040000000p-3, 0x1.390cd80000000p-8, 0x1.f439340000000p-4, 0x1.4677820000000p-2, 0x1.cc6e8c0000000p-5, -0x1.e1a4d60000000p-5, 0x1.1a47ce0000000p-5, 0x1.e890d60000000p-3, -0x1.85b0720000000p-3, 0x1.6443200000000p-4, -0x1.b055d20000000p-7, -0x1.5896c80000000p-3, -0x1.87623c0000000p-3, 0x1.8ab08e0000000p-9, 0x1.8c6d480000000p-6, -0x1.20a6480000000p-5, 0x1.78eeb80000000p-3, 0x1.baf1860000000p-4, 0x1.4048ae0000000p-3, -0x1.c720060000000p-7, -0x1.dc90b60000000p-6, 0x1.a3d9ec0000000p-5, -0x1.8cafc00000000p-5, 0x1.b5c55e0000000p-4, -0x1.0e0ce40000000p-3, -0x1.3549320000000p-4, 0x1.27cd620000000p-4, -0x1.820f780000000p-3, -0x1.2a65b00000000p-5, 0x1.2b533e0000000p-3, 0x1.e3d9200000000p-7, -0x1.38107e0000000p-4}
, {-0x1.98de5c0000000p-5, -0x1.c3785c0000000p-3, -0x1.5ce93a0000000p-2, 0x1.62f8f60000000p-5, -0x1.5171080000000p-4, 0x1.03f3800000000p-5, 0x1.14b8a80000000p-8, -0x1.4017d00000000p-3, -0x1.478b280000000p-2, -0x1.a5b9340000000p-3, -0x1.1d4d0c0000000p-3, -0x1.e6fcac0000000p-4, -0x1.2169540000000p-4, 0x1.190b660000000p-3, -0x1.2fd2c20000000p-3, -0x1.d36e120000000p-5, 0x1.43d61c0000000p-4, -0x1.b31e500000000p-4, -0x1.71e76c0000000p-5, -0x1.9450800000000p-4, -0x1.ead98a0000000p-4, 0x1.a8090e0000000p-4, -0x1.53fd180000000p-3, -0x1.fc9e9a0000000p-6, -0x1.00086e0000000p-5, -0x1.12a1020000000p-4, -0x1.8480c40000000p-6, 0x1.75f6d60000000p-5, 0x1.bfedc00000000p-5, 0x1.7420d60000000p-4, 0x1.4bf1ca0000000p-8, 0x1.af04020000000p-4, 0x1.1c6a7c0000000p-5, -0x1.39eedc0000000p-4, -0x1.eedda80000000p-6, 0x1.a8c0180000000p-5, -0x1.cfb41c0000000p-5, 0x1.6c4ca80000000p-4, -0x1.08790c0000000p-3, -0x1.7444600000000p-4, 0x1.2459bc0000000p-3, 0x1.c8c6be0000000p-3, -0x1.4626020000000p-3, -0x1.357bd40000000p-6, 0x1.3fa3e40000000p-5, -0x1.9bb06a0000000p-3, 0x1.ca5ce20000000p-7, 0x1.2811d20000000p-5, 0x1.09ef800000000p-5, 0x1.17ef5e0000000p-3, -0x1.25a2b40000000p-4, -0x1.ec74300000000p-3, -0x1.8defde0000000p-5, -0x1.34b4f40000000p-5, -0x1.418c9e0000000p-3, 0x1.ef526c0000000p-3, 0x1.955b2a0000000p-3, -0x1.2809160000000p-2, 0x1.3232940000000p-4, -0x1.b4305e0000000p-5, 0x1.dfe77e0000000p-4, -0x1.8d94680000000p-4, -0x1.f17fdc0000000p-4, -0x1.d321680000000p-9}
}
, {{-0x1.2321680000000p-5, 0x1.2a9c7c0000000p-3, -0x1.b46fd40000000p-3, -0x1.4262720000000p-3, 0x1.0edd040000000p-3, 0x1.211d9e0000000p-3, 0x1.1475b20000000p-4, -0x1.387a120000000p-2, -0x1.f680c40000000p-3, -0x1.80df980000000p-4, 0x1.7e70b80000000p-4, -0x1.cd06920000000p-4, 0x1.7ca75c0000000p-4, -0x1.1f6ca00000000p-11, -0x1.6637480000000p-3, -0x1.d5b2b80000000p-3, 0x1.bf522a0000000p-4, -0x1.aa5bc60000000p-5, 0x1.e2cfaa0000000p-3, 0x1.e186800000000p-4, 0x1.18059a0000000p-10, 0x1.0ed2260000000p-6, 0x1.92f90c0000000p-6, 0x1.0db3d80000000p-2, -0x1.9da76e0000000p-3, 0x1.2c94dc0000000p-4, 0x1.18a1520000000p-4, 0x1.e9036e0000000p-4, 0x1.2077ca0000000p-4, -0x1.3df3200000000p-6, -0x1.d453c40000000p-6, 0x1.39c13e0000000p-3, -0x1.75748c0000000p-5, 0x1.385ca20000000p-5, 0x1.2dfee40000000p-3, -0x1.b6a3a80000000p-4, -0x1.199f280000000p-4, 0x1.0a87b40000000p-4, 0x1.0310280000000p-3, 0x1.d739e80000000p-5, 0x1.106d380000000p-2, -0x1.26b2500000000p-3, -0x1.a262220000000p-6, -0x1.b38bc60000000p-4, -0x1.eea0980000000p-5, -0x1.4edc560000000p-2, 0x1.54d23e0000000p-3, -0x1.ffa99a0000000p-3, 0x1.0f5f220000000p-5, -0x1.a508920000000p-7, -0x1.b5ca740000000p-5, 0x1.3d5eec0000000p-4, 0x1.dd5ab00000000p-4, -0x1.44400a0000000p-6, 0x1.3a97fe0000000p-3, 0x1.60c1a40000000p-7, 0x1.b6a2e20000000p-4, -0x1.92de5c0000000p-6, 0x1.a596c60000000p-3, -0x1.29d72a0000000p-7, -0x1.dcb06a0000000p-5, -0x1.3e62040000000p-8, -0x1.e599ee0000000p-4, -0x1.67cc120000000p-3}
, {-0x1.3409320000000p-4, 0x1.c7dc1e0000000p-10, -0x1.47ab3a0000000p-5, -0x1.26fcc40000000p-3, -0x1.08a7f20000000p-4, -0x1.155ac00000000p-6, 0x1.95007c0000000p-4, 0x1.3f48200000000p-6, -0x1.b07e420000000p-6, 0x1.293f7e0000000p-10, 0x1.307d900000000p-3, -0x1.e382b60000000p-8, -0x1.cc32440000000p-5, -0x1.a0e7760000000p-4, 0x1.c18da40000000p-6, 0x1.8c843c0000000p-7, -0x1.e9c3dc0000000p-5, 0x1.ee3b980000000p-5, -0x1.5b813e0000000p-6, 0x1.e3d69c0000000p-4, 0x1.fa321a0000000p-7, 0x1.533d620000000p-9, -0x1.a877c20000000p-6, -0x1.737d820000000p-11, 0x1.38a5260000000p-5, -0x1.8455fa0000000p-7, 0x1.4d375e0000000p-5, -0x1.56b8f60000000p-3, -0x1.61653e0000000p-3, 0x1.10a16e0000000p-7, 0x1.6f9ffc0000000p-5, -0x1.5fa8c20000000p-2, -0x1.ea08560000000p-4, -0x1.5f272e0000000p-7, -0x1.66d5640000000p-8, -0x1.d1d0c20000000p-6, -0x1.1c7e6e0000000p-3, -0x1.53035c0000000p-5, -0x1.0cc4360000000p-3, 0x1.bcaf020000000p-4, -0x1.2b46960000000p-4, -0x1.44f3da0000000p-3, -0x1.3980120000000p-3, -0x1.958d8c0000000p-7, 0x1.a1fcce0000000p-4, -0x1.4628340000000p-4, -0x1.a4dcc60000000p-4, -0x1.bac4f20000000p-4, -0x1.7bad720000000p-6, -0x1.3087ac0000000p-3, -0x1.2e15060000000p-3, -0x1.1707780000000p-3, -0x1.53fdb80000000p-5, -0x1.355d5a0000000p-5, 0x1.c988b40000000p-5, -0x1.5c4c5a0000000p-3, -0x1.a537140000000p-3, 0x1.815c080000000p-5, -0x1.350cfe0000000p-3, -0x1.1d4baa0000000p-2, -0x1.ee624c0000000p-5, -0x1.8faad20000000p-3, -0x1.d2d1d80000000p-4, -0x1.edcb560000000p-4}
, {0x1.31a2100000000p-4, -0x1.94aed00000000p-3, -0x1.c5fc420000000p-8, 0x1.9132020000000p-2, -0x1.99ef7c0000000p-3, -0x1.96dbb80000000p-3, 0x1.03cf940000000p-4, 0x1.cba8c60000000p-8, 0x1.3b675c0000000p-4, 0x1.d7cfcc0000000p-4, -0x1.97c4400000000p-3, 0x1.1031580000000p-4, -0x1.bbc2b20000000p-8, -0x1.c738dc0000000p-3, 0x1.b3a2520000000p-8, 0x1.0aa06a0000000p-9, -0x1.2124040000000p-3, 0x1.1150020000000p-2, -0x1.82f2500000000p-2, 0x1.7eadea0000000p-3, -0x1.8a7ebe0000000p-5, -0x1.d0f96e0000000p-4, 0x1.4228860000000p-3, 0x1.2439640000000p-5, -0x1.5b4c580000000p-2, -0x1.bf7f860000000p-5, 0x1.4109280000000p-3, 0x1.b486480000000p-5, -0x1.4eba920000000p-2, -0x1.4803c00000000p-4, -0x1.5d602c0000000p-5, -0x1.11bc180000000p-3, 0x1.ed85560000000p-4, -0x1.b59dc00000000p-4, -0x1.937b4c0000000p-3, -0x1.83fe980000000p-4, 0x1.476ffa0000000p-3, -0x1.9f8e9c0000000p-2, -0x1.160f4a0000000p-2, -0x1.2089000000000p-2, -0x1.5093e60000000p-3, -0x1.37ccbe0000000p-5, 0x1.f4505e0000000p-4, -0x1.6115780000000p-3, 0x1.0539320000000p-6, 0x1.5c4ad80000000p-3, -0x1.dc54380000000p-3, -0x1.07751c0000000p-5, -0x1.9d1ea60000000p-2, 0x1.3e20da0000000p-3, -0x1.ee4a1a0000000p-7, -0x1.32247a0000000p-4, -0x1.3714180000000p-5, 0x1.8b37780000000p-7, -0x1.77aed40000000p-3, -0x1.67c13e0000000p-5, -0x1.4a89340000000p-2, 0x1.1f83220000000p-4, -0x1.fac54e0000000p-3, 0x1.b0e0760000000p-6, -0x1.85156c0000000p-4, -0x1.e378080000000p-4, -0x1.2d29840000000p-5, -0x1.a9bb3a0000000p-4}
}
, {{-0x1.aae4ba0000000p-4, -0x1.1c2c640000000p-3, -0x1.b53a640000000p-7, 0x1.431c460000000p-5, 0x1.55526c0000000p-6, 0x1.2f889c0000000p-6, 0x1.16d4480000000p-3, -0x1.468bf20000000p-3, -0x1.72261e0000000p-4, 0x1.18c43c0000000p-5, -0x1.729b6a0000000p-5, 0x1.555d100000000p-7, -0x1.0549940000000p-6, -0x1.1873be0000000p-4, 0x1.230dbe0000000p-3, -0x1.1f5eae0000000p-4, 0x1.7cc8b60000000p-4, -0x1.6020580000000p-4, 0x1.a914fa0000000p-3, 0x1.7b56380000000p-5, 0x1.af8fc40000000p-7, 0x1.744b5e0000000p-3, -0x1.1792ea0000000p-3, 0x1.48d9c80000000p-3, 0x1.3276420000000p-5, 0x1.6cbb2c0000000p-7, 0x1.64e6ce0000000p-4, 0x1.1b4fd80000000p-3, -0x1.78e2ae0000000p-5, -0x1.09bfc60000000p-4, -0x1.91fffe0000000p-3, 0x1.2c7d780000000p-3, 0x1.f6861e0000000p-5, -0x1.0e75c20000000p-4, -0x1.bad2280000000p-3, -0x1.e154ce0000000p-3, 0x1.092e400000000p-2, 0x1.5af3f00000000p-3, 0x1.3fc9ae0000000p-4, -0x1.6c2dfa0000000p-6, -0x1.46106a0000000p-6, -0x1.7f727a0000000p-4, 0x1.35a39c0000000p-3, 0x1.1862e00000000p-4, 0x1.caf7f40000000p-6, 0x1.5605340000000p-3, -0x1.fe3c960000000p-6, -0x1.d8f8040000000p-4, 0x1.aa07780000000p-4, -0x1.a0069e0000000p-3, 0x1.6c0e520000000p-5, -0x1.d461660000000p-6, 0x1.79ddca0000000p-5, 0x1.a4111c0000000p-5, 0x1.ee52ba0000000p-4, -0x1.89bb2e0000000p-5, 0x1.e9384c0000000p-4, -0x1.bc04980000000p-4, 0x1.a00fb60000000p-3, 0x1.c82f1a0000000p-3, 0x1.2351740000000p-4, 0x1.5480b00000000p-4, 0x1.170c440000000p-2, 0x1.4440d40000000p-3}
, {-0x1.60f7e20000000p-3, 0x1.1dbeaa0000000p-3, 0x1.3a01ac0000000p-5, -0x1.b2f1420000000p-4, -0x1.f9f5fe0000000p-4, 0x1.30f6080000000p-4, 0x1.162e080000000p-4, 0x1.758afa0000000p-4, -0x1.8330640000000p-10, 0x1.b9634e0000000p-7, -0x1.0f5dfe0000000p-5, 0x1.4cc04e0000000p-5, -0x1.e547e20000000p-6, -0x1.9c1c940000000p-6, 0x1.68d8a60000000p-4, 0x1.c637ca0000000p-5, 0x1.aded3e0000000p-6, 0x1.2845fa0000000p-7, -0x1.9d801e0000000p-7, -0x1.8a44dc0000000p-4, -0x1.558f440000000p-3, -0x1.3c2a300000000p-5, -0x1.18e97a0000000p-5, 0x1.3e9f9c0000000p-4, 0x1.d2bb760000000p-5, -0x1.bd193a0000000p-7, 0x1.6128680000000p-5, 0x1.7ace900000000p-3, -0x1.1552b80000000p-5, 0x1.a061b20000000p-5, -0x1.53e7c20000000p-4, 0x1.06151c0000000p-5, -0x1.dc8dca0000000p-3, -0x1.1ea8f60000000p-4, -0x1.499e540000000p-5, -0x1.8044820000000p-4, -0x1.fef6100000000p-4, -0x1.a2f5e00000000p-11, -0x1.9537ac0000000p-3, 0x1.43e22c0000000p-4, -0x1.a4fbb00000000p-4, 0x1.8f79680000000p-4, -0x1.da3a780000000p-4, -0x1.4905ce0000000p-7, -0x1.3985000000000p-3, -0x1.1118b80000000p-3, 0x1.f2cc600000000p-4, 0x1.179de80000000p-3, -0x1.60de580000000p-4, -0x1.ceb5980000000p-5, 0x1.b5ce1c0000000p-5, 0x1.bb47260000000p-4, -0x1.0407e40000000p-4, -0x1.608a220000000p-8, -0x1.a2c21e0000000p-6, -0x1.27f8ee0000000p-3, 0x1.df39a20000000p-7, 0x1.37115e0000000p-4, 0x1.919e0c0000000p-5, -0x1.b454ee0000000p-3, -0x1.3dbcb20000000p-4, 0x1.4240b40000000p-8, 0x1.91837c0000000p-4, 0x1.7363360000000p-4}
, {0x1.bf037a0000000p-4, -0x1.96fa600000000p-5, 0x1.6a02400000000p-5, 0x1.9bc75a0000000p-3, 0x1.fdbfbc0000000p-3, -0x1.d523cc0000000p-5, 0x1.2ebda20000000p-2, 0x1.74b6920000000p-7, 0x1.33fd6a0000000p-7, 0x1.5f4cb00000000p-5, -0x1.393b7a0000000p-8, 0x1.9ff1240000000p-5, -0x1.7b1a060000000p-3, -0x1.3a315c0000000p-3, -0x1.fc1c080000000p-7, -0x1.abc8b80000000p-3, -0x1.1622680000000p-4, 0x1.bd55640000000p-4, -0x1.c6a84a0000000p-3, 0x1.e791ba0000000p-3, -0x1.e71d8c0000000p-4, 0x1.1f2d3a0000000p-3, 0x1.9bdbd40000000p-3, 0x1.2fb7b00000000p-3, -0x1.ea3fd60000000p-4, 0x1.dfdd760000000p-4, 0x1.6cd22c0000000p-4, 0x1.6d763a0000000p-2, -0x1.4da5a80000000p-3, -0x1.7a2a5c0000000p-3, 0x1.1721700000000p-5, -0x1.d7befa0000000p-5, 0x1.adb51a0000000p-4, -0x1.d3f3fe0000000p-5, -0x1.b01dd00000000p-3, -0x1.a827500000000p-4, 0x1.4903a20000000p-3, -0x1.d734640000000p-3, -0x1.ba4e980000000p-5, -0x1.4a22240000000p-4, 0x1.086e1c0000000p-2, -0x1.5537620000000p-3, 0x1.301e020000000p-3, -0x1.6a52d60000000p-12, 0x1.ce4cf00000000p-5, 0x1.824c760000000p-5, -0x1.9b1aa00000000p-4, 0x1.ac00200000000p-3, -0x1.f3c3b20000000p-3, 0x1.da97c00000000p-7, 0x1.fac4cc0000000p-5, 0x1.6c727e0000000p-4, -0x1.678d980000000p-8, 0x1.c3170c0000000p-4, 0x1.1b39820000000p-4, -0x1.8245a00000000p-4, 0x1.3fdf280000000p-5, 0x1.4c51f20000000p-5, -0x1.b1ffd00000000p-4, 0x1.364b0c0000000p-3, -0x1.9914b40000000p-5, -0x1.6a14340000000p-3, 0x1.1b581a0000000p-2, -0x1.b84d800000000p-6}
}
, {{-0x1.18566e0000000p-5, -0x1.11d9540000000p-2, -0x1.fbf1200000000p-6, -0x1.12d3ce0000000p-6, -0x1.ade54c0000000p-5, -0x1.9156aa0000000p-4, 0x1.ecb7480000000p-4, -0x1.0e6ac80000000p-2, 0x1.635e740000000p-9, -0x1.2514580000000p-2, 0x1.1218b40000000p-5, -0x1.ec8bfc0000000p-5, -0x1.5f43320000000p-8, 0x1.80c5d40000000p-8, -0x1.7998840000000p-7, -0x1.29e5f40000000p-4, -0x1.2aa2720000000p-3, -0x1.f627580000000p-5, -0x1.76da1a0000000p-5, 0x1.37272c0000000p-4, 0x1.7d71660000000p-4, 0x1.4c36be0000000p-6, 0x1.f660de0000000p-5, 0x1.9e8c3a0000000p-3, -0x1.277f0e0000000p-3, -0x1.8d35d20000000p-3, 0x1.cb961e0000000p-3, 0x1.314a800000000p-6, 0x1.69092e0000000p-6, -0x1.22d68a0000000p-4, 0x1.43ce580000000p-6, -0x1.203f160000000p-5, -0x1.80ab5a0000000p-4, -0x1.49e4660000000p-3, -0x1.fc2d320000000p-3, -0x1.0153fe0000000p-4, 0x1.c537660000000p-3, -0x1.2be26c0000000p-4, -0x1.cc0ce80000000p-6, -0x1.56a9f80000000p-3, 0x1.610ac60000000p-2, -0x1.06db0a0000000p-4, -0x1.e6e0000000000p-4, -0x1.88ca180000000p-3, 0x1.1da9ea0000000p-3, -0x1.cd5e420000000p-5, 0x1.840c020000000p-5, -0x1.4a1cd60000000p-5, -0x1.c4b1c80000000p-3, 0x1.023e200000000p-4, 0x1.399d520000000p-6, -0x1.ed20740000000p-3, -0x1.3b11d60000000p-3, -0x1.0c4d7e0000000p-7, 0x1.2835e20000000p-3, -0x1.02d1e20000000p-4, 0x1.27faf80000000p-2, -0x1.7cf2e20000000p-3, 0x1.3a2c640000000p-4, 0x1.3bdd1a0000000p-2, -0x1.3a0a400000000p-4, 0x1.4c98c60000000p-4, -0x1.8ce88e0000000p-3, -0x1.11fad00000000p-6}
, {0x1.d9f8be0000000p-6, -0x1.5ffacc0000000p-4, 0x1.4b435a0000000p-5, -0x1.fb50c40000000p-7, 0x1.607a880000000p-5, -0x1.cdcca40000000p-6, 0x1.761e800000000p-3, 0x1.5598940000000p-4, 0x1.30de000000000p-4, 0x1.0bfa9c0000000p-5, -0x1.111af40000000p-6, 0x1.af05ea0000000p-4, -0x1.13a6260000000p-3, -0x1.417cbe0000000p-4, -0x1.6cd4600000000p-4, 0x1.d13ef00000000p-5, -0x1.e115bc0000000p-5, 0x1.976cc80000000p-4, -0x1.5f32be0000000p-5, -0x1.39da3a0000000p-6, 0x1.f068aa0000000p-8, 0x1.2ac0800000000p-4, 0x1.80d90e0000000p-3, 0x1.24ed3e0000000p-5, -0x1.0c28940000000p-7, -0x1.b6dc740000000p-3, -0x1.88f2000000000p-4, 0x1.12bdbc0000000p-4, -0x1.dab2be0000000p-3, 0x1.6720be0000000p-4, 0x1.262dfa0000000p-4, 0x1.a5f73e0000000p-4, 0x1.e79ccc0000000p-5, -0x1.2dc38a0000000p-3, 0x1.4be7620000000p-4, -0x1.bbcbb60000000p-4, -0x1.62640c0000000p-4, -0x1.4d5c6c0000000p-3, 0x1.10dc4c0000000p-2, -0x1.bb4a040000000p-5, 0x1.48438c0000000p-5, -0x1.d0825a0000000p-6, -0x1.cb536c0000000p-5, -0x1.6d6cb00000000p-3, 0x1.1e92a20000000p-4, -0x1.00e8740000000p-5, -0x1.07af3e0000000p-3, 0x1.651e8a0000000p-8, -0x1.d6cfc20000000p-7, -0x1.5978380000000p-6, 0x1.2ec54e0000000p-3, -0x1.02dba20000000p-5, 0x1.4b322a0000000p-3, 0x1.c34cb20000000p-6, -0x1.3e1abc0000000p-6, -0x1.d117500000000p-4, -0x1.654a720000000p-4, -0x1.23e6b60000000p-3, -0x1.b39cd80000000p-4, -0x1.6235dc0000000p-6, -0x1.2b0bc00000000p-3, 0x1.ef14540000000p-3, 0x1.0dd0860000000p-3, 0x1.6e39620000000p-4}
, {-0x1.2728ba0000000p-7, 0x1.085b480000000p-3, -0x1.84ecf80000000p-3, -0x1.8a187a0000000p-4, 0x1.21cf140000000p-4, 0x1.8d888a0000000p-2, -0x1.0da7480000000p-4, -0x1.d768920000000p-6, -0x1.253e300000000p-4, -0x1.aad5a80000000p-4, -0x1.06aa0c0000000p-4, 0x1.8f10d40000000p-9, -0x1.0dd2140000000p-4, -0x1.486a2e0000000p-5, -0x1.a1dfa60000000p-3, 0x1.94d2c40000000p-5, 0x1.5202560000000p-3, 0x1.5cb1960000000p-3, 0x1.f7c2020000000p-5, 0x1.43a20a0000000p-5, 0x1.c585660000000p-4, -0x1.4371820000000p-5, -0x1.19ca0c0000000p-7, 0x1.ac22760000000p-5, 0x1.ac2afc0000000p-3, -0x1.4206ca0000000p-4, 0x1.4178840000000p-7, -0x1.1f3c900000000p-3, -0x1.3dc4aa0000000p-5, 0x1.07dd7e0000000p-2, -0x1.d15e4c0000000p-7, -0x1.c6dc540000000p-9, 0x1.b89bb00000000p-7, -0x1.1fd8640000000p-4, 0x1.0401640000000p-3, -0x1.3dc5040000000p-5, 0x1.c816720000000p-6, 0x1.5bc0340000000p-4, 0x1.9afd520000000p-4, 0x1.46d8c80000000p-4, 0x1.9faf040000000p-3, -0x1.3c6c920000000p-2, 0x1.4294240000000p-5, -0x1.9ab1ce0000000p-3, -0x1.68ebca0000000p-4, -0x1.3ee6e20000000p-4, -0x1.929d840000000p-6, -0x1.54939a0000000p-8, 0x1.c1e19a0000000p-6, 0x1.b047d80000000p-4, -0x1.3293f20000000p-3, 0x1.305b7e0000000p-5, 0x1.e8757c0000000p-5, 0x1.e088dc0000000p-4, 0x1.c319080000000p-4, 0x1.9de6c60000000p-4, 0x1.29e5320000000p-6, -0x1.86ae1e0000000p-6, 0x1.0d55a20000000p-2, 0x1.548c480000000p-4, -0x1.06a5460000000p-3, -0x1.69b3640000000p-3, 0x1.0a63e60000000p-5, 0x1.b9514a0000000p-5}
}
, {{0x1.47aa620000000p-6, 0x1.286a560000000p-4, 0x1.a292440000000p-6, 0x1.2785c60000000p-6, 0x1.efc0020000000p-6, 0x1.1029180000000p-4, 0x1.fc39020000000p-6, 0x1.4ff3ba0000000p-3, -0x1.b7d3580000000p-5, 0x1.6fa0980000000p-3, 0x1.07c10c0000000p-3, 0x1.a409d60000000p-3, 0x1.0bcf400000000p-3, 0x1.6df26e0000000p-7, -0x1.c921ee0000000p-3, 0x1.00c44a0000000p-2, -0x1.3b68840000000p-14, -0x1.d4070a0000000p-6, 0x1.34e7ca0000000p-3, -0x1.138e620000000p-4, 0x1.0c083a0000000p-4, 0x1.c395280000000p-3, 0x1.5e23780000000p-5, 0x1.b5a76c0000000p-4, 0x1.b226520000000p-5, -0x1.09c76a0000000p-3, 0x1.564ef60000000p-3, 0x1.6ad4d60000000p-4, 0x1.8302300000000p-6, 0x1.2bde5a0000000p-4, -0x1.c7c6b20000000p-6, 0x1.7967e40000000p-4, -0x1.07a3960000000p-5, 0x1.7ff7240000000p-3, 0x1.19a7420000000p-3, -0x1.d6b59e0000000p-4, -0x1.19d2020000000p-4, -0x1.1f475a0000000p-6, 0x1.2f7a2e0000000p-4, -0x1.c24f120000000p-3, 0x1.e7fc280000000p-4, 0x1.9410020000000p-3, 0x1.47e6600000000p-5, 0x1.6b35720000000p-4, 0x1.33d7760000000p-5, 0x1.e8fb4e0000000p-3, 0x1.f8b1ae0000000p-3, 0x1.8bb46c0000000p-4, -0x1.299bd40000000p-4, -0x1.26098c0000000p-4, 0x1.82b7be0000000p-3, -0x1.1fe0ee0000000p-4, 0x1.1fab8a0000000p-3, 0x1.26a24a0000000p-5, -0x1.a27dac0000000p-4, 0x1.c0dbba0000000p-6, 0x1.2b1d3a0000000p-3, -0x1.47c04c0000000p-4, -0x1.b107ee0000000p-5, 0x1.fb33600000000p-5, -0x1.cc2c640000000p-5, 0x1.f840a60000000p-3, 0x1.131f740000000p-3, 0x1.d4636e0000000p-5}
, {-0x1.e3cc660000000p-5, 0x1.1497220000000p-4, 0x1.0f601a0000000p-4, 0x1.36389e0000000p-10, 0x1.141a980000000p-4, 0x1.39fa760000000p-4, 0x1.2135400000000p-4, -0x1.afc5000000000p-5, 0x1.87d9080000000p-4, 0x1.f9cb1a0000000p-6, 0x1.c41f5e0000000p-4, 0x1.6968ec0000000p-7, 0x1.5b9be80000000p-5, 0x1.382fa60000000p-5, -0x1.0f71ba0000000p-3, 0x1.1e94120000000p-5, -0x1.4effc40000000p-7, -0x1.d378600000000p-5, 0x1.ff6b540000000p-5, -0x1.36c79e0000000p-3, 0x1.1b1bf80000000p-4, 0x1.1031600000000p-3, 0x1.4760cc0000000p-4, 0x1.9bac460000000p-5, -0x1.1b5db40000000p-5, 0x1.fd7c600000000p-4, 0x1.6835fa0000000p-6, 0x1.83f3c80000000p-7, 0x1.df35cc0000000p-5, 0x1.05597c0000000p-3, -0x1.652af60000000p-4, -0x1.4a46980000000p-5, 0x1.89cbaa0000000p-8, 0x1.391ffa0000000p-5, 0x1.855b880000000p-6, 0x1.f0ae400000000p-5, 0x1.d68d2e0000000p-5, -0x1.210d900000000p-6, 0x1.665e560000000p-4, -0x1.55f3840000000p-6, 0x1.9000800000000p-4, 0x1.4707880000000p-6, -0x1.a33c780000000p-11, -0x1.6640680000000p-4, 0x1.2acd380000000p-3, 0x1.196bea0000000p-3, -0x1.47bd3a0000000p-6, 0x1.3085620000000p-5, 0x1.d46ada0000000p-5, -0x1.a3b1b20000000p-4, 0x1.29da240000000p-6, -0x1.a676d20000000p-4, -0x1.8b9f4e0000000p-6, 0x1.fd241c0000000p-5, 0x1.f460120000000p-4, -0x1.7c96580000000p-10, 0x1.f649660000000p-5, -0x1.4660840000000p-5, 0x1.fc795c0000000p-6, 0x1.8927480000000p-3, -0x1.3a69c20000000p-4, -0x1.4ffeac0000000p-3, 0x1.0fa82c0000000p-4, 0x1.edcd9c0000000p-10}
, {0x1.d7a76a0000000p-4, 0x1.26ae3a0000000p-4, 0x1.32f8500000000p-3, 0x1.7ae3280000000p-4, 0x1.7cacb20000000p-4, -0x1.08ae200000000p-7, 0x1.f23e860000000p-5, -0x1.01a42c0000000p-4, 0x1.bf6d5e0000000p-4, 0x1.34da5e0000000p-2, 0x1.01f30e0000000p-3, 0x1.1c113a0000000p-2, 0x1.e43e940000000p-4, -0x1.4f63160000000p-4, -0x1.5d7f2e0000000p-5, 0x1.23bdc60000000p-5, -0x1.06c11c0000000p-7, -0x1.9f0fc80000000p-4, 0x1.5f31fa0000000p-5, 0x1.9f8eb40000000p-5, 0x1.71c52c0000000p-6, 0x1.3c21fe0000000p-4, 0x1.139efe0000000p-3, 0x1.7a26460000000p-3, -0x1.0cd7e80000000p-5, 0x1.62b9720000000p-3, 0x1.788ace0000000p-9, 0x1.25479e0000000p-3, 0x1.8e60d20000000p-4, 0x1.f0f2280000000p-4, -0x1.0e09140000000p-5, 0x1.12a0480000000p-6, -0x1.8209fc0000000p-5, 0x1.7891dc0000000p-4, 0x1.7fd0b60000000p-4, -0x1.0d6d600000000p-5, -0x1.1256460000000p-6, -0x1.6f70680000000p-4, 0x1.e704b40000000p-5, 0x1.55e33c0000000p-5, 0x1.0ab3a60000000p-3, -0x1.d6f2b00000000p-6, -0x1.c370a80000000p-9, -0x1.fc82740000000p-5, 0x1.5857020000000p-3, 0x1.f199600000000p-3, 0x1.41a0f60000000p-4, 0x1.3f03320000000p-4, 0x1.719d940000000p-5, -0x1.36edb40000000p-3, 0x1.01465c0000000p-2, 0x1.0856d60000000p-3, 0x1.4c4fd80000000p-4, 0x1.c174360000000p-5, -0x1.228bc20000000p-5, 0x1.b787d40000000p-5, -0x1.ad17280000000p-5, 0x1.a5424e0000000p-3, 0x1.92abdc0000000p-6, 0x1.1473e40000000p-2, -0x1.be57f00000000p-5, 0x1.3c09a80000000p-4, 0x1.670c1a0000000p-3, 0x1.dc24960000000p-5}
}
, {{-0x1.7921ce0000000p-3, -0x1.90a1180000000p-4, -0x1.e1f7f40000000p-4, -0x1.b0cbc00000000p-8, 0x1.1e2c460000000p-4, -0x1.f1e4de0000000p-5, 0x1.b324060000000p-4, 0x1.0b5dfa0000000p-3, -0x1.bc4b620000000p-4, 0x1.ad95560000000p-5, 0x1.45d01c0000000p-4, -0x1.17144a0000000p-4, -0x1.26ca0e0000000p-5, 0x1.375b540000000p-6, 0x1.1a2aa60000000p-4, -0x1.0938380000000p-14, 0x1.9340b40000000p-4, -0x1.0473940000000p-7, 0x1.af9cd60000000p-7, 0x1.0a59300000000p-3, 0x1.7448980000000p-4, -0x1.6ef05e0000000p-6, -0x1.b817b00000000p-4, 0x1.3bf48a0000000p-6, 0x1.2e30e80000000p-5, -0x1.dd90260000000p-4, 0x1.7faf9a0000000p-4, 0x1.7bd8460000000p-8, 0x1.07994a0000000p-3, 0x1.8ec0440000000p-4, 0x1.6344e20000000p-6, 0x1.7ae61c0000000p-4, 0x1.e538600000000p-4, -0x1.cebb780000000p-4, -0x1.10cfd60000000p-3, 0x1.e37e280000000p-4, 0x1.2298a20000000p-3, 0x1.f61aae0000000p-5, -0x1.df6d180000000p-5, 0x1.21c0cc0000000p-6, 0x1.11bf180000000p-3, 0x1.4671cc0000000p-3, 0x1.3e76220000000p-5, -0x1.a9a7360000000p-4, -0x1.6781520000000p-3, 0x1.e853e00000000p-5, -0x1.3f27480000000p-4, 0x1.4c582c0000000p-4, 0x1.9142220000000p-4, 0x1.af70e80000000p-6, 0x1.65e2940000000p-4, 0x1.6833580000000p-4, 0x1.bbbb620000000p-6, -0x1.5d607c0000000p-5, 0x1.4744b00000000p-10, 0x1.afc89a0000000p-3, 0x1.857c080000000p-4, 0x1.0ade4e0000000p-4, -0x1.1ed3d80000000p-3, -0x1.cceb8c0000000p-4, 0x1.f0480c0000000p-3, -0x1.7385920000000p-4, 0x1.b923780000000p-4, -0x1.6e90b80000000p-6}
, {-0x1.3eb5a60000000p-3, 0x1.7579100000000p-4, -0x1.956b680000000p-5, 0x1.0197b80000000p-13, 0x1.c777940000000p-4, -0x1.b4e5520000000p-8, -0x1.98c08a0000000p-5, -0x1.28177a0000000p-3, -0x1.3d79380000000p-4, 0x1.38c8860000000p-4, -0x1.b6d91c0000000p-8, 0x1.1e2ebc0000000p-7, 0x1.c1c0d40000000p-4, 0x1.da687a0000000p-4, -0x1.03d39c0000000p-3, -0x1.3320080000000p-4, 0x1.5d99b80000000p-5, -0x1.7a9b100000000p-5, 0x1.9e2cb00000000p-4, 0x1.7bafe60000000p-10, 0x1.63540e0000000p-7, -0x1.6e3a280000000p-4, 0x1.de3f140000000p-5, 0x1.28458e0000000p-3, -0x1.3fb61a0000000p-5, -0x1.e7bf160000000p-8, 0x1.92ba200000000p-6, 0x1.5715820000000p-4, -0x1.6d4a280000000p-5, 0x1.b336660000000p-5, -0x1.6b6ad60000000p-8, 0x1.db052a0000000p-3, 0x1.1c5ca40000000p-4, -0x1.bb99680000000p-5, 0x1.c578800000000p-7, 0x1.29db720000000p-9, 0x1.4ef5700000000p-3, -0x1.bbae9c0000000p-6, 0x1.14dbca0000000p-2, -0x1.a131700000000p-6, 0x1.ddc1320000000p-4, 0x1.e74b160000000p-5, 0x1.3db47a0000000p-3, -0x1.2dd7280000000p-6, -0x1.7cfa2e0000000p-6, 0x1.0ecd440000000p-4, -0x1.24284a0000000p-4, -0x1.0351a40000000p-3, 0x1.2759820000000p-3, 0x1.5ceeea0000000p-3, 0x1.dd672c0000000p-5, 0x1.775d420000000p-4, 0x1.c7b1180000000p-5, 0x1.96bddc0000000p-4, 0x1.33a2120000000p-3, 0x1.41033a0000000p-4, 0x1.148b940000000p-3, -0x1.7d0bd00000000p-3, 0x1.cca4740000000p-3, 0x1.b0b8860000000p-6, 0x1.aaa8d80000000p-3, -0x1.8fb41c0000000p-12, 0x1.8f01be0000000p-5, 0x1.3034ae0000000p-3}
, {-0x1.0ed9480000000p-9, -0x1.84cd6a0000000p-3, -0x1.8f8d920000000p-5, 0x1.4567760000000p-4, 0x1.0b51fa0000000p-3, 0x1.2c5f2c0000000p-7, 0x1.51842c0000000p-5, -0x1.0115280000000p-4, -0x1.98d1d60000000p-4, -0x1.78eddc0000000p-3, -0x1.57e62c0000000p-5, -0x1.6266da0000000p-3, 0x1.ffadd60000000p-4, -0x1.c9c42e0000000p-6, -0x1.a0df320000000p-3, -0x1.5e3be20000000p-4, -0x1.853b220000000p-4, -0x1.6d66100000000p-3, -0x1.8aa28c0000000p-4, 0x1.1324f60000000p-3, 0x1.7312c40000000p-5, 0x1.1863320000000p-4, -0x1.bab1d00000000p-7, 0x1.b10e040000000p-4, -0x1.6e31980000000p-7, -0x1.6cc3500000000p-3, 0x1.c568380000000p-4, 0x1.4dbe5a0000000p-3, 0x1.eab1b40000000p-3, 0x1.2808fa0000000p-3, -0x1.8233aa0000000p-3, -0x1.648a0c0000000p-5, 0x1.5f5dfe0000000p-3, 0x1.4b61440000000p-3, -0x1.0f97e20000000p-7, -0x1.687f8c0000000p-4, -0x1.b8c32c0000000p-4, -0x1.3d3bee0000000p-4, -0x1.43a9720000000p-4, -0x1.8ef61a0000000p-6, 0x1.c1ffa80000000p-5, 0x1.d1322c0000000p-4, -0x1.58de4c0000000p-4, -0x1.b5b1460000000p-8, 0x1.c6c6960000000p-7, 0x1.88efc80000000p-3, 0x1.c7959e0000000p-4, -0x1.f2e95a0000000p-7, 0x1.6309220000000p-3, 0x1.0122740000000p-5, 0x1.7652cc0000000p-5, 0x1.798e840000000p-6, -0x1.38dbba0000000p-4, 0x1.d3cb0e0000000p-4, 0x1.b97ac80000000p-5, 0x1.2f7d5e0000000p-3, -0x1.0037740000000p-5, 0x1.ac6dc80000000p-5, 0x1.13b9c40000000p-3, 0x1.0e4a980000000p-4, 0x1.3035300000000p-2, 0x1.5d342e0000000p-7, -0x1.e5ba860000000p-4, 0x1.4a5e120000000p-3}
}
, {{-0x1.5da9a20000000p-4, 0x1.27bd980000000p-4, -0x1.fe70020000000p-5, -0x1.85f3aa0000000p-5, 0x1.5256a40000000p-3, 0x1.b6bac40000000p-6, -0x1.298b200000000p-6, -0x1.d286800000000p-3, -0x1.9fab960000000p-5, -0x1.e8f2ae0000000p-6, -0x1.b734100000000p-6, -0x1.4ebf820000000p-2, 0x1.43e5440000000p-4, 0x1.19aeb40000000p-2, -0x1.83dbea0000000p-3, -0x1.897b720000000p-5, 0x1.1c983c0000000p-4, -0x1.8e3f940000000p-3, 0x1.af296a0000000p-5, -0x1.fc70580000000p-3, 0x1.1ca5ac0000000p-4, -0x1.5701b00000000p-3, 0x1.74c5440000000p-5, 0x1.2b6aee0000000p-3, -0x1.53ba440000000p-5, -0x1.48c2f40000000p-4, 0x1.3071920000000p-3, -0x1.855a300000000p-5, 0x1.fdbe3a0000000p-6, -0x1.d0c0f40000000p-5, -0x1.9f15cc0000000p-6, 0x1.d571a40000000p-3, 0x1.999bca0000000p-3, -0x1.5bbe0c0000000p-4, -0x1.69ec380000000p-3, 0x1.161bf40000000p-5, -0x1.7b385c0000000p-6, 0x1.aa660a0000000p-3, 0x1.2725500000000p-3, -0x1.ff7e2e0000000p-4, 0x1.0b8be00000000p-2, -0x1.b898440000000p-4, -0x1.5323320000000p-5, -0x1.5542fa0000000p-5, -0x1.63d8ec0000000p-5, -0x1.9cd7060000000p-4, -0x1.0587400000000p-6, -0x1.b367fa0000000p-4, 0x1.df1f300000000p-5, -0x1.0055b40000000p-2, -0x1.f875d20000000p-5, 0x1.93c5880000000p-4, -0x1.6511460000000p-4, 0x1.da82f40000000p-5, 0x1.fab54a0000000p-4, 0x1.41e0980000000p-3, -0x1.56b8400000000p-11, -0x1.5c8ab60000000p-3, 0x1.93a80c0000000p-4, 0x1.2ca8380000000p-3, 0x1.81390a0000000p-5, -0x1.f3e2860000000p-8, 0x1.da09ba0000000p-5, 0x1.847c440000000p-3}
, {0x1.eedb940000000p-7, 0x1.c282880000000p-8, 0x1.481d7a0000000p-4, -0x1.59b7f80000000p-5, -0x1.2789c00000000p-4, 0x1.dd8b980000000p-4, 0x1.7fcb560000000p-3, 0x1.a264080000000p-5, 0x1.2c8cb20000000p-5, 0x1.4323fc0000000p-4, -0x1.9160940000000p-6, 0x1.37416c0000000p-7, 0x1.ae85740000000p-6, 0x1.58d46e0000000p-3, 0x1.7b86d00000000p-4, -0x1.5c05e20000000p-5, 0x1.0f28520000000p-4, -0x1.643a920000000p-3, 0x1.3d00d60000000p-6, -0x1.b035440000000p-8, 0x1.bfe7300000000p-6, 0x1.7414940000000p-5, -0x1.0b549e0000000p-4, -0x1.ad96da0000000p-9, -0x1.6278d20000000p-5, 0x1.e1fc520000000p-4, 0x1.a089ac0000000p-6, 0x1.e332cc0000000p-5, 0x1.e97be60000000p-5, 0x1.d19bf00000000p-4, -0x1.14e31a0000000p-4, 0x1.36f4dc0000000p-3, 0x1.0de6a60000000p-6, 0x1.bd72ce0000000p-4, 0x1.be7ad80000000p-5, 0x1.68e2060000000p-4, 0x1.3343340000000p-6, 0x1.efc1a20000000p-5, 0x1.ac77100000000p-4, 0x1.c07cb00000000p-6, 0x1.53ee900000000p-5, 0x1.6479880000000p-4, 0x1.8518560000000p-4, -0x1.e7919c0000000p-6, 0x1.66e6a80000000p-5, -0x1.e03c020000000p-4, 0x1.0d9d320000000p-2, 0x1.25097c0000000p-4, 0x1.0640100000000p-3, -0x1.0e9af00000000p-3, 0x1.8b8bac0000000p-4, 0x1.710a220000000p-4, 0x1.2bf7e00000000p-4, 0x1.b03e360000000p-3, 0x1.d692900000000p-4, 0x1.8344a60000000p-4, 0x1.0105860000000p-4, -0x1.f485280000000p-10, -0x1.03cee80000000p-4, -0x1.721f0c0000000p-5, -0x1.60eb3e0000000p-3, 0x1.a188ae0000000p-6, -0x1.24d5900000000p-4, -0x1.a8d2760000000p-4}
, {0x1.882bd20000000p-8, 0x1.32f1600000000p-3, 0x1.a69d420000000p-3, -0x1.76ae760000000p-6, 0x1.9d61f20000000p-3, -0x1.2958080000000p-4, 0x1.832f0a0000000p-5, 0x1.194f260000000p-3, 0x1.b671fc0000000p-8, 0x1.dd8fdc0000000p-4, 0x1.5815880000000p-3, 0x1.a5eef80000000p-8, 0x1.63c72c0000000p-4, 0x1.06cd9e0000000p-3, 0x1.610a020000000p-7, -0x1.5e7dfc0000000p-4, 0x1.03a9400000000p-3, -0x1.38bac00000000p-3, 0x1.490eb40000000p-4, -0x1.779dae0000000p-9, -0x1.6cb1720000000p-4, -0x1.a0882e0000000p-5, -0x1.8050480000000p-4, 0x1.c4fee60000000p-4, -0x1.85b3e60000000p-7, 0x1.bcf6060000000p-3, 0x1.6cf47a0000000p-5, 0x1.46a5020000000p-5, 0x1.194fbc0000000p-3, 0x1.2936360000000p-4, 0x1.dd7f080000000p-4, 0x1.4e17040000000p-4, 0x1.db6a2a0000000p-6, -0x1.4187f20000000p-4, -0x1.2204b20000000p-6, 0x1.9c6f2e0000000p-4, 0x1.df9dd00000000p-5, 0x1.18dc5e0000000p-7, 0x1.a057f00000000p-5, 0x1.162a180000000p-5, 0x1.1bf42c0000000p-4, 0x1.b4fa0e0000000p-3, 0x1.37fe1a0000000p-5, -0x1.7fa6500000000p-7, 0x1.88432e0000000p-5, -0x1.af3ec80000000p-6, 0x1.9381080000000p-4, -0x1.00be480000000p-3, 0x1.a979da0000000p-5, -0x1.07b10c0000000p-3, 0x1.95fcaa0000000p-3, 0x1.99f5d00000000p-3, 0x1.b2e7be0000000p-5, 0x1.a6e0780000000p-3, -0x1.0ea42a0000000p-3, 0x1.1eea220000000p-3, 0x1.30af4e0000000p-4, 0x1.7824280000000p-5, 0x1.ab9b2e0000000p-5, -0x1.7000200000000p-4, 0x1.a2624e0000000p-4, 0x1.2eebd40000000p-5, -0x1.8a98440000000p-3, -0x1.1d12ae0000000p-6}
}
, {{0x1.94dc5c0000000p-2, 0x1.230b040000000p-5, 0x1.7cce4c0000000p-7, -0x1.0726fe0000000p-3, 0x1.5b1ffa0000000p-3, -0x1.9f7a8c0000000p-4, -0x1.534c0c0000000p-4, 0x1.5f60c60000000p-4, 0x1.576e9a0000000p-5, 0x1.ea2ff20000000p-6, -0x1.4063e20000000p-3, 0x1.52f61a0000000p-4, -0x1.d768e60000000p-3, 0x1.0f275c0000000p-3, -0x1.f7deb20000000p-3, 0x1.54238a0000000p-5, 0x1.2a30dc0000000p-7, 0x1.78ac2c0000000p-8, 0x1.2992300000000p-3, 0x1.610da00000000p-5, -0x1.1880360000000p-4, 0x1.005dc00000000p-4, 0x1.819eb00000000p-4, 0x1.ee7fe80000000p-7, 0x1.5b57760000000p-4, 0x1.074bce0000000p-4, -0x1.bf95ee0000000p-6, 0x1.718c280000000p-3, 0x1.c07ec60000000p-6, -0x1.0ba8040000000p-3, 0x1.d76b500000000p-4, -0x1.e780680000000p-4, -0x1.752ec20000000p-6, 0x1.55be420000000p-4, -0x1.1fcb0a0000000p-3, -0x1.37ce0a0000000p-3, -0x1.3111260000000p-4, 0x1.2f36f20000000p-5, 0x1.3fbe9e0000000p-3, -0x1.316cae0000000p-3, 0x1.b1bb2c0000000p-3, -0x1.552da00000000p-3, -0x1.1443f00000000p-5, 0x1.1d1dac0000000p-6, -0x1.7f29960000000p-5, -0x1.04f27e0000000p-2, -0x1.70b7b80000000p-5, -0x1.9066920000000p-9, -0x1.a552100000000p-4, 0x1.b193320000000p-5, 0x1.bb26780000000p-6, -0x1.bf05ce0000000p-7, 0x1.a2b8c60000000p-5, 0x1.0b678c0000000p-3, 0x1.684f500000000p-5, -0x1.c98d2a0000000p-4, 0x1.75c7ea0000000p-3, 0x1.4864c40000000p-5, -0x1.c35b980000000p-8, 0x1.7fa1660000000p-6, 0x1.9660fe0000000p-6, 0x1.1469dc0000000p-1, 0x1.345de20000000p-5, 0x1.77c1ca0000000p-4}
, {-0x1.0e2cc20000000p-3, -0x1.68a9a00000000p-5, 0x1.0911560000000p-3, 0x1.042b1a0000000p-7, 0x1.2359f60000000p-4, -0x1.5a1df20000000p-4, 0x1.b417900000000p-6, 0x1.0ebb2e0000000p-4, 0x1.eb7d3e0000000p-5, 0x1.daeca80000000p-9, -0x1.5c45200000000p-3, 0x1.59f31a0000000p-4, 0x1.eb9bc80000000p-4, 0x1.13452e0000000p-3, 0x1.ae70720000000p-4, 0x1.827ade0000000p-3, -0x1.ef21fa0000000p-6, -0x1.6dff180000000p-4, 0x1.afecaa0000000p-4, -0x1.206b780000000p-5, 0x1.5b33ec0000000p-5, -0x1.0b874e0000000p-2, 0x1.e432980000000p-4, -0x1.0cbc400000000p-4, 0x1.c9dba00000000p-5, 0x1.60854a0000000p-3, -0x1.8ec1ea0000000p-6, 0x1.61f4de0000000p-6, 0x1.7194740000000p-6, -0x1.bf1bc40000000p-7, 0x1.e751440000000p-5, -0x1.062ff00000000p-2, -0x1.fbc59a0000000p-5, -0x1.2494d60000000p-3, 0x1.22bc960000000p-5, -0x1.c06bd60000000p-5, -0x1.a0c4960000000p-6, 0x1.21bbbe0000000p-3, 0x1.e91da60000000p-4, -0x1.3186560000000p-4, -0x1.d53db00000000p-4, -0x1.c4c0280000000p-4, 0x1.b020560000000p-3, -0x1.0e1b600000000p-4, -0x1.2ec33a0000000p-3, -0x1.2f8e260000000p-3, -0x1.243c900000000p-4, 0x1.df78360000000p-3, -0x1.0900fc0000000p-3, -0x1.35184a0000000p-2, 0x1.02e73a0000000p-4, 0x1.6c02600000000p-7, -0x1.ab8ddc0000000p-6, -0x1.61ddb00000000p-3, -0x1.dc57920000000p-5, -0x1.d4a7700000000p-4, -0x1.282aac0000000p-5, 0x1.0ab9100000000p-3, -0x1.ea3e9c0000000p-3, -0x1.ea3b6e0000000p-7, -0x1.68da8e0000000p-3, 0x1.3d3b8e0000000p-3, -0x1.0b3cd20000000p-3, 0x1.febdba0000000p-4}
, {-0x1.5934a00000000p-6, 0x1.ac53440000000p-4, 0x1.c49e1c0000000p-3, 0x1.f279ec0000000p-3, 0x1.1584940000000p-5, -0x1.39c4c20000000p-5, -0x1.ee182e0000000p-5, 0x1.5aaf160000000p-4, -0x1.7e19a80000000p-5, 0x1.0fd3780000000p-3, 0x1.570b760000000p-3, -0x1.dc2f380000000p-5, -0x1.62ff820000000p-4, 0x1.7e6a240000000p-5, 0x1.e6f67a0000000p-7, -0x1.b1b9ce0000000p-4, 0x1.26becc0000000p-5, -0x1.13a0480000000p-3, -0x1.8b9b200000000p-6, -0x1.00f0480000000p-3, -0x1.0879d60000000p-3, -0x1.08cc520000000p-2, -0x1.8f59180000000p-3, 0x1.d552fe0000000p-3, -0x1.473d300000000p-3, 0x1.9462620000000p-3, 0x1.87b9f40000000p-6, 0x1.6953b40000000p-3, -0x1.cba3b40000000p-3, -0x1.d998b20000000p-4, 0x1.093b040000000p-4, -0x1.7915fe0000000p-7, 0x1.02b9ba0000000p-3, -0x1.16214e0000000p-4, -0x1.e144300000000p-3, -0x1.0804a40000000p-3, 0x1.b5ed1c0000000p-2, 0x1.0096620000000p-5, 0x1.8f082c0000000p-4, -0x1.b933f20000000p-5, 0x1.20b4520000000p-5, 0x1.a9f38a0000000p-3, 0x1.0ae19e0000000p-2, -0x1.65993a0000000p-4, 0x1.93a41c0000000p-3, -0x1.1f6bd20000000p-3, -0x1.02d31e0000000p-5, 0x1.a4b7e20000000p-4, -0x1.58ce4c0000000p-3, 0x1.5baa4c0000000p-3, 0x1.bc95fa0000000p-4, 0x1.59d5d60000000p-5, 0x1.945a120000000p-4, 0x1.23075a0000000p-6, 0x1.73df900000000p-5, -0x1.57cbd20000000p-6, -0x1.ae9ca60000000p-9, 0x1.8031c60000000p-4, -0x1.62353e0000000p-6, 0x1.a224780000000p-3, -0x1.147b800000000p-5, 0x1.2dee100000000p-3, 0x1.48ead20000000p-6, 0x1.1c27cc0000000p-6}
}
, {{0x1.364cb00000000p-2, 0x1.88c5360000000p-5, 0x1.17a9200000000p-3, -0x1.3006b40000000p-7, -0x1.563c0c0000000p-3, 0x1.28934a0000000p-4, -0x1.3508c20000000p-2, 0x1.3631dc0000000p-2, 0x1.fd057a0000000p-5, 0x1.1ca0840000000p-4, 0x1.09d20a0000000p-2, 0x1.e434a60000000p-3, -0x1.4f78580000000p-4, -0x1.c8323a0000000p-3, -0x1.f97c7a0000000p-6, 0x1.0682560000000p-3, 0x1.bb86ec0000000p-5, 0x1.2a5e1a0000000p-4, 0x1.17a5320000000p-3, 0x1.0334460000000p-3, 0x1.1597200000000p-3, 0x1.fa0a380000000p-3, -0x1.d9dca00000000p-4, 0x1.2d02f60000000p-3, -0x1.4b34240000000p-3, -0x1.bc11e00000000p-7, 0x1.4374020000000p-4, 0x1.0ef7da0000000p-4, 0x1.764e280000000p-3, -0x1.8684280000000p-3, 0x1.bd30200000000p-4, 0x1.24a2ae0000000p-7, -0x1.8dc76c0000000p-4, 0x1.a0df580000000p-4, 0x1.e20a900000000p-4, -0x1.9a90fe0000000p-5, 0x1.a4066e0000000p-4, -0x1.002ada0000000p-3, 0x1.66e3660000000p-3, 0x1.0d46fa0000000p-2, 0x1.346d580000000p-7, -0x1.05e9180000000p-5, -0x1.0b8d4e0000000p-3, -0x1.313cc20000000p-5, 0x1.1054380000000p-4, -0x1.7fb98c0000000p-3, -0x1.2ac6f80000000p-4, 0x1.b803280000000p-3, -0x1.9566300000000p-6, 0x1.40f6240000000p-3, 0x1.8c620a0000000p-4, 0x1.5e36e20000000p-3, 0x1.ba0ea80000000p-3, 0x1.c7fbf40000000p-4, -0x1.e031a00000000p-5, -0x1.2a13da0000000p-2, -0x1.01aa440000000p-4, 0x1.12f4c80000000p-2, 0x1.41d2aa0000000p-3, -0x1.5072d00000000p-3, 0x1.0954700000000p-5, 0x1.0be5740000000p-2, 0x1.ee30460000000p-4, -0x1.901eda0000000p-3}
, {-0x1.097ce20000000p-5, -0x1.7269340000000p-5, 0x1.16635e0000000p-5, -0x1.a272780000000p-7, -0x1.aff19e0000000p-5, -0x1.0515400000000p-4, -0x1.aa91d80000000p-5, 0x1.24210a0000000p-3, 0x1.a830900000000p-4, 0x1.b34eb60000000p-6, 0x1.6376ea0000000p-3, 0x1.25c5720000000p-4, 0x1.e127660000000p-8, 0x1.3671740000000p-4, 0x1.0d00920000000p-3, 0x1.454cf20000000p-5, -0x1.41e84a0000000p-6, -0x1.0a67020000000p-4, 0x1.597ed00000000p-9, -0x1.afd5ec0000000p-5, 0x1.8e6cc00000000p-4, -0x1.7bfc7a0000000p-3, -0x1.e24cc80000000p-4, 0x1.fdd9540000000p-4, -0x1.294af40000000p-4, -0x1.495d120000000p-5, 0x1.57323c0000000p-5, -0x1.10f2e20000000p-4, 0x1.2329680000000p-5, -0x1.4ef19e0000000p-4, -0x1.022c240000000p-3, -0x1.01c8460000000p-3, 0x1.7e897a0000000p-4, -0x1.ab0bcc0000000p-4, 0x1.09db4e0000000p-5, 0x1.0ac4140000000p-4, 0x1.3290a60000000p-4, 0x1.22899e0000000p-3, -0x1.3124c40000000p-5, 0x1.d47d040000000p-3, -0x1.8e2a080000000p-2, 0x1.3b48ba0000000p-4, -0x1.92ac240000000p-4, -0x1.3b87720000000p-3, 0x1.3a3fc80000000p-8, 0x1.dae3840000000p-7, -0x1.a600520000000p-4, 0x1.2627b40000000p-3, 0x1.7492740000000p-7, -0x1.ac60840000000p-5, -0x1.7acf5a0000000p-5, 0x1.60e8460000000p-5, 0x1.5d9ac60000000p-5, 0x1.ab38aa0000000p-5, -0x1.9d87da0000000p-6, 0x1.65167c0000000p-5, -0x1.7cc0b20000000p-3, -0x1.a27a380000000p-9, 0x1.3f7b120000000p-4, 0x1.5846620000000p-7, 0x1.3eb9c40000000p-4, 0x1.5cd84e0000000p-3, -0x1.849e220000000p-3, 0x1.5cde2a0000000p-4}
, {-0x1.7bf74c0000000p-5, 0x1.7bb9960000000p-5, -0x1.4f5f9a0000000p-4, -0x1.0826040000000p-5, -0x1.d3444e0000000p-3, 0x1.1bfb220000000p-3, -0x1.6841840000000p-4, -0x1.1f10880000000p-7, 0x1.686a6c0000000p-3, 0x1.3ccfb60000000p-4, -0x1.9dce320000000p-4, -0x1.0ccdf60000000p-3, -0x1.12a81e0000000p-5, -0x1.287ed40000000p-2, 0x1.1f393a0000000p-6, 0x1.ab125c0000000p-5, -0x1.10e5460000000p-5, 0x1.fd000a0000000p-9, 0x1.08c7f40000000p-3, -0x1.a919040000000p-4, -0x1.022ee40000000p-4, -0x1.3cfc8c0000000p-9, -0x1.64ef320000000p-4, -0x1.68da260000000p-4, 0x1.a10b480000000p-5, 0x1.3e76200000000p-4, 0x1.4f69e80000000p-4, 0x1.2dbf480000000p-7, 0x1.5f5f120000000p-4, 0x1.f3af700000000p-3, -0x1.79d0f80000000p-2, -0x1.25e4740000000p-4, -0x1.d516600000000p-6, 0x1.14fad40000000p-4, 0x1.978c800000000p-3, 0x1.a51b100000000p-3, 0x1.3a475a0000000p-6, 0x1.9370c40000000p-4, -0x1.cdef840000000p-5, 0x1.1a3c520000000p-2, -0x1.9eeee00000000p-2, -0x1.156e9c0000000p-6, -0x1.767fc60000000p-4, -0x1.b9f8e00000000p-4, -0x1.5489f20000000p-3, -0x1.7617fc0000000p-4, 0x1.ba3a240000000p-5, 0x1.0c48500000000p-5, 0x1.10acdc0000000p-2, 0x1.42f36e0000000p-4, 0x1.2c440a0000000p-4, 0x1.0700280000000p-10, 0x1.33a4a60000000p-3, -0x1.0f591c0000000p-2, 0x1.1ff3b60000000p-4, -0x1.5d219a0000000p-7, -0x1.513f020000000p-4, 0x1.d7f2e00000000p-5, 0x1.2ede880000000p-7, -0x1.7fb1500000000p-5, 0x1.4f67f00000000p-8, -0x1.c916220000000p-5, -0x1.68277e0000000p-4, 0x1.bfd6220000000p-6}
}
, {{-0x1.629e860000000p-4, 0x1.4f9ada0000000p-5, 0x1.3f5ce00000000p-5, 0x1.38782e0000000p-5, -0x1.b71b720000000p-5, -0x1.957f060000000p-4, 0x1.2fba180000000p-6, -0x1.c53b000000000p-5, 0x1.fc4fa20000000p-7, 0x1.2474aa0000000p-3, -0x1.9f6f180000000p-5, -0x1.68bbea0000000p-4, 0x1.9057080000000p-3, 0x1.e568780000000p-4, 0x1.03c0420000000p-2, 0x1.2e27d60000000p-3, -0x1.1bd6640000000p-7, 0x1.a2f7c40000000p-5, 0x1.bae13e0000000p-4, -0x1.b779da0000000p-3, 0x1.437ae40000000p-4, -0x1.13c4b80000000p-3, -0x1.397fb20000000p-7, 0x1.95f0ba0000000p-7, 0x1.dd922e0000000p-4, 0x1.c823e20000000p-4, -0x1.a195060000000p-5, 0x1.2efda40000000p-4, 0x1.7bbebc0000000p-6, 0x1.3f14460000000p-4, 0x1.29f2e80000000p-9, -0x1.3463a40000000p-2, 0x1.97c38e0000000p-3, -0x1.06234e0000000p-4, -0x1.6c87b00000000p-4, 0x1.5d47f40000000p-13, 0x1.d0de2a0000000p-4, 0x1.74b9ce0000000p-4, -0x1.0553c20000000p-6, 0x1.1ef0820000000p-4, -0x1.61aa2c0000000p-2, -0x1.a4a47c0000000p-4, 0x1.8d56320000000p-4, 0x1.5a9cd20000000p-5, -0x1.784d8e0000000p-3, 0x1.b15b520000000p-4, -0x1.5cb0fa0000000p-3, -0x1.b53a8e0000000p-10, 0x1.428c320000000p-5, -0x1.e613c40000000p-3, -0x1.0af31c0000000p-6, -0x1.864ed60000000p-5, 0x1.86337e0000000p-5, -0x1.2121200000000p-7, 0x1.a97aa20000000p-4, -0x1.5113120000000p-4, -0x1.cba01a0000000p-7, 0x1.509a900000000p-4, -0x1.74a3a20000000p-5, -0x1.fa19260000000p-7, -0x1.7527a20000000p-3, -0x1.38848c0000000p-2, -0x1.084af40000000p-4, 0x1.0b62060000000p-3}
, {0x1.5edd6e0000000p-4, 0x1.a26f7a0000000p-6, 0x1.56533c0000000p-3, 0x1.066a160000000p-3, -0x1.cd296c0000000p-3, 0x1.ea46380000000p-5, -0x1.ee3a300000000p-4, 0x1.cafdc80000000p-4, -0x1.c7d9d80000000p-6, 0x1.5751840000000p-4, 0x1.ae1c640000000p-3, 0x1.1151b60000000p-6, 0x1.34956a0000000p-5, -0x1.00f7760000000p-4, -0x1.fde4620000000p-6, 0x1.8206640000000p-4, 0x1.0700ae0000000p-3, -0x1.0722300000000p-4, 0x1.ef936e0000000p-6, -0x1.50ad3e0000000p-5, 0x1.8f19260000000p-7, 0x1.5017e00000000p-5, -0x1.e020ae0000000p-6, -0x1.47cf3e0000000p-4, 0x1.0a6e980000000p-6, 0x1.42e3ac0000000p-4, 0x1.5eb20c0000000p-3, 0x1.49317a0000000p-3, 0x1.2b4bd60000000p-3, 0x1.2109a00000000p-3, -0x1.ac77040000000p-5, -0x1.621dfa0000000p-6, -0x1.01171e0000000p-3, 0x1.d240ec0000000p-4, 0x1.b916c00000000p-3, 0x1.406edc0000000p-4, 0x1.c78c180000000p-4, 0x1.05044a0000000p-2, -0x1.4779240000000p-5, 0x1.07e64c0000000p-2, -0x1.f84b520000000p-7, 0x1.f8c8b40000000p-4, -0x1.2b5cf20000000p-4, 0x1.b331780000000p-4, 0x1.b180120000000p-5, -0x1.69b53a0000000p-8, 0x1.9d75580000000p-10, 0x1.4686680000000p-3, 0x1.c6ad060000000p-3, -0x1.23fac40000000p-4, 0x1.83a2e80000000p-4, -0x1.d3e05c0000000p-7, 0x1.3d067e0000000p-4, 0x1.11c5c40000000p-3, 0x1.984b0a0000000p-4, -0x1.9c720a0000000p-5, -0x1.efef0c0000000p-4, 0x1.63cf5c0000000p-3, 0x1.1964380000000p-3, -0x1.01bccc0000000p-4, -0x1.ba1aa00000000p-3, 0x1.9e36f00000000p-4, 0x1.26c1600000000p-5, 0x1.3f627a0000000p-4}
, {0x1.2c992a0000000p-4, -0x1.0de4520000000p-5, -0x1.17cd9e0000000p-3, 0x1.c0dc940000000p-5, -0x1.9a6aa80000000p-6, 0x1.e3acd00000000p-5, -0x1.a38b240000000p-8, 0x1.b259c80000000p-4, 0x1.b3e1e40000000p-6, 0x1.be2c8c0000000p-6, 0x1.e0e8460000000p-3, 0x1.96b4360000000p-6, 0x1.ad95fa0000000p-3, -0x1.fe1e4c0000000p-6, 0x1.4b36ae0000000p-7, 0x1.c81e4c0000000p-5, 0x1.e032fa0000000p-8, 0x1.3e99380000000p-6, -0x1.1f0bcc0000000p-4, -0x1.5ff8760000000p-5, 0x1.27ef4e0000000p-7, -0x1.8b1c560000000p-5, -0x1.5638360000000p-9, 0x1.1ff7a40000000p-3, 0x1.7587440000000p-4, 0x1.d461b60000000p-5, 0x1.1e86fa0000000p-3, 0x1.6aacec0000000p-4, 0x1.fd4c280000000p-6, -0x1.1f75e60000000p-3, -0x1.0fc4ec0000000p-7, 0x1.60d2c60000000p-4, 0x1.3ef5580000000p-3, 0x1.f273e40000000p-5, -0x1.2beb420000000p-3, 0x1.c4b9740000000p-3, 0x1.2c7dce0000000p-2, -0x1.08a1d40000000p-10, -0x1.df3dec0000000p-5, -0x1.03e65e0000000p-4, 0x1.b8fc780000000p-4, -0x1.f81a1a0000000p-4, 0x1.00d3ee0000000p-3, -0x1.36828a0000000p-3, -0x1.e3d6b80000000p-4, 0x1.6f28700000000p-5, -0x1.64cea60000000p-5, 0x1.855fae0000000p-5, 0x1.0c7a1e0000000p-4, -0x1.fc6f160000000p-5, 0x1.c6e1c80000000p-10, -0x1.c4e7ac0000000p-6, -0x1.c7ff880000000p-8, 0x1.6c1fcc0000000p-3, -0x1.2dbeac0000000p-6, 0x1.1472b40000000p-2, 0x1.39e21c0000000p-4, 0x1.bf5b240000000p-4, 0x1.4d89460000000p-5, -0x1.895b8c0000000p-16, 0x1.6619b60000000p-3, -0x1.61eee60000000p-4, -0x1.167b7c0000000p-4, 0x1.7596100000000p-3}
}
, {{-0x1.bcc2740000000p-4, -0x1.a3601a0000000p-3, -0x1.8ebefc0000000p-6, -0x1.ea80fc0000000p-4, -0x1.7e1f2e0000000p-4, 0x1.5f8c800000000p-3, 0x1.10d0c80000000p-3, 0x1.c4fd000000000p-4, -0x1.3b68420000000p-3, -0x1.8ce36c0000000p-6, 0x1.0f15500000000p-4, -0x1.f2b98a0000000p-4, -0x1.154dc80000000p-4, 0x1.e3821c0000000p-4, 0x1.1bbedc0000000p-4, 0x1.b9a9e40000000p-9, -0x1.bcd5c80000000p-4, 0x1.5045080000000p-5, 0x1.b7a7780000000p-6, -0x1.4517e60000000p-6, 0x1.3a67640000000p-5, -0x1.99d8620000000p-7, -0x1.8cfff60000000p-3, -0x1.feb7f00000000p-4, 0x1.4ad48e0000000p-3, -0x1.ced2b00000000p-4, -0x1.a778a00000000p-4, -0x1.a6d6da0000000p-4, -0x1.9dd5180000000p-5, 0x1.10c4d80000000p-2, -0x1.bb7fd40000000p-6, 0x1.765b6a0000000p-4, 0x1.30f4e60000000p-4, 0x1.ee412e0000000p-4, 0x1.7b8a7a0000000p-5, 0x1.b7edaa0000000p-4, 0x1.7fd9720000000p-4, 0x1.1b80600000000p-5, -0x1.8324c00000000p-5, 0x1.887bd80000000p-4, 0x1.d47dcc0000000p-10, -0x1.1964ec0000000p-3, 0x1.8c59b00000000p-4, 0x1.29f8b60000000p-6, -0x1.8c17060000000p-9, -0x1.af6e0a0000000p-5, -0x1.400c880000000p-4, 0x1.0189260000000p-3, 0x1.3ed5d20000000p-3, -0x1.4af7de0000000p-4, 0x1.4366c20000000p-5, -0x1.a4398c0000000p-4, 0x1.5a4b240000000p-9, 0x1.27bd760000000p-6, 0x1.a63f2c0000000p-5, 0x1.d6355e0000000p-4, 0x1.2eb90a0000000p-5, -0x1.8618440000000p-6, -0x1.6496140000000p-4, 0x1.106f800000000p-5, 0x1.8f32560000000p-4, -0x1.7523180000000p-3, 0x1.5310900000000p-5, 0x1.202dae0000000p-4}
, {0x1.4d074e0000000p-4, 0x1.ddcec60000000p-3, -0x1.fba05a0000000p-4, 0x1.0449520000000p-5, 0x1.57517a0000000p-3, -0x1.d9c55a0000000p-5, 0x1.4cfb320000000p-4, 0x1.5479de0000000p-3, 0x1.0443ba0000000p-3, 0x1.3a64da0000000p-4, 0x1.ce36380000000p-4, -0x1.d7f0220000000p-3, 0x1.a43bca0000000p-10, 0x1.32f61a0000000p-4, -0x1.3520660000000p-3, -0x1.aca6180000000p-5, 0x1.1477300000000p-4, -0x1.166ac00000000p-3, 0x1.439a560000000p-4, -0x1.cb35c40000000p-7, -0x1.f468840000000p-5, -0x1.bbf6fc0000000p-4, -0x1.a5ef0c0000000p-5, -0x1.771b280000000p-4, -0x1.3d47380000000p-5, 0x1.0b58f00000000p-4, -0x1.922e800000000p-4, 0x1.15e4320000000p-2, 0x1.fe3ee40000000p-4, 0x1.2e45d80000000p-3, 0x1.9ca2da0000000p-5, 0x1.6bd2940000000p-3, 0x1.b44a820000000p-3, 0x1.7965ca0000000p-3, -0x1.88a02e0000000p-7, -0x1.51403c0000000p-5, 0x1.1f2b320000000p-4, 0x1.0a70160000000p-5, 0x1.85ef0a0000000p-5, 0x1.d528f60000000p-4, -0x1.d3a1420000000p-4, 0x1.53652c0000000p-3, 0x1.1fcdac0000000p-5, -0x1.5863280000000p-4, -0x1.8c032a0000000p-5, 0x1.5698ac0000000p-5, -0x1.f4cb700000000p-5, 0x1.7b60fc0000000p-6, -0x1.8147440000000p-6, -0x1.a655060000000p-6, -0x1.b4433c0000000p-7, 0x1.9c06dc0000000p-4, -0x1.7783ea0000000p-5, 0x1.9146480000000p-3, -0x1.5da8480000000p-4, 0x1.32353a0000000p-7, -0x1.4446e20000000p-4, 0x1.57fcda0000000p-4, 0x1.86d5d20000000p-4, -0x1.f73c2e0000000p-4, -0x1.b893de0000000p-7, 0x1.9372c60000000p-4, 0x1.6a6a100000000p-3, 0x1.a7c1400000000p-6}
, {0x1.551e600000000p-7, -0x1.37bf260000000p-4, -0x1.2792240000000p-2, -0x1.65dd080000000p-4, 0x1.4069ca0000000p-5, 0x1.5846c20000000p-6, 0x1.2d05920000000p-3, -0x1.aeba440000000p-4, 0x1.725f620000000p-5, 0x1.d49e000000000p-4, 0x1.e4106a0000000p-4, -0x1.1dd94a0000000p-4, 0x1.ff914a0000000p-3, -0x1.55e0f40000000p-6, -0x1.5171700000000p-4, -0x1.228d120000000p-4, 0x1.1c01ac0000000p-3, 0x1.5d59de0000000p-5, 0x1.5212f20000000p-5, -0x1.82d5200000000p-3, 0x1.34eaea0000000p-4, 0x1.a94c1e0000000p-5, -0x1.fd805e0000000p-3, 0x1.529c9e0000000p-4, 0x1.25b5100000000p-4, 0x1.58b4940000000p-7, -0x1.2616200000000p-6, 0x1.6740ea0000000p-6, 0x1.fd7bfc0000000p-5, 0x1.49bea00000000p-3, -0x1.97e0ce0000000p-3, -0x1.892cd20000000p-5, -0x1.81e4a60000000p-6, 0x1.212d320000000p-3, 0x1.9f54b60000000p-5, 0x1.c13e800000000p-10, 0x1.651b380000000p-4, -0x1.2b391a0000000p-5, 0x1.aa03b60000000p-5, 0x1.35a5bc0000000p-3, 0x1.353f600000000p-5, 0x1.1bd7c60000000p-6, 0x1.e33c420000000p-6, -0x1.ce79dc0000000p-7, 0x1.5a4b220000000p-7, 0x1.8a42aa0000000p-7, 0x1.0b3c500000000p-8, -0x1.8b78e60000000p-5, 0x1.160afc0000000p-2, 0x1.de6b400000000p-6, 0x1.a11f740000000p-5, -0x1.e9cd400000000p-6, 0x1.2c690c0000000p-5, 0x1.34a7fe0000000p-4, 0x1.cc9a4c0000000p-3, 0x1.0989600000000p-3, -0x1.6366860000000p-5, 0x1.fefd960000000p-4, 0x1.f645920000000p-7, 0x1.8dfa820000000p-9, 0x1.01e0aa0000000p-3, -0x1.5f32cc0000000p-3, 0x1.19b1da0000000p-3, 0x1.26f1e00000000p-3}
}
, {{0x1.65448c0000000p-7, 0x1.641f0a0000000p-3, -0x1.3e35340000000p-2, 0x1.15714c0000000p-6, -0x1.7086520000000p-4, 0x1.3b041e0000000p-3, -0x1.49a1400000000p-3, 0x1.468e5a0000000p-4, -0x1.5152100000000p-2, -0x1.1da70a0000000p-3, -0x1.5f22dc0000000p-4, 0x1.9fcfd60000000p-9, -0x1.a17c880000000p-3, -0x1.2c57040000000p-3, 0x1.5d61e60000000p-5, -0x1.02de4c0000000p-4, 0x1.2033740000000p-3, -0x1.adb8480000000p-6, -0x1.b6be000000000p-6, 0x1.12cdd40000000p-2, -0x1.d8a5100000000p-3, 0x1.57ceb80000000p-5, -0x1.94acd80000000p-4, -0x1.6e304c0000000p-4, -0x1.f830620000000p-5, -0x1.4fdd520000000p-10, -0x1.0643d60000000p-3, -0x1.8e107e0000000p-4, -0x1.dbf8580000000p-7, 0x1.40308a0000000p-3, -0x1.2437200000000p-4, 0x1.96db880000000p-3, -0x1.6ab4e40000000p-4, -0x1.2eb0fc0000000p-2, 0x1.208cc60000000p-2, 0x1.e7c8b60000000p-4, 0x1.5d0fde0000000p-3, -0x1.6aa74c0000000p-6, -0x1.9bd0960000000p-4, -0x1.226e6e0000000p-4, 0x1.8e00120000000p-3, -0x1.2526d40000000p-4, -0x1.7b8a400000000p-3, -0x1.544ce80000000p-4, 0x1.f162280000000p-14, -0x1.1682220000000p-3, 0x1.d9135e0000000p-4, -0x1.92e75c0000000p-3, 0x1.05bb3c0000000p-5, -0x1.b98ed60000000p-4, -0x1.e73b960000000p-6, -0x1.174d800000000p-3, 0x1.8281c40000000p-7, 0x1.7bb07c0000000p-3, -0x1.4f779a0000000p-3, 0x1.32235e0000000p-3, 0x1.b6add60000000p-4, -0x1.5e5c140000000p-7, 0x1.5b34fa0000000p-3, -0x1.8cacfe0000000p-13, 0x1.58d90c0000000p-3, -0x1.271bea0000000p-4, -0x1.a0e3660000000p-4, -0x1.0aa8da0000000p-3}
, {-0x1.c540e20000000p-3, -0x1.6a7ea00000000p-6, -0x1.1873680000000p-5, -0x1.326cbc0000000p-2, -0x1.03ddb80000000p-3, 0x1.2f82f80000000p-3, 0x1.d576960000000p-7, -0x1.17ef3e0000000p-5, -0x1.a70a400000000p-3, -0x1.cf43700000000p-5, 0x1.7386e20000000p-3, -0x1.260d2a0000000p-3, 0x1.35e6560000000p-6, -0x1.a362740000000p-7, -0x1.db7c9c0000000p-4, -0x1.79d1be0000000p-5, 0x1.0f60980000000p-7, -0x1.1cebd80000000p-4, -0x1.0e6d7e0000000p-4, 0x1.a2189c0000000p-6, 0x1.7a06b20000000p-4, 0x1.70e6ea0000000p-5, -0x1.06c4c20000000p-2, 0x1.9166780000000p-3, -0x1.3c2b8a0000000p-5, -0x1.c2a34e0000000p-4, 0x1.a06efc0000000p-5, 0x1.e457440000000p-6, -0x1.691bf60000000p-5, 0x1.705ee00000000p-3, 0x1.235d1c0000000p-5, 0x1.9f94ee0000000p-5, 0x1.c235b60000000p-7, -0x1.0a9c2e0000000p-3, -0x1.acd2580000000p-10, -0x1.4abe0c0000000p-4, 0x1.d6a3d60000000p-5, -0x1.7715860000000p-7, 0x1.990c1a0000000p-6, -0x1.a2bcc60000000p-4, -0x1.5d1a5a0000000p-7, -0x1.3fff320000000p-4, -0x1.4ecdb00000000p-3, -0x1.1a65ee0000000p-3, -0x1.2571860000000p-5, -0x1.fdabce0000000p-5, 0x1.eda0100000000p-5, -0x1.0cbcd40000000p-3, -0x1.547cf20000000p-4, -0x1.def9420000000p-4, -0x1.8090e60000000p-7, -0x1.ebb5d60000000p-10, 0x1.0c64cc0000000p-6, 0x1.d0db020000000p-4, 0x1.f632f00000000p-5, 0x1.2f564e0000000p-3, 0x1.89916c0000000p-3, 0x1.70f7ea0000000p-4, 0x1.52c9120000000p-3, 0x1.bc93bc0000000p-4, 0x1.261a500000000p-5, -0x1.c60bae0000000p-4, 0x1.a81e8e0000000p-6, 0x1.ad080e0000000p-5}
, {-0x1.8b3e020000000p-4, 0x1.44ce8e0000000p-3, 0x1.04a9600000000p-4, 0x1.52600e0000000p-7, 0x1.c6251c0000000p-5, -0x1.7c1ec40000000p-6, 0x1.9f22140000000p-5, 0x1.8429620000000p-3, 0x1.08b8900000000p-3, 0x1.4a8ee60000000p-4, 0x1.86d0820000000p-3, 0x1.915d700000000p-6, -0x1.9004a60000000p-4, 0x1.42b8920000000p-4, -0x1.7fbc280000000p-4, 0x1.8b63b60000000p-10, 0x1.a3682a0000000p-4, 0x1.d693c20000000p-5, 0x1.6f9e1a0000000p-6, -0x1.1157940000000p-3, 0x1.107b2a0000000p-3, -0x1.047b2c0000000p-5, 0x1.13572e0000000p-3, 0x1.ace2280000000p-3, 0x1.4c23020000000p-3, -0x1.57b9680000000p-4, 0x1.0a87f40000000p-4, 0x1.1d23d00000000p-3, -0x1.28ad880000000p-4, -0x1.c5ca060000000p-9, -0x1.bcbb080000000p-7, 0x1.56b7800000000p-3, 0x1.297faa0000000p-7, 0x1.a781340000000p-5, 0x1.99f59e0000000p-5, 0x1.a3985c0000000p-4, -0x1.2ebdf20000000p-3, -0x1.8401de0000000p-7, 0x1.bc46280000000p-4, 0x1.536cfc0000000p-5, 0x1.63c1c40000000p-2, 0x1.4ea9d00000000p-5, 0x1.767ca60000000p-11, -0x1.3ed4540000000p-4, 0x1.4c86b80000000p-4, -0x1.7968880000000p-5, -0x1.06abfe0000000p-3, -0x1.02d1d80000000p-3, -0x1.896ed20000000p-7, -0x1.a61e220000000p-5, -0x1.41b7ca0000000p-3, -0x1.cd4aa20000000p-4, -0x1.27c2bc0000000p-5, 0x1.2d8b440000000p-3, 0x1.f65a100000000p-4, -0x1.8674d80000000p-4, -0x1.6e9a8c0000000p-10, 0x1.2506940000000p-3, 0x1.6dcea60000000p-4, -0x1.74beb40000000p-5, -0x1.6e6a700000000p-3, 0x1.f6364e0000000p-6, 0x1.77eab80000000p-3, -0x1.f827aa0000000p-5}
}
, {{0x1.657f5a0000000p-4, -0x1.44a83a0000000p-3, -0x1.f3ef1c0000000p-5, -0x1.6525960000000p-4, 0x1.7e40ec0000000p-3, 0x1.261f3c0000000p-4, 0x1.ce47f00000000p-4, 0x1.038eb60000000p-2, 0x1.59375e0000000p-3, 0x1.b098ac0000000p-8, 0x1.6c178e0000000p-7, 0x1.0eb9d40000000p-4, 0x1.1085dc0000000p-5, -0x1.0923700000000p-4, -0x1.a6ff660000000p-4, 0x1.000ef40000000p-2, -0x1.b844f80000000p-5, -0x1.dc1a5c0000000p-5, -0x1.7197460000000p-4, 0x1.b859ae0000000p-5, -0x1.8509960000000p-10, 0x1.51d0060000000p-4, 0x1.611f140000000p-5, 0x1.60e6d40000000p-4, -0x1.495fda0000000p-3, 0x1.3677180000000p-3, -0x1.f7760c0000000p-5, -0x1.11668e0000000p-3, -0x1.bcb1000000000p-4, -0x1.24d0a40000000p-3, -0x1.64833e0000000p-4, 0x1.2905d60000000p-7, -0x1.c6ce260000000p-6, -0x1.9bdf400000000p-3, -0x1.3396760000000p-5, -0x1.44d3b20000000p-8, -0x1.703cf80000000p-4, 0x1.de8cea0000000p-6, -0x1.66d3be0000000p-4, -0x1.a248ba0000000p-4, 0x1.8f40060000000p-3, -0x1.ff66700000000p-5, -0x1.6488d00000000p-4, -0x1.a82ea00000000p-3, 0x1.7609660000000p-6, -0x1.84af0a0000000p-5, 0x1.7f096e0000000p-3, -0x1.383c120000000p-3, -0x1.3e67e00000000p-3, 0x1.517d920000000p-7, -0x1.3400e40000000p-5, -0x1.ca03be0000000p-4, 0x1.a2063e0000000p-7, -0x1.ad77ce0000000p-5, -0x1.4e607e0000000p-2, -0x1.b36ad00000000p-4, -0x1.3467a60000000p-4, 0x1.80ac880000000p-3, 0x1.2376000000000p-3, -0x1.d147960000000p-4, -0x1.835aae0000000p-4, 0x1.7d44b60000000p-3, -0x1.e3fb940000000p-3, -0x1.00f6780000000p-3}
, {0x1.8849fe0000000p-9, 0x1.710ea80000000p-4, -0x1.6d23780000000p-7, -0x1.585c780000000p-6, -0x1.2b277c0000000p-4, -0x1.21e6de0000000p-5, -0x1.a8599e0000000p-6, 0x1.a1be500000000p-5, -0x1.e5952e0000000p-6, -0x1.abf0d80000000p-3, -0x1.eae0200000000p-6, 0x1.6dd04e0000000p-5, -0x1.ab3c820000000p-5, -0x1.dedcfe0000000p-3, -0x1.3b6c820000000p-3, -0x1.716dc60000000p-5, -0x1.72f2220000000p-3, -0x1.4577fa0000000p-6, -0x1.5c986e0000000p-5, -0x1.ad46720000000p-6, -0x1.be1bb60000000p-3, -0x1.fb650a0000000p-5, -0x1.15725a0000000p-4, 0x1.4191860000000p-6, 0x1.127bfc0000000p-4, 0x1.4b69fe0000000p-2, 0x1.111ea60000000p-4, -0x1.1c3e3a0000000p-3, 0x1.3deec80000000p-6, -0x1.cded9c0000000p-5, 0x1.4ec0020000000p-3, 0x1.3cc4b20000000p-6, 0x1.6812ee0000000p-5, -0x1.4a4f260000000p-6, -0x1.10d2240000000p-4, -0x1.bc37420000000p-11, 0x1.ab278a0000000p-4, -0x1.4b012e0000000p-4, 0x1.3d676a0000000p-4, -0x1.09e2f40000000p-4, 0x1.e8ca8e0000000p-4, -0x1.626dde0000000p-3, -0x1.57bf240000000p-3, -0x1.5a6b1e0000000p-4, -0x1.c6b3ae0000000p-6, -0x1.5262cc0000000p-4, 0x1.897ace0000000p-3, -0x1.4917460000000p-3, -0x1.da89ce0000000p-4, 0x1.9df9380000000p-4, 0x1.b1cc1a0000000p-5, -0x1.13ce380000000p-3, -0x1.8ae0980000000p-4, 0x1.a5ab8c0000000p-6, -0x1.8555260000000p-3, -0x1.86cb1a0000000p-6, -0x1.d1025c0000000p-4, -0x1.35175c0000000p-4, -0x1.5a4dec0000000p-5, 0x1.675efe0000000p-4, -0x1.475c180000000p-6, 0x1.0a0c840000000p-5, 0x1.94c6740000000p-5, 0x1.0083240000000p-4}
, {0x1.57e8a00000000p-4, 0x1.a031760000000p-4, 0x1.3674de0000000p-4, -0x1.b561b00000000p-5, -0x1.57bec00000000p-3, -0x1.f740a80000000p-4, -0x1.9fe6c00000000p-6, 0x1.4a26b00000000p-5, 0x1.2d24100000000p-4, -0x1.9051c40000000p-5, -0x1.c18a1a0000000p-5, -0x1.bfe45e0000000p-7, -0x1.3f66140000000p-3, 0x1.39e4c60000000p-3, 0x1.b9a6060000000p-5, -0x1.c96dca0000000p-6, -0x1.1045e80000000p-3, -0x1.390e340000000p-3, 0x1.f13ee00000000p-4, 0x1.1702920000000p-5, -0x1.429ac60000000p-4, -0x1.1956b40000000p-3, -0x1.0d9c2c0000000p-4, -0x1.6bc6d80000000p-3, -0x1.f517140000000p-6, 0x1.0ff38e0000000p-2, -0x1.64f8b60000000p-3, 0x1.fa21020000000p-5, -0x1.0b3a520000000p-6, -0x1.e6662a0000000p-5, -0x1.22f5200000000p-3, 0x1.5d68900000000p-4, -0x1.1bcfb20000000p-5, -0x1.4172fe0000000p-5, 0x1.03ea140000000p-2, 0x1.2b44ca0000000p-4, -0x1.3f6d9a0000000p-2, 0x1.bfcdd20000000p-5, -0x1.1e4bb40000000p-4, 0x1.566fd20000000p-4, -0x1.1c559c0000000p-3, 0x1.670cc40000000p-8, -0x1.da20cc0000000p-3, 0x1.b1016a0000000p-5, -0x1.bb4e320000000p-3, -0x1.a4744c0000000p-4, -0x1.be17580000000p-5, -0x1.7ec9220000000p-5, 0x1.84f6200000000p-3, 0x1.2585380000000p-4, 0x1.8aec140000000p-4, -0x1.343eee0000000p-4, -0x1.7f1ec80000000p-3, -0x1.0678780000000p-3, 0x1.c47cda0000000p-3, 0x1.aa43a00000000p-6, -0x1.cf9c260000000p-5, 0x1.28c6cc0000000p-3, -0x1.8582fe0000000p-3, -0x1.9936e00000000p-3, -0x1.ccfc1e0000000p-9, 0x1.3df11e0000000p-5, 0x1.16d52a0000000p-7, -0x1.0c5b260000000p-3}
}
, {{0x1.13475c0000000p-5, 0x1.8a8e5a0000000p-4, -0x1.2931ce0000000p-5, 0x1.b436d60000000p-5, 0x1.aff4780000000p-3, 0x1.e9e35c0000000p-3, 0x1.c0c6000000000p-4, -0x1.5765b00000000p-7, -0x1.2565f80000000p-4, -0x1.0057b80000000p-5, 0x1.7b961c0000000p-4, 0x1.60f4b60000000p-9, -0x1.d7b2600000000p-7, 0x1.0536cc0000000p-5, -0x1.d679560000000p-5, -0x1.11047e0000000p-5, -0x1.176c6e0000000p-3, -0x1.3e68100000000p-3, 0x1.a309240000000p-7, 0x1.dc9f120000000p-5, 0x1.1f0c7a0000000p-4, 0x1.383f160000000p-5, -0x1.23f3380000000p-5, 0x1.bcabe80000000p-4, 0x1.0d44b20000000p-6, 0x1.a3d1f80000000p-5, 0x1.fcbad00000000p-6, 0x1.c8dafa0000000p-7, 0x1.e1581a0000000p-4, -0x1.792f9c0000000p-5, 0x1.0d70900000000p-10, 0x1.7a7cee0000000p-4, 0x1.9d795e0000000p-3, 0x1.0206e80000000p-3, -0x1.7779060000000p-4, 0x1.b2db2e0000000p-6, 0x1.c8e9520000000p-7, -0x1.0564580000000p-7, 0x1.04d6860000000p-3, 0x1.719d840000000p-3, 0x1.1a460e0000000p-2, -0x1.c8e4480000000p-5, -0x1.7e569e0000000p-6, -0x1.c6ceee0000000p-10, 0x1.b0af140000000p-4, 0x1.dd64460000000p-7, 0x1.2a0a1e0000000p-3, -0x1.f613500000000p-7, 0x1.821f0c0000000p-5, 0x1.ac023c0000000p-4, 0x1.8ed4520000000p-3, 0x1.1d33200000000p-4, 0x1.d193140000000p-4, 0x1.026f280000000p-5, -0x1.e03b060000000p-15, 0x1.2eeee60000000p-4, 0x1.955ae20000000p-3, 0x1.a8a6740000000p-4, 0x1.d471f80000000p-4, 0x1.d79d540000000p-3, 0x1.ff89e00000000p-4, 0x1.7bdc220000000p-4, 0x1.00397e0000000p-4, 0x1.26a3b80000000p-3}
, {-0x1.9a2ca40000000p-4, 0x1.f7cb9a0000000p-5, 0x1.0f813e0000000p-5, 0x1.a2d23c0000000p-5, -0x1.56bbca0000000p-5, 0x1.b1fa5a0000000p-6, 0x1.3239280000000p-6, 0x1.1546500000000p-4, -0x1.1a6cc80000000p-4, 0x1.92df400000000p-6, 0x1.2c58ee0000000p-3, 0x1.0a0c4c0000000p-7, 0x1.4eb1940000000p-4, 0x1.3adb3a0000000p-4, -0x1.a87fbe0000000p-3, 0x1.3a9ea40000000p-3, 0x1.32be460000000p-3, -0x1.c1c7e20000000p-6, 0x1.bc35f60000000p-4, 0x1.09cf0e0000000p-7, 0x1.a929a80000000p-7, 0x1.fd1b2a0000000p-4, 0x1.fc0cb80000000p-5, 0x1.0452540000000p-4, -0x1.7ae9ec0000000p-6, 0x1.842fca0000000p-3, -0x1.e727b60000000p-7, 0x1.99908e0000000p-6, 0x1.a315d00000000p-4, -0x1.50d8d80000000p-4, 0x1.7edd0c0000000p-4, 0x1.bfa0500000000p-4, -0x1.0e869e0000000p-3, -0x1.e1e1e20000000p-4, 0x1.8a03580000000p-4, 0x1.3e5b980000000p-3, 0x1.20bd7c0000000p-3, -0x1.c5548a0000000p-6, 0x1.93df5a0000000p-4, 0x1.1203a40000000p-4, -0x1.8ed7ee0000000p-5, 0x1.d0788c0000000p-4, -0x1.901f6a0000000p-7, -0x1.d19af20000000p-4, -0x1.89d95a0000000p-3, 0x1.e341fc0000000p-6, 0x1.3988920000000p-3, 0x1.32bc6c0000000p-5, 0x1.da73860000000p-4, -0x1.f09cb40000000p-5, 0x1.2a8dee0000000p-3, 0x1.afdbd80000000p-3, 0x1.545b540000000p-4, 0x1.2d01f20000000p-3, 0x1.7a66360000000p-4, 0x1.73c8f60000000p-7, -0x1.50259c0000000p-12, 0x1.613c700000000p-4, -0x1.a80ca40000000p-8, 0x1.a7bf5c0000000p-3, 0x1.a55a320000000p-7, 0x1.e269380000000p-3, 0x1.762d8a0000000p-4, 0x1.cd3f600000000p-4}
, {0x1.7d91b20000000p-4, 0x1.3c09be0000000p-6, -0x1.c2dad00000000p-6, -0x1.d74dae0000000p-5, -0x1.85ba520000000p-7, 0x1.574c780000000p-6, 0x1.3877700000000p-4, -0x1.6d8e480000000p-8, 0x1.d694d40000000p-5, 0x1.59aaec0000000p-6, 0x1.bb98dc0000000p-5, 0x1.30ec0e0000000p-5, 0x1.0f39300000000p-6, 0x1.5c059c0000000p-3, -0x1.dfda060000000p-3, 0x1.b18f360000000p-3, 0x1.9545de0000000p-3, 0x1.434b880000000p-4, 0x1.8b45000000000p-3, 0x1.6a44fc0000000p-5, -0x1.25a4fc0000000p-5, 0x1.52cca80000000p-3, -0x1.8355a60000000p-5, 0x1.71ea6c0000000p-3, 0x1.ba49480000000p-4, 0x1.6bde120000000p-5, 0x1.c46b220000000p-6, -0x1.df94d40000000p-4, -0x1.7143fe0000000p-7, 0x1.a82ea80000000p-3, -0x1.3624e60000000p-4, 0x1.7e086a0000000p-2, -0x1.a761a40000000p-4, -0x1.b3875c0000000p-3, 0x1.6b34be0000000p-5, 0x1.ac6ac60000000p-6, 0x1.9f4cb20000000p-3, 0x1.aeebbe0000000p-4, -0x1.5822c80000000p-11, 0x1.950cc20000000p-4, 0x1.4ea42c0000000p-3, 0x1.15fbea0000000p-3, 0x1.981d460000000p-7, -0x1.3652ae0000000p-4, -0x1.3459c00000000p-5, 0x1.5731160000000p-5, 0x1.ae49b00000000p-3, -0x1.4f80340000000p-5, 0x1.684a080000000p-4, -0x1.5bb7620000000p-2, 0x1.5a7d4a0000000p-3, 0x1.cd47420000000p-4, 0x1.8d56980000000p-7, 0x1.3bbc9a0000000p-3, 0x1.affecc0000000p-6, 0x1.d7ad560000000p-3, 0x1.95e3a00000000p-6, 0x1.4be06c0000000p-3, 0x1.746c0e0000000p-3, 0x1.ad1a9c0000000p-4, 0x1.8bfb1c0000000p-4, 0x1.26d9760000000p-4, -0x1.d227580000000p-6, 0x1.c595620000000p-4}
}
, {{-0x1.3986de0000000p-4, 0x1.8250d80000000p-4, 0x1.e6aaec0000000p-7, 0x1.06d8520000000p-2, 0x1.8cc2fe0000000p-4, -0x1.47b0e00000000p-3, 0x1.432bbe0000000p-6, -0x1.2e8a6c0000000p-6, 0x1.2276900000000p-3, 0x1.7db55a0000000p-3, 0x1.0ecd140000000p-3, 0x1.e2e8000000000p-7, -0x1.14534a0000000p-4, -0x1.156b7c0000000p-6, 0x1.0cf8c00000000p-3, -0x1.1f8a520000000p-5, -0x1.0b5b200000000p-7, -0x1.c16d740000000p-4, -0x1.a371240000000p-5, -0x1.33dcc80000000p-3, 0x1.5939d20000000p-7, -0x1.a27df40000000p-3, -0x1.8d5ccc0000000p-4, -0x1.019aa60000000p-4, 0x1.d8ad900000000p-4, -0x1.1607e60000000p-4, 0x1.3fd93c0000000p-4, 0x1.a188140000000p-5, -0x1.7419e20000000p-5, 0x1.26c10e0000000p-8, -0x1.132ada0000000p-4, -0x1.f9a4e40000000p-3, 0x1.8442ee0000000p-3, 0x1.3194ae0000000p-10, -0x1.c718860000000p-4, -0x1.10c16e0000000p-3, 0x1.a5c3da0000000p-3, -0x1.08c9c80000000p-5, -0x1.f997a40000000p-4, 0x1.c45fae0000000p-5, -0x1.032cea0000000p-1, 0x1.2eb1160000000p-4, 0x1.170a960000000p-6, 0x1.04ffa80000000p-4, -0x1.2f8aaa0000000p-4, 0x1.f0d6220000000p-4, 0x1.6d7fce0000000p-7, 0x1.5181ec0000000p-3, 0x1.508b880000000p-5, -0x1.65a99e0000000p-7, 0x1.7d27620000000p-3, 0x1.dc2bf20000000p-5, 0x1.55bb8a0000000p-5, 0x1.d547380000000p-6, -0x1.14f8f80000000p-7, -0x1.587aa80000000p-3, -0x1.0864de0000000p-3, 0x1.7abf860000000p-4, -0x1.1ee35c0000000p-4, 0x1.28c7ce0000000p-4, -0x1.0577940000000p-4, -0x1.b4d1b40000000p-4, 0x1.5aa36e0000000p-3, 0x1.94a4a40000000p-5}
, {-0x1.476d800000000p-6, -0x1.65876e0000000p-5, -0x1.dfee900000000p-5, 0x1.7d08c20000000p-3, 0x1.21d3fc0000000p-6, -0x1.4d74960000000p-6, -0x1.0e85280000000p-3, 0x1.3c72d80000000p-5, 0x1.da06280000000p-6, 0x1.1927200000000p-3, 0x1.fa32460000000p-4, -0x1.673b3e0000000p-4, -0x1.2959300000000p-5, 0x1.6a7ce20000000p-4, 0x1.c8a5500000000p-5, 0x1.d5bbc40000000p-5, 0x1.131bbc0000000p-4, -0x1.dae8da0000000p-3, 0x1.af99520000000p-7, -0x1.696c580000000p-4, -0x1.e8297c0000000p-7, 0x1.7ea86c0000000p-8, 0x1.7966cc0000000p-4, -0x1.0c3ad40000000p-5, 0x1.24e7cc0000000p-5, -0x1.a7122a0000000p-4, 0x1.32a22a0000000p-3, 0x1.c7ea1a0000000p-5, -0x1.ebeb860000000p-4, 0x1.a473100000000p-6, -0x1.8dd8760000000p-4, 0x1.3f701c0000000p-6, 0x1.a568180000000p-3, 0x1.cf79ca0000000p-6, -0x1.2eda6a0000000p-4, -0x1.5dbbd60000000p-4, -0x1.12bdfc0000000p-5, 0x1.b38cb00000000p-4, -0x1.b451340000000p-4, 0x1.489da20000000p-3, -0x1.a040860000000p-5, 0x1.e947ba0000000p-4, 0x1.662f7e0000000p-3, 0x1.a5094a0000000p-5, -0x1.a812ec0000000p-4, 0x1.20d72e0000000p-6, 0x1.1f12520000000p-7, 0x1.0ce9ec0000000p-2, 0x1.5373b60000000p-4, -0x1.3aa9660000000p-4, -0x1.4c12220000000p-6, 0x1.1c57f60000000p-5, 0x1.a7f72a0000000p-5, 0x1.14b77e0000000p-4, 0x1.2ac9fe0000000p-9, -0x1.3b65900000000p-9, 0x1.890f260000000p-4, -0x1.c9b9740000000p-5, -0x1.693dda0000000p-4, 0x1.e1de6e0000000p-4, 0x1.ce3b040000000p-4, -0x1.34e0180000000p-4, 0x1.f32b9c0000000p-4, 0x1.9945ac0000000p-5}
, {0x1.f5d53a0000000p-8, -0x1.95977e0000000p-5, -0x1.05af240000000p-5, 0x1.7c727a0000000p-3, 0x1.19cebe0000000p-5, -0x1.647ba20000000p-12, -0x1.bee1f60000000p-5, -0x1.b6cbaa0000000p-4, 0x1.1a05f20000000p-3, 0x1.a34c2c0000000p-3, 0x1.2e7bc40000000p-3, 0x1.cd4f720000000p-5, -0x1.a34c180000000p-3, -0x1.b22f0a0000000p-4, 0x1.aa20aa0000000p-7, -0x1.afb15a0000000p-5, 0x1.c105ea0000000p-5, -0x1.ad45540000000p-10, -0x1.cc928a0000000p-4, -0x1.565cdc0000000p-3, -0x1.10d5340000000p-5, -0x1.057d440000000p-6, 0x1.a8da2a0000000p-4, 0x1.cb65580000000p-4, -0x1.893b620000000p-4, -0x1.38ac740000000p-3, -0x1.0c3c6c0000000p-4, 0x1.2e71fe0000000p-4, -0x1.2af2280000000p-4, 0x1.02d58c0000000p-4, 0x1.4043ae0000000p-4, 0x1.2a58820000000p-4, 0x1.743cf20000000p-2, 0x1.319cec0000000p-5, -0x1.6288d20000000p-4, 0x1.f99c1c0000000p-3, 0x1.17827e0000000p-3, -0x1.904cfc0000000p-5, -0x1.f6a8b40000000p-4, 0x1.3ed8460000000p-3, 0x1.4c5ac80000000p-3, 0x1.3af0180000000p-3, 0x1.e859300000000p-6, -0x1.78e5de0000000p-5, -0x1.feb3900000000p-5, 0x1.0b7ba20000000p-3, 0x1.707fde0000000p-8, 0x1.cecf600000000p-3, 0x1.4c66e20000000p-5, 0x1.e254600000000p-5, 0x1.3cde2e0000000p-4, 0x1.951b0a0000000p-10, 0x1.e346860000000p-8, 0x1.bd64820000000p-3, 0x1.1eff500000000p-4, 0x1.7b3ab20000000p-3, 0x1.194ec80000000p-4, -0x1.6a29420000000p-5, -0x1.c8e5660000000p-7, 0x1.d0c3920000000p-3, 0x1.29c89c0000000p-2, 0x1.e1df700000000p-3, 0x1.be84420000000p-2, 0x1.f047100000000p-6}
}
, {{-0x1.99ed460000000p-3, 0x1.c733b40000000p-4, 0x1.0ee0a80000000p-3, -0x1.45f4c60000000p-5, 0x1.0f095c0000000p-3, 0x1.6b74e40000000p-7, -0x1.1b29720000000p-3, 0x1.4756360000000p-4, -0x1.c7ca740000000p-5, 0x1.daebd20000000p-7, 0x1.44d20c0000000p-7, 0x1.b8d66c0000000p-4, 0x1.bd18940000000p-3, 0x1.71efd00000000p-10, -0x1.df0e380000000p-3, -0x1.1d323c0000000p-5, 0x1.7922de0000000p-6, -0x1.3cb1240000000p-3, 0x1.03f0340000000p-3, -0x1.bd8c8e0000000p-4, 0x1.b071900000000p-6, 0x1.a6d5c40000000p-3, -0x1.53882c0000000p-4, 0x1.ac35740000000p-4, -0x1.417e3a0000000p-3, 0x1.62df100000000p-4, 0x1.1615680000000p-3, 0x1.4d33460000000p-3, 0x1.1c2ea80000000p-3, -0x1.8eee380000000p-10, -0x1.1904980000000p-3, 0x1.c336e60000000p-3, 0x1.6317fa0000000p-3, 0x1.7c9dd00000000p-4, 0x1.8b88ac0000000p-5, 0x1.2aaca00000000p-3, 0x1.ecaf2a0000000p-4, 0x1.5803360000000p-6, -0x1.a7e4fe0000000p-5, 0x1.f87c680000000p-4, 0x1.8f843e0000000p-4, 0x1.84200a0000000p-3, 0x1.3b35340000000p-2, -0x1.34da4c0000000p-4, -0x1.8ec85e0000000p-3, 0x1.2c30c80000000p-4, 0x1.7803560000000p-4, 0x1.595f0c0000000p-4, 0x1.b03d700000000p-5, -0x1.c719f60000000p-11, 0x1.6bac8e0000000p-6, 0x1.2210920000000p-3, 0x1.4235680000000p-3, 0x1.6281ec0000000p-5, -0x1.2602740000000p-3, 0x1.410b340000000p-8, 0x1.78e1ca0000000p-6, 0x1.9066280000000p-4, -0x1.531b6a0000000p-7, 0x1.1582560000000p-3, 0x1.c4656e0000000p-3, 0x1.455d8a0000000p-2, 0x1.8462540000000p-3, -0x1.62f8ee0000000p-3}
, {-0x1.2dfbc80000000p-2, 0x1.170c220000000p-4, -0x1.7054680000000p-5, -0x1.1d8e560000000p-5, -0x1.e730a40000000p-5, 0x1.3d6eaa0000000p-4, 0x1.220d320000000p-5, -0x1.02e49e0000000p-2, 0x1.be15200000000p-6, 0x1.7b75840000000p-5, 0x1.7cc4480000000p-5, 0x1.0a001e0000000p-4, 0x1.2adfb80000000p-3, -0x1.4033840000000p-3, -0x1.46e6360000000p-3, -0x1.ead4c80000000p-7, 0x1.cd40dc0000000p-4, -0x1.3bdc740000000p-4, 0x1.1b3dfa0000000p-4, -0x1.b5b9da0000000p-4, 0x1.203a640000000p-3, 0x1.6557e20000000p-3, -0x1.89ef800000000p-3, 0x1.7e681c0000000p-6, -0x1.c4267e0000000p-5, 0x1.9f417e0000000p-9, 0x1.61fa700000000p-4, -0x1.2d1e220000000p-5, 0x1.6c80480000000p-4, 0x1.ecfa2c0000000p-4, -0x1.9067fc0000000p-3, 0x1.2fb4180000000p-4, 0x1.2400320000000p-4, -0x1.64bc540000000p-8, -0x1.3cd1ce0000000p-5, 0x1.f83b080000000p-7, 0x1.77b4c00000000p-3, -0x1.2cc5200000000p-6, 0x1.4d2e480000000p-4, -0x1.25164c0000000p-4, 0x1.92142e0000000p-4, 0x1.51fa720000000p-4, 0x1.7dff060000000p-3, 0x1.abf3e40000000p-4, -0x1.12a75c0000000p-4, -0x1.1f750a0000000p-4, -0x1.ff6f860000000p-5, 0x1.4fff000000000p-3, -0x1.4d12d40000000p-5, -0x1.4a24ee0000000p-3, 0x1.76b9a80000000p-4, -0x1.1f145a0000000p-6, -0x1.b7466c0000000p-5, 0x1.387ec40000000p-5, 0x1.11383a0000000p-3, 0x1.10d8000000000p-3, 0x1.98b81c0000000p-4, -0x1.c84fe20000000p-6, -0x1.142d080000000p-5, 0x1.28b4480000000p-3, 0x1.4bb5000000000p-3, -0x1.c559980000000p-4, -0x1.56050c0000000p-6, -0x1.3201060000000p-7}
, {-0x1.1ee3e60000000p-2, 0x1.3a661a0000000p-4, 0x1.05e1b60000000p-9, 0x1.d304f60000000p-5, 0x1.d429e20000000p-5, 0x1.e7ab840000000p-6, 0x1.b2164c0000000p-8, -0x1.a78bb00000000p-3, -0x1.1ae6a60000000p-3, 0x1.02a3580000000p-3, -0x1.651a720000000p-4, -0x1.dc83400000000p-7, -0x1.9d80c20000000p-4, 0x1.91e3a20000000p-7, -0x1.bbc8580000000p-4, -0x1.7cb91e0000000p-4, 0x1.1729a20000000p-2, -0x1.f5122c0000000p-5, 0x1.9600860000000p-4, -0x1.6997da0000000p-3, 0x1.96c77a0000000p-5, 0x1.2a8cfc0000000p-3, -0x1.3aab980000000p-5, -0x1.242c500000000p-5, 0x1.d117c80000000p-5, 0x1.af75320000000p-5, 0x1.41c9040000000p-4, 0x1.4669500000000p-5, 0x1.740a960000000p-3, -0x1.02f0520000000p-3, 0x1.40f6920000000p-4, 0x1.ba675a0000000p-3, -0x1.889b9c0000000p-4, 0x1.5c393c0000000p-3, -0x1.b38d6c0000000p-3, -0x1.bf17a40000000p-8, 0x1.0c7dee0000000p-3, -0x1.afc0880000000p-4, -0x1.246bea0000000p-3, -0x1.3931c00000000p-5, -0x1.72880a0000000p-4, 0x1.856ffc0000000p-3, -0x1.0166de0000000p-13, -0x1.d37b200000000p-4, 0x1.47778a0000000p-2, -0x1.70d8980000000p-5, 0x1.227cde0000000p-3, -0x1.7a8ab80000000p-10, 0x1.3f75300000000p-3, 0x1.c9452c0000000p-7, 0x1.482f1c0000000p-5, 0x1.9256220000000p-6, 0x1.dff7360000000p-4, -0x1.697cfa0000000p-11, -0x1.0044c20000000p-3, 0x1.a14b220000000p-4, 0x1.95a74c0000000p-5, 0x1.2db7120000000p-5, 0x1.8bed2e0000000p-4, 0x1.14c66c0000000p-4, 0x1.057d6c0000000p-2, 0x1.87c3c20000000p-5, -0x1.dab4900000000p-5, -0x1.73f12a0000000p-4}
}
, {{0x1.684da20000000p-4, -0x1.53f39c0000000p-4, -0x1.e204c00000000p-4, 0x1.08fdb80000000p-6, -0x1.8738c40000000p-3, -0x1.6c4d8a0000000p-3, 0x1.a7135c0000000p-6, -0x1.e9dab40000000p-5, -0x1.1e14280000000p-5, 0x1.2fbd1a0000000p-3, 0x1.6db8100000000p-4, 0x1.7229fc0000000p-3, -0x1.98fe900000000p-6, -0x1.4e0d880000000p-3, 0x1.6b50c80000000p-4, 0x1.5b36fc0000000p-4, -0x1.acbcaa0000000p-4, 0x1.ae94b00000000p-4, -0x1.37ac500000000p-5, -0x1.5fe86a0000000p-7, -0x1.e982ec0000000p-6, -0x1.0f58aa0000000p-7, -0x1.89b5580000000p-4, -0x1.27f1120000000p-5, -0x1.43c1f00000000p-2, 0x1.474dc40000000p-6, 0x1.1245b80000000p-7, -0x1.501f760000000p-3, -0x1.ceb3780000000p-4, -0x1.173b680000000p-3, 0x1.4e24f00000000p-3, 0x1.fa9ba60000000p-5, 0x1.ef3c840000000p-9, -0x1.cada860000000p-5, -0x1.7102b20000000p-5, 0x1.696ff20000000p-3, -0x1.2dcb140000000p-6, -0x1.96d3360000000p-5, -0x1.b82fcc0000000p-3, -0x1.2083380000000p-5, -0x1.95f6960000000p-5, 0x1.73276a0000000p-4, -0x1.7e9d7a0000000p-3, -0x1.a5d6680000000p-3, 0x1.70c2e80000000p-4, 0x1.6722a20000000p-3, -0x1.97f1740000000p-7, -0x1.5032040000000p-4, -0x1.273f4a0000000p-3, -0x1.f1adee0000000p-6, 0x1.114dc00000000p-5, -0x1.42418a0000000p-8, 0x1.50b0b40000000p-4, 0x1.5451f00000000p-5, -0x1.d41ec60000000p-3, -0x1.30d58c0000000p-5, -0x1.c378e80000000p-12, -0x1.a688c00000000p-7, 0x1.c2939e0000000p-4, 0x1.8ba6fa0000000p-6, 0x1.5ed9980000000p-4, 0x1.ff01720000000p-3, -0x1.27096c0000000p-3, -0x1.b1cf560000000p-6}
, {0x1.7159e20000000p-5, 0x1.b404720000000p-4, -0x1.86f7100000000p-4, -0x1.1962740000000p-8, -0x1.6ac4a80000000p-4, 0x1.1062d40000000p-3, -0x1.2fcf680000000p-4, -0x1.e90b3e0000000p-7, -0x1.2e67200000000p-4, 0x1.b297520000000p-5, 0x1.04803a0000000p-4, 0x1.98fe5a0000000p-3, 0x1.6fb3700000000p-6, -0x1.5286460000000p-3, -0x1.11a28c0000000p-3, 0x1.0a95320000000p-5, 0x1.2d63cc0000000p-6, -0x1.81e5a20000000p-6, 0x1.21e00a0000000p-5, -0x1.26d8760000000p-6, 0x1.92cb3e0000000p-6, 0x1.b4c3ee0000000p-6, -0x1.8dc6640000000p-3, -0x1.4fbaa40000000p-4, -0x1.7331a00000000p-5, -0x1.306aea0000000p-7, 0x1.04d92e0000000p-7, 0x1.1df3ba0000000p-4, -0x1.820a3a0000000p-4, -0x1.8f2e540000000p-4, 0x1.37cee80000000p-5, -0x1.638da60000000p-5, -0x1.3553560000000p-4, -0x1.35c9e40000000p-3, -0x1.11a0de0000000p-4, 0x1.e025e40000000p-4, -0x1.3402b40000000p-7, -0x1.2938300000000p-2, -0x1.c75a940000000p-8, 0x1.15b8c40000000p-3, -0x1.9da6540000000p-5, -0x1.0ecbe60000000p-6, -0x1.5e0c740000000p-4, -0x1.2dd9de0000000p-3, 0x1.f7d7320000000p-7, 0x1.ff21760000000p-5, 0x1.0466fa0000000p-4, 0x1.a433b00000000p-5, 0x1.704afe0000000p-4, -0x1.1606dc0000000p-5, 0x1.f64aee0000000p-3, -0x1.13121a0000000p-4, 0x1.5c1cb60000000p-10, 0x1.2db6d80000000p-4, -0x1.905f640000000p-3, -0x1.5605040000000p-5, 0x1.f150e60000000p-6, 0x1.c273c20000000p-4, 0x1.fd18320000000p-4, 0x1.2040200000000p-6, -0x1.dbc1c20000000p-6, 0x1.2dfa5c0000000p-5, -0x1.049eca0000000p-4, -0x1.7614c60000000p-3}
, {0x1.e438d20000000p-5, 0x1.63e90c0000000p-6, -0x1.b090740000000p-4, 0x1.0d32e60000000p-3, 0x1.4f30ec0000000p-4, 0x1.84206e0000000p-6, -0x1.a861fe0000000p-7, -0x1.7003860000000p-4, -0x1.4338580000000p-3, 0x1.29c0500000000p-5, 0x1.77ee020000000p-5, 0x1.6a41c00000000p-2, -0x1.60fa080000000p-3, 0x1.71de700000000p-4, -0x1.8773d00000000p-3, 0x1.327c5c0000000p-4, -0x1.44114e0000000p-4, 0x1.e181320000000p-6, 0x1.27404e0000000p-4, 0x1.4797640000000p-7, -0x1.4ab3640000000p-3, 0x1.20cfea0000000p-5, 0x1.8610c80000000p-3, 0x1.4754be0000000p-4, -0x1.7202040000000p-3, -0x1.8e0e0c0000000p-4, -0x1.ed792c0000000p-4, -0x1.91d9b80000000p-4, -0x1.49ef140000000p-5, -0x1.cc19160000000p-3, -0x1.c2437e0000000p-7, 0x1.1223ca0000000p-9, -0x1.ee74660000000p-4, -0x1.40a9640000000p-6, -0x1.816bc20000000p-4, -0x1.a7cfc80000000p-3, 0x1.6fd3f20000000p-5, -0x1.6b12cc0000000p-2, 0x1.1a87f00000000p-3, -0x1.01d7240000000p-4, -0x1.cfc67c0000000p-5, 0x1.df1f700000000p-4, 0x1.40013e0000000p-8, -0x1.39be8a0000000p-5, -0x1.23c9ce0000000p-4, 0x1.10c80a0000000p-3, -0x1.4412300000000p-4, -0x1.bd6b320000000p-3, -0x1.84708c0000000p-3, 0x1.3629700000000p-2, 0x1.62e1900000000p-5, -0x1.1724c00000000p-3, -0x1.93e4180000000p-4, -0x1.67c7480000000p-4, -0x1.d119b40000000p-3, 0x1.4938d00000000p-3, -0x1.5433400000000p-3, -0x1.5ee21e0000000p-3, -0x1.d65d380000000p-6, 0x1.71a9e20000000p-5, -0x1.560cda0000000p-4, 0x1.6d53ae0000000p-3, -0x1.98a9980000000p-2, -0x1.1c10c20000000p-3}
}
, {{-0x1.82f5680000000p-5, -0x1.6f775c0000000p-3, 0x1.3f82cc0000000p-2, 0x1.ecfa520000000p-6, -0x1.3d605c0000000p-3, -0x1.2d3bac0000000p-3, -0x1.1d38ac0000000p-3, -0x1.dc82c80000000p-5, 0x1.ff7b0e0000000p-3, -0x1.bd0d760000000p-4, 0x1.115f580000000p-5, 0x1.19cbb80000000p-4, -0x1.2961ae0000000p-4, -0x1.cbe9680000000p-3, 0x1.5d84c40000000p-2, -0x1.2e5bf60000000p-6, 0x1.276d8c0000000p-3, 0x1.7528940000000p-4, -0x1.6b479c0000000p-4, -0x1.be6e1c0000000p-2, -0x1.1eff3a0000000p-2, -0x1.1d5e9e0000000p-2, -0x1.9041fa0000000p-6, -0x1.6fdfaa0000000p-4, 0x1.5ddbda0000000p-4, -0x1.bd29440000000p-3, -0x1.59b9140000000p-7, -0x1.6f77400000000p-5, -0x1.b803380000000p-4, -0x1.0987060000000p-4, -0x1.3531160000000p-3, -0x1.e82ba80000000p-3, 0x1.1900e00000000p-5, 0x1.e268800000000p-3, -0x1.8f7bc00000000p-5, 0x1.252b280000000p-3, 0x1.9ab9400000000p-5, 0x1.42451e0000000p-4, -0x1.1c09c40000000p-2, 0x1.3eea780000000p-4, -0x1.cfc1dc0000000p-2, 0x1.fc205a0000000p-4, 0x1.b5f4d00000000p-6, 0x1.3cc9200000000p-5, -0x1.47b8f80000000p-2, 0x1.2834b20000000p-3, -0x1.c9ebb40000000p-3, 0x1.42645c0000000p-1, 0x1.84c6440000000p-3, -0x1.00d0880000000p-3, 0x1.2a6ba20000000p-3, 0x1.1059140000000p-3, 0x1.3527e60000000p-4, 0x1.2a53840000000p-4, -0x1.e04c0a0000000p-3, 0x1.b195e60000000p-7, -0x1.cfe7a60000000p-4, -0x1.6eb6de0000000p-3, 0x1.3629ba0000000p-4, -0x1.8e48ea0000000p-5, -0x1.6073200000000p-4, -0x1.8a60c00000000p-5, 0x1.4e6cea0000000p-5, 0x1.04d46c0000000p-4}
, {0x1.b2d72e0000000p-3, -0x1.e2978a0000000p-7, -0x1.3f08120000000p-2, -0x1.750ade0000000p-4, 0x1.2c7ba00000000p-5, 0x1.8ccfd40000000p-6, 0x1.140fc80000000p-4, 0x1.5c77220000000p-6, -0x1.17f8c80000000p-4, -0x1.c2d7840000000p-4, 0x1.aca1d20000000p-4, -0x1.04809c0000000p-5, -0x1.216f620000000p-3, 0x1.3aaafc0000000p-4, -0x1.3e58fc0000000p-6, -0x1.a367f00000000p-6, 0x1.65ea3c0000000p-5, 0x1.5c42220000000p-3, -0x1.ba89c80000000p-10, -0x1.2540740000000p-4, -0x1.574d040000000p-3, 0x1.3318ec0000000p-5, -0x1.27d0f00000000p-2, -0x1.04c6060000000p-4, -0x1.cc92040000000p-4, 0x1.e8d6be0000000p-4, -0x1.6ffefa0000000p-3, -0x1.430c860000000p-2, 0x1.407a040000000p-11, 0x1.78beb60000000p-5, -0x1.4705ac0000000p-3, 0x1.fbf0f00000000p-5, 0x1.5f12060000000p-5, -0x1.eeddaa0000000p-3, -0x1.5714f00000000p-3, 0x1.cd872c0000000p-3, 0x1.288b780000000p-3, 0x1.0ec64e0000000p-4, -0x1.938a5a0000000p-4, -0x1.23ca200000000p-3, -0x1.f307500000000p-9, -0x1.2f22ce0000000p-3, -0x1.bc1ce60000000p-8, 0x1.76554e0000000p-3, -0x1.a6d0d00000000p-4, 0x1.adc7380000000p-5, -0x1.525a6e0000000p-5, 0x1.26ae140000000p-2, -0x1.740c0e0000000p-5, 0x1.e26e4a0000000p-7, 0x1.7f40360000000p-10, -0x1.59b8140000000p-5, -0x1.bf79e60000000p-5, -0x1.c788e80000000p-4, 0x1.7c78b00000000p-4, 0x1.2ee51a0000000p-4, 0x1.5de8520000000p-5, -0x1.3fca500000000p-4, 0x1.35bcba0000000p-5, -0x1.7bd37e0000000p-3, -0x1.4da5920000000p-6, -0x1.1961ee0000000p-4, -0x1.b4164e0000000p-5, 0x1.4aade40000000p-4}
, {0x1.3af3e40000000p-4, 0x1.eebd260000000p-4, 0x1.e364f40000000p-7, 0x1.0cf6020000000p-4, 0x1.2e982e0000000p-3, -0x1.abe7b60000000p-4, 0x1.a1d2060000000p-2, 0x1.e4aed40000000p-4, 0x1.6db1220000000p-3, -0x1.50182c0000000p-3, 0x1.4642380000000p-5, -0x1.0647480000000p-3, 0x1.1a6fd20000000p-6, 0x1.917f6e0000000p-5, 0x1.61294a0000000p-4, 0x1.71013e0000000p-5, -0x1.b023bc0000000p-5, -0x1.f991120000000p-4, -0x1.1db26c0000000p-8, -0x1.c3799c0000000p-8, -0x1.ed3da80000000p-6, -0x1.9a68e20000000p-4, -0x1.0ad9e40000000p-5, -0x1.445ad60000000p-6, -0x1.a72b220000000p-5, 0x1.5c09960000000p-4, 0x1.0e4f0c0000000p-4, 0x1.71f9ee0000000p-3, -0x1.cce1da0000000p-5, -0x1.6626940000000p-5, 0x1.7faa800000000p-7, 0x1.ac83260000000p-7, -0x1.0e6c120000000p-4, -0x1.96622a0000000p-4, -0x1.3b86240000000p-3, 0x1.84ce320000000p-5, 0x1.e45ed40000000p-6, -0x1.17a07c0000000p-3, 0x1.79fd840000000p-4, -0x1.fbcada0000000p-7, -0x1.0489120000000p-4, -0x1.9893420000000p-3, 0x1.2fb81c0000000p-10, -0x1.16f6d20000000p-6, 0x1.2d8d0c0000000p-4, 0x1.144fde0000000p-4, -0x1.fda4a00000000p-4, -0x1.96df0a0000000p-3, -0x1.af9bf20000000p-4, -0x1.b31b4c0000000p-5, 0x1.a3ae220000000p-4, 0x1.6270bc0000000p-5, -0x1.a8d9c60000000p-3, -0x1.945d8a0000000p-3, 0x1.ba05aa0000000p-5, -0x1.046c220000000p-4, -0x1.4642f40000000p-3, 0x1.bb24b60000000p-2, -0x1.1d28680000000p-6, -0x1.e9a3880000000p-4, -0x1.e6954a0000000p-5, -0x1.2db2900000000p-4, -0x1.22b4b00000000p-6, -0x1.afccee0000000p-7}
}
, {{-0x1.082c420000000p-3, 0x1.6d6cbe0000000p-3, 0x1.9cb7760000000p-4, 0x1.0c4c3e0000000p-3, 0x1.de90e60000000p-5, 0x1.b2ba1c0000000p-4, -0x1.a974d20000000p-4, 0x1.1520180000000p-6, -0x1.97bda80000000p-8, -0x1.df603e0000000p-6, 0x1.4bf20c0000000p-4, 0x1.c4f8520000000p-7, 0x1.1d5dc60000000p-2, -0x1.2ad9420000000p-3, 0x1.73493c0000000p-6, 0x1.d131120000000p-7, 0x1.186f680000000p-4, 0x1.b0fa320000000p-5, -0x1.024e2c0000000p-3, -0x1.9dfeae0000000p-4, 0x1.83a8f40000000p-7, 0x1.697c0e0000000p-4, 0x1.5b548a0000000p-4, -0x1.e49a200000000p-5, -0x1.c6524e0000000p-5, 0x1.d072b60000000p-4, 0x1.bd5d380000000p-2, -0x1.878dc40000000p-5, -0x1.d458a20000000p-5, -0x1.1293ea0000000p-5, 0x1.444bfa0000000p-5, 0x1.696d640000000p-4, 0x1.9fd25a0000000p-3, 0x1.e40cf80000000p-3, 0x1.bbcd380000000p-3, 0x1.be59580000000p-3, 0x1.7f534e0000000p-4, 0x1.b1a13c0000000p-4, -0x1.0efeb00000000p-4, 0x1.45037e0000000p-2, 0x1.f1a6dc0000000p-6, -0x1.722c6c0000000p-7, -0x1.ffd0120000000p-4, -0x1.d494000000000p-3, -0x1.d8a6cc0000000p-5, 0x1.0bf1c80000000p-2, -0x1.70167e0000000p-3, 0x1.50ca100000000p-3, 0x1.237cfc0000000p-3, -0x1.3670340000000p-4, 0x1.7e27e40000000p-2, 0x1.5a8f260000000p-3, 0x1.3e2e080000000p-4, 0x1.b7bc7a0000000p-5, -0x1.b9354c0000000p-6, 0x1.0fe10c0000000p-4, 0x1.87c7f00000000p-6, 0x1.66ecaa0000000p-3, 0x1.06295e0000000p-4, 0x1.c34b400000000p-4, 0x1.d3d2040000000p-5, 0x1.82aec40000000p-9, 0x1.4bac300000000p-4, -0x1.e4f86c0000000p-6}
, {-0x1.3c90b00000000p-4, 0x1.79d90c0000000p-4, -0x1.34b35a0000000p-3, 0x1.f5447c0000000p-6, -0x1.0c47160000000p-4, 0x1.24f4440000000p-3, 0x1.1d7bc20000000p-3, 0x1.d7a8540000000p-7, -0x1.198b980000000p-3, 0x1.86fbac0000000p-5, 0x1.8174320000000p-10, 0x1.51fc660000000p-3, 0x1.0702ca0000000p-4, 0x1.1dc7140000000p-4, 0x1.03b1880000000p-4, 0x1.28b8640000000p-3, 0x1.6b8f240000000p-4, 0x1.65fc8a0000000p-6, -0x1.01bf6e0000000p-5, -0x1.0312900000000p-5, -0x1.22789c0000000p-4, 0x1.3c32de0000000p-3, -0x1.c361bc0000000p-4, -0x1.a92bb80000000p-5, -0x1.50a0820000000p-3, 0x1.4f9ad80000000p-5, 0x1.dcfa500000000p-3, -0x1.562fce0000000p-5, 0x1.12b51c0000000p-3, 0x1.e45ec80000000p-5, 0x1.2d57520000000p-5, 0x1.c814d20000000p-4, 0x1.655db00000000p-4, -0x1.b056760000000p-7, 0x1.154aa40000000p-4, 0x1.4b61a80000000p-4, 0x1.ab1b2c0000000p-3, 0x1.99de3a0000000p-4, -0x1.606cd60000000p-5, 0x1.67c9580000000p-3, 0x1.31fc5e0000000p-7, 0x1.1738600000000p-3, -0x1.bffffe0000000p-6, 0x1.2c00360000000p-5, -0x1.896a0c0000000p-6, 0x1.e77f2c0000000p-3, -0x1.a1a9b00000000p-5, -0x1.9b35da0000000p-3, -0x1.8a55020000000p-9, -0x1.af4d420000000p-4, 0x1.99cedc0000000p-3, 0x1.d1d6340000000p-4, 0x1.c9124e0000000p-6, 0x1.402c460000000p-3, 0x1.a539580000000p-4, 0x1.81db880000000p-5, 0x1.073aca0000000p-2, -0x1.7987460000000p-4, 0x1.7a7e7a0000000p-8, -0x1.648d660000000p-5, 0x1.1948a20000000p-3, -0x1.e9e8300000000p-4, 0x1.3985b20000000p-4, 0x1.901dea0000000p-4}
, {-0x1.5c8c200000000p-4, 0x1.0069160000000p-3, -0x1.2015700000000p-3, -0x1.1a29240000000p-5, -0x1.ae99680000000p-6, -0x1.b879880000000p-4, 0x1.b899460000000p-3, -0x1.b572100000000p-11, 0x1.6cd8c60000000p-4, 0x1.de73840000000p-5, 0x1.9b66460000000p-5, -0x1.054e280000000p-6, -0x1.891cae0000000p-3, 0x1.5671ce0000000p-3, -0x1.39747c0000000p-3, 0x1.4e77a20000000p-3, 0x1.ca3f720000000p-8, -0x1.e98baa0000000p-5, -0x1.8009e20000000p-7, 0x1.34c9e20000000p-5, -0x1.4b9c400000000p-6, 0x1.54e76e0000000p-4, -0x1.0912ae0000000p-4, 0x1.aa13120000000p-4, 0x1.f9c8520000000p-6, -0x1.1e50ae0000000p-3, 0x1.89526e0000000p-5, -0x1.ba57240000000p-6, 0x1.a6aa3c0000000p-5, 0x1.6d6ed00000000p-4, -0x1.13affe0000000p-4, 0x1.200bce0000000p-4, 0x1.6d8f360000000p-4, 0x1.0efd140000000p-3, -0x1.21fc9a0000000p-5, 0x1.4628740000000p-3, 0x1.c399ce0000000p-4, 0x1.93b62c0000000p-4, -0x1.c092da0000000p-6, 0x1.13a5b20000000p-6, -0x1.5c2c760000000p-6, 0x1.87952e0000000p-5, 0x1.c25fe80000000p-4, 0x1.8960e80000000p-4, -0x1.2793cc0000000p-8, 0x1.9d4d9c0000000p-8, 0x1.db399a0000000p-3, -0x1.7499780000000p-5, -0x1.bab9560000000p-8, -0x1.1534260000000p-3, 0x1.b5fb3a0000000p-6, -0x1.dc9b1c0000000p-5, 0x1.9f34aa0000000p-9, 0x1.5b091c0000000p-3, -0x1.8037a60000000p-4, 0x1.3081680000000p-4, 0x1.4248180000000p-4, 0x1.92eb200000000p-7, -0x1.ab20460000000p-4, 0x1.6349a00000000p-4, 0x1.60acd80000000p-4, -0x1.5472e00000000p-6, -0x1.2b0a180000000p-7, 0x1.9276420000000p-3}
}
, {{0x1.4391bc0000000p-3, -0x1.0e8cb80000000p-5, -0x1.b796820000000p-4, -0x1.30e2820000000p-4, -0x1.5ff1dc0000000p-3, 0x1.9d6a120000000p-3, -0x1.75c3dc0000000p-4, 0x1.3e207e0000000p-3, -0x1.ba594a0000000p-3, -0x1.0727220000000p-5, 0x1.94f52c0000000p-3, 0x1.de770e0000000p-3, 0x1.0ac0520000000p-5, -0x1.b63bcc0000000p-3, -0x1.bac1ac0000000p-3, 0x1.a28e080000000p-4, 0x1.4bae200000000p-4, -0x1.db98020000000p-4, 0x1.945f4a0000000p-4, -0x1.06cfa60000000p-6, -0x1.e063200000000p-5, 0x1.d6aa5a0000000p-3, -0x1.6cf16c0000000p-3, 0x1.89b5460000000p-4, -0x1.00c5900000000p-3, -0x1.0d65f60000000p-5, 0x1.fc6d440000000p-7, 0x1.7f844c0000000p-3, -0x1.0f5d960000000p-6, 0x1.dc98c20000000p-5, -0x1.0843700000000p-4, 0x1.7618140000000p-3, 0x1.65bf180000000p-7, -0x1.4cfb040000000p-12, 0x1.a926e60000000p-4, 0x1.e154540000000p-3, 0x1.00fd9e0000000p-3, -0x1.2f05d80000000p-4, 0x1.94ef9e0000000p-5, 0x1.6f879c0000000p-4, 0x1.4d4ef60000000p-6, 0x1.cdbbce0000000p-5, -0x1.048a0c0000000p-3, -0x1.22b7d20000000p-3, 0x1.888ca80000000p-7, 0x1.2cbadc0000000p-4, 0x1.fab6c80000000p-3, -0x1.5219aa0000000p-7, 0x1.cd66b00000000p-3, 0x1.ea95800000000p-4, 0x1.d05eea0000000p-3, -0x1.814c3a0000000p-3, 0x1.57f5ce0000000p-3, 0x1.6fb32c0000000p-4, 0x1.12e99e0000000p-4, 0x1.5d91c20000000p-4, -0x1.e3b3880000000p-5, -0x1.5011960000000p-3, 0x1.b3d0a80000000p-4, 0x1.228a8c0000000p-3, 0x1.3ceda60000000p-3, 0x1.93dae80000000p-4, 0x1.34ff420000000p-2, -0x1.d7281e0000000p-3}
, {-0x1.5d30ec0000000p-4, -0x1.a146f20000000p-8, -0x1.e7b20e0000000p-5, -0x1.5c7d9c0000000p-3, -0x1.9817820000000p-6, 0x1.18e9c60000000p-5, 0x1.a284980000000p-9, -0x1.c910840000000p-3, 0x1.6aba9c0000000p-4, -0x1.8a78920000000p-5, 0x1.3b6ea40000000p-5, -0x1.b77c580000000p-8, 0x1.26c0280000000p-6, -0x1.5410d80000000p-4, -0x1.26a6e00000000p-5, 0x1.27a8400000000p-4, 0x1.3cd7000000000p-5, -0x1.d4d5800000000p-11, 0x1.9a0b360000000p-4, -0x1.1131d20000000p-6, 0x1.169b400000000p-6, 0x1.7103060000000p-4, -0x1.0115ec0000000p-3, 0x1.cba8c60000000p-4, -0x1.7a1c160000000p-4, 0x1.f670840000000p-6, -0x1.c350780000000p-15, 0x1.068d040000000p-6, 0x1.fd98c00000000p-4, 0x1.7337380000000p-4, -0x1.08e7200000000p-3, 0x1.ab9d5c0000000p-3, 0x1.3743d00000000p-4, -0x1.ac06780000000p-5, 0x1.1ece8a0000000p-4, 0x1.6649b60000000p-4, -0x1.9ed3240000000p-4, 0x1.2ef4980000000p-7, 0x1.5f213e0000000p-4, -0x1.27ac6c0000000p-5, -0x1.f7ae680000000p-11, 0x1.b24f760000000p-5, -0x1.8827c20000000p-7, -0x1.7f71860000000p-4, -0x1.74666c0000000p-4, 0x1.30408a0000000p-3, 0x1.60d15a0000000p-4, -0x1.1292840000000p-3, 0x1.7f4d9e0000000p-3, 0x1.d9d81e0000000p-6, -0x1.0f58a80000000p-3, -0x1.1d84c60000000p-4, 0x1.169ec60000000p-3, -0x1.d003d00000000p-6, -0x1.14b2100000000p-4, 0x1.771d8a0000000p-3, 0x1.023e900000000p-4, 0x1.903b260000000p-6, 0x1.f5a5c00000000p-5, -0x1.11f8000000000p-3, 0x1.11ddd00000000p-4, 0x1.cac5700000000p-4, -0x1.93373e0000000p-4, -0x1.0c22900000000p-7}
, {-0x1.0a07840000000p-3, 0x1.1ca8b00000000p-4, 0x1.000a940000000p-4, -0x1.f55f160000000p-6, -0x1.0f7bd00000000p-5, -0x1.3cbaaa0000000p-4, 0x1.7e51d80000000p-5, -0x1.0f33ca0000000p-4, 0x1.5ecb960000000p-3, -0x1.e611fa0000000p-5, -0x1.3464380000000p-4, 0x1.87d7ac0000000p-4, 0x1.84eeb60000000p-3, -0x1.b3393c0000000p-5, -0x1.5041a40000000p-5, 0x1.0c876e0000000p-4, 0x1.8ea8780000000p-4, -0x1.2cb01a0000000p-3, 0x1.2283ca0000000p-3, -0x1.a38fe00000000p-5, 0x1.99109a0000000p-3, -0x1.6854000000000p-4, -0x1.8523b00000000p-6, 0x1.6b55d80000000p-3, -0x1.fc744a0000000p-5, 0x1.3a26300000000p-5, -0x1.a3a4180000000p-7, 0x1.4a60a60000000p-5, -0x1.4cfb440000000p-10, 0x1.023c700000000p-5, -0x1.3477f00000000p-3, 0x1.d46e800000000p-6, -0x1.c134100000000p-5, 0x1.76793e0000000p-4, 0x1.6914aa0000000p-4, 0x1.4e2b740000000p-6, -0x1.ec539e0000000p-4, 0x1.27f4480000000p-3, 0x1.17df0c0000000p-3, 0x1.4d424a0000000p-3, 0x1.f22ed60000000p-7, -0x1.52509a0000000p-5, 0x1.7dd0660000000p-7, -0x1.3a74ba0000000p-3, 0x1.9767b40000000p-5, 0x1.0ad06a0000000p-6, -0x1.f3ad8a0000000p-5, -0x1.0c7b620000000p-3, 0x1.a26c2c0000000p-3, -0x1.e02bba0000000p-6, 0x1.00fdc00000000p-19, 0x1.20894a0000000p-3, -0x1.17ae720000000p-3, 0x1.47a07e0000000p-3, 0x1.a53b2a0000000p-5, -0x1.98c6e60000000p-3, -0x1.19954a0000000p-5, 0x1.1476380000000p-4, 0x1.2ce3c80000000p-3, 0x1.46ec080000000p-4, -0x1.853ade0000000p-4, 0x1.e530f60000000p-4, 0x1.2524a60000000p-5, -0x1.70c3ba0000000p-8}
}
, {{0x1.e910ae0000000p-8, -0x1.0c86960000000p-3, 0x1.3508880000000p-2, -0x1.9fb9100000000p-4, -0x1.b7cfd60000000p-3, 0x1.d8585a0000000p-5, -0x1.fbfc040000000p-8, 0x1.8a97120000000p-3, 0x1.dd7aae0000000p-3, 0x1.6dec020000000p-4, 0x1.b79fb20000000p-5, 0x1.86e80e0000000p-4, -0x1.9e1dca0000000p-4, -0x1.6f91b00000000p-4, -0x1.7794d00000000p-8, 0x1.d68d180000000p-8, -0x1.35c30e0000000p-4, 0x1.03d6c00000000p-3, -0x1.907b780000000p-4, -0x1.9dfcd20000000p-4, 0x1.0046380000000p-3, -0x1.c3eb900000000p-4, -0x1.a0de080000000p-6, 0x1.04c5e80000000p-6, -0x1.9f35ca0000000p-3, -0x1.c583d60000000p-4, 0x1.fca96c0000000p-5, 0x1.335c360000000p-5, -0x1.24735e0000000p-3, 0x1.0314b40000000p-9, -0x1.e4e35c0000000p-5, -0x1.42440c0000000p-2, -0x1.a46c3e0000000p-3, -0x1.3151180000000p-6, 0x1.5e86da0000000p-4, 0x1.e0bd0c0000000p-4, 0x1.27d0ae0000000p-6, -0x1.1356960000000p-3, 0x1.8a24760000000p-3, 0x1.03f2880000000p-2, -0x1.7e3d700000000p-2, 0x1.9985040000000p-3, 0x1.64daf80000000p-5, 0x1.85b1da0000000p-7, 0x1.6f06720000000p-4, 0x1.a225d20000000p-7, -0x1.80e6d00000000p-3, 0x1.75d1c80000000p-4, -0x1.54dda40000000p-8, -0x1.7947380000000p-4, 0x1.2ab1820000000p-4, 0x1.3892ca0000000p-5, 0x1.257a700000000p-3, 0x1.c5b0e40000000p-4, -0x1.72b5960000000p-4, -0x1.77422c0000000p-3, -0x1.5ce59c0000000p-3, 0x1.509b680000000p-4, -0x1.9675f40000000p-3, -0x1.5b99740000000p-3, -0x1.9fe1d60000000p-3, 0x1.e507aa0000000p-9, -0x1.bf91de0000000p-6, -0x1.5bf5660000000p-4}
, {-0x1.3046440000000p-7, 0x1.cd61980000000p-4, 0x1.413ece0000000p-4, -0x1.da606a0000000p-5, -0x1.9f39380000000p-8, 0x1.22cda20000000p-9, 0x1.55ae160000000p-6, -0x1.9525100000000p-5, 0x1.843a180000000p-4, 0x1.9ec0640000000p-4, 0x1.6c2de60000000p-9, 0x1.f2dd520000000p-5, 0x1.e35e020000000p-5, -0x1.04f58c0000000p-5, 0x1.6a8a200000000p-3, 0x1.e636d00000000p-4, -0x1.91cc0e0000000p-4, -0x1.7d93920000000p-4, -0x1.06f21e0000000p-3, -0x1.166b160000000p-4, -0x1.ab9eca0000000p-4, 0x1.98b3da0000000p-4, 0x1.a61a500000000p-4, -0x1.1858060000000p-6, -0x1.2dd8680000000p-4, -0x1.0ed16c0000000p-3, 0x1.e878860000000p-5, -0x1.a6085e0000000p-4, -0x1.b728ac0000000p-4, -0x1.ac75920000000p-4, 0x1.1432380000000p-5, -0x1.78a3a00000000p-6, -0x1.3775600000000p-5, -0x1.a6c1de0000000p-5, 0x1.4ee6460000000p-3, -0x1.d6a80e0000000p-3, -0x1.70d8f40000000p-3, 0x1.398c400000000p-4, 0x1.1372dc0000000p-4, 0x1.2880da0000000p-6, 0x1.2e27e00000000p-3, 0x1.92d5c00000000p-4, -0x1.ec38bc0000000p-3, -0x1.f16a980000000p-5, 0x1.690b6c0000000p-3, 0x1.7f54300000000p-9, 0x1.af7cc20000000p-6, 0x1.0d90c40000000p-5, -0x1.42432c0000000p-4, 0x1.f02a8e0000000p-4, -0x1.000e6e0000000p-3, -0x1.87c0f00000000p-3, 0x1.782f320000000p-6, -0x1.2848120000000p-8, -0x1.0ddcd80000000p-4, -0x1.3b69e80000000p-2, 0x1.3ce3920000000p-5, -0x1.9226620000000p-4, 0x1.074d040000000p-3, -0x1.11a6200000000p-5, -0x1.162ab00000000p-4, 0x1.fe23e60000000p-4, 0x1.b8f83e0000000p-6, -0x1.2b9f720000000p-3}
, {0x1.1c398c0000000p-4, 0x1.77490e0000000p-5, 0x1.06e63e0000000p-3, -0x1.43cc480000000p-5, -0x1.a169760000000p-4, 0x1.42f5480000000p-5, -0x1.d6f6f60000000p-3, 0x1.85ebf00000000p-7, 0x1.1d38840000000p-3, 0x1.53b7a60000000p-3, 0x1.b5fe5c0000000p-6, 0x1.63ce8e0000000p-4, 0x1.47120c0000000p-3, -0x1.1112ba0000000p-4, 0x1.74abb60000000p-3, 0x1.8bf0020000000p-6, -0x1.7081d40000000p-8, 0x1.c774f80000000p-7, -0x1.6265920000000p-5, -0x1.8e1b900000000p-4, 0x1.d7b1880000000p-4, -0x1.b1a9720000000p-4, 0x1.f1a54c0000000p-7, -0x1.7f2c440000000p-5, -0x1.0241ea0000000p-3, 0x1.2518c40000000p-6, 0x1.06122c0000000p-4, -0x1.a8f18e0000000p-4, 0x1.533dac0000000p-5, 0x1.05ddd40000000p-7, 0x1.20c8ee0000000p-2, -0x1.b187700000000p-3, -0x1.8b0e4e0000000p-3, 0x1.cb16040000000p-7, 0x1.89bab80000000p-3, 0x1.3748780000000p-4, -0x1.8ca7a00000000p-4, 0x1.f1002a0000000p-4, -0x1.aa3b660000000p-4, 0x1.7cafea0000000p-4, -0x1.9bf2080000000p-5, 0x1.b87f9c0000000p-5, -0x1.806b640000000p-3, 0x1.2d1ddc0000000p-3, 0x1.14daa20000000p-4, -0x1.c9277a0000000p-5, 0x1.20cc700000000p-5, 0x1.1fc6880000000p-6, -0x1.01a8640000000p-4, -0x1.5872de0000000p-3, 0x1.15fc860000000p-3, -0x1.90fe2e0000000p-3, 0x1.0a5b7e0000000p-4, 0x1.aead560000000p-6, -0x1.3a3bc00000000p-6, -0x1.50267a0000000p-1, 0x1.82e96e0000000p-5, 0x1.f645d80000000p-7, 0x1.c84eb40000000p-8, -0x1.6e79520000000p-3, -0x1.43fd5c0000000p-2, 0x1.60bc540000000p-3, -0x1.fef7fe0000000p-3, -0x1.cd05360000000p-5}
}
, {{-0x1.b9327e0000000p-5, 0x1.de94fc0000000p-5, 0x1.65f4d80000000p-3, -0x1.f052f40000000p-3, -0x1.35745a0000000p-8, -0x1.1dbcde0000000p-3, 0x1.8e93880000000p-3, 0x1.1d48fe0000000p-3, 0x1.f299a20000000p-4, -0x1.d4b8380000000p-4, 0x1.7ba9040000000p-2, 0x1.0102600000000p-6, -0x1.330dce0000000p-5, -0x1.2b84080000000p-10, 0x1.3d70ea0000000p-5, 0x1.556f760000000p-3, 0x1.bd20240000000p-5, 0x1.6187a20000000p-3, -0x1.ea13820000000p-5, 0x1.6b03120000000p-3, 0x1.57689a0000000p-4, -0x1.5396f20000000p-7, -0x1.f9634e0000000p-4, 0x1.56195c0000000p-4, -0x1.9c153e0000000p-3, -0x1.49bc3a0000000p-4, -0x1.91ab5c0000000p-4, -0x1.3f0cda0000000p-3, -0x1.13acd00000000p-3, -0x1.95643e0000000p-3, 0x1.23f6fa0000000p-3, -0x1.72f2cc0000000p-7, -0x1.12f0240000000p-3, -0x1.1ec41c0000000p-2, -0x1.e69ea00000000p-4, 0x1.ed513e0000000p-5, 0x1.b20fca0000000p-6, 0x1.778fca0000000p-6, -0x1.c860d00000000p-4, -0x1.1fd74a0000000p-3, -0x1.5c01be0000000p-5, -0x1.f88cc20000000p-5, -0x1.30ccd80000000p-5, 0x1.6faa120000000p-8, -0x1.c88fbc0000000p-4, 0x1.770ae20000000p-3, 0x1.9f5dc40000000p-5, 0x1.cc8c160000000p-5, -0x1.f76cd00000000p-4, 0x1.0629f40000000p-3, 0x1.230be00000000p-2, 0x1.843fd60000000p-4, 0x1.2b6c3c0000000p-3, 0x1.5eafa20000000p-10, -0x1.6226320000000p-2, 0x1.262cb60000000p-4, -0x1.dc67400000000p-6, 0x1.3cacfa0000000p-3, -0x1.1148320000000p-2, -0x1.64cb1a0000000p-3, 0x1.346db40000000p-3, -0x1.0145c20000000p-4, -0x1.26ac780000000p-9, -0x1.8f971e0000000p-3}
, {0x1.dfb9d80000000p-6, 0x1.70a2a60000000p-9, -0x1.475bda0000000p-3, -0x1.d4954a0000000p-6, 0x1.49bbea0000000p-7, -0x1.7968500000000p-6, 0x1.518b600000000p-5, -0x1.75696c0000000p-3, -0x1.283f020000000p-4, -0x1.cc51760000000p-6, -0x1.878bd00000000p-7, -0x1.92125c0000000p-3, -0x1.fa28c40000000p-7, 0x1.d68f660000000p-6, -0x1.2554b20000000p-3, 0x1.9d67440000000p-10, -0x1.191edc0000000p-4, -0x1.409ccc0000000p-7, -0x1.ddd2aa0000000p-3, 0x1.5f643c0000000p-3, -0x1.789e920000000p-3, -0x1.b02d880000000p-4, -0x1.35d0260000000p-3, -0x1.b03f540000000p-5, -0x1.6fbafc0000000p-3, -0x1.9d0b340000000p-7, -0x1.edc33c0000000p-5, -0x1.931d0e0000000p-7, -0x1.d6c36c0000000p-5, -0x1.a5e76c0000000p-5, 0x1.274f560000000p-4, -0x1.052ba60000000p-3, -0x1.7739680000000p-8, -0x1.89186c0000000p-3, -0x1.fe94880000000p-4, 0x1.3d30c00000000p-5, 0x1.3ac85e0000000p-2, 0x1.99de600000000p-4, -0x1.7774520000000p-3, -0x1.62a55a0000000p-5, 0x1.e1e4380000000p-4, -0x1.799e680000000p-5, -0x1.69aa480000000p-4, -0x1.a387620000000p-5, -0x1.38bbfc0000000p-5, 0x1.0c46d40000000p-5, -0x1.c6fec00000000p-4, -0x1.1a2cf20000000p-7, -0x1.39c6660000000p-3, 0x1.704da00000000p-8, 0x1.67ee6c0000000p-9, -0x1.fbbcee0000000p-6, -0x1.ebc3fc0000000p-5, -0x1.51ee9c0000000p-6, -0x1.54575a0000000p-4, 0x1.ebe17e0000000p-6, 0x1.56c0b20000000p-3, 0x1.c2b8a80000000p-6, -0x1.11f9660000000p-3, 0x1.7616de0000000p-4, 0x1.9736440000000p-5, 0x1.7f75d80000000p-5, -0x1.5c96e80000000p-6, 0x1.fd9b7c0000000p-7}
, {-0x1.a82ac80000000p-3, 0x1.6493860000000p-4, 0x1.e2ae340000000p-3, 0x1.46e1460000000p-4, 0x1.49e0d80000000p-5, 0x1.aeae6c0000000p-5, -0x1.6a7c4e0000000p-3, 0x1.b012d80000000p-6, 0x1.2321e60000000p-2, 0x1.43b8c40000000p-4, -0x1.6d4b320000000p-3, 0x1.36389e0000000p-3, 0x1.16c8f80000000p-4, -0x1.eacdc20000000p-5, -0x1.aedc920000000p-3, -0x1.9e17c80000000p-4, -0x1.0534fc0000000p-1, -0x1.5c7dfc0000000p-3, 0x1.333c060000000p-2, 0x1.dade3e0000000p-5, -0x1.b1fda80000000p-9, 0x1.39471c0000000p-2, 0x1.64a2f60000000p-2, 0x1.c4a4720000000p-3, -0x1.ba43ba0000000p-6, 0x1.ac8ef80000000p-2, 0x1.2fb8500000000p-3, 0x1.1b6b880000000p-3, -0x1.54458a0000000p-3, 0x1.afbc460000000p-7, -0x1.08b55c0000000p-3, -0x1.4012f60000000p-3, -0x1.1899640000000p-3, 0x1.8d8de20000000p-4, 0x1.4373340000000p-5, 0x1.6dbd620000000p-3, -0x1.6544be0000000p-2, 0x1.d23e900000000p-3, 0x1.1b29280000000p-4, 0x1.56f6900000000p-4, -0x1.58bc520000000p-2, -0x1.64d33c0000000p-3, -0x1.91f2ba0000000p-7, 0x1.8a65ce0000000p-4, -0x1.357b0a0000000p-2, 0x1.8e02ac0000000p-6, 0x1.2818200000000p-4, 0x1.e8836c0000000p-3, 0x1.0861e20000000p-3, 0x1.1e398e0000000p-4, 0x1.5109880000000p-2, 0x1.8c5dbc0000000p-5, -0x1.3d65280000000p-4, -0x1.a8cfce0000000p-2, 0x1.4b99be0000000p-3, -0x1.8bbb580000000p-2, -0x1.eeb0040000000p-3, 0x1.a5e58c0000000p-2, -0x1.ddbce20000000p-6, 0x1.18beac0000000p-6, -0x1.f87df00000000p-3, -0x1.74f9fe0000000p-4, -0x1.143e720000000p-3, 0x1.08d8ce0000000p-5}
}
, {{0x1.458cca0000000p-5, 0x1.0e9bbc0000000p-4, -0x1.4a07b60000000p-8, 0x1.f77c660000000p-5, 0x1.59ed9a0000000p-6, 0x1.f7a4560000000p-6, 0x1.859a040000000p-4, -0x1.a2a34e0000000p-7, 0x1.c56c0c0000000p-5, 0x1.efc94a0000000p-4, 0x1.32672c0000000p-3, -0x1.1676880000000p-3, 0x1.5516c20000000p-3, -0x1.fa77960000000p-6, -0x1.4c7b500000000p-3, -0x1.de22740000000p-3, 0x1.e412d60000000p-7, -0x1.1dd8620000000p-3, 0x1.5289040000000p-3, -0x1.84d11e0000000p-5, -0x1.003d580000000p-8, 0x1.15d2160000000p-7, -0x1.58cc800000000p-3, 0x1.1da9a80000000p-3, 0x1.25cf100000000p-5, -0x1.2620400000000p-8, 0x1.9fc79c0000000p-5, 0x1.df91540000000p-3, 0x1.db7e300000000p-5, 0x1.cf622e0000000p-4, -0x1.9ad97a0000000p-5, -0x1.c6c1160000000p-6, 0x1.0a48720000000p-4, 0x1.a7f6260000000p-6, -0x1.7ae9480000000p-4, 0x1.19bc460000000p-4, 0x1.bc36ae0000000p-6, 0x1.248f2e0000000p-8, 0x1.b92f600000000p-3, -0x1.a106820000000p-5, 0x1.af90940000000p-6, 0x1.474fe80000000p-8, 0x1.cd6b760000000p-4, 0x1.0caefa0000000p-3, 0x1.2a407c0000000p-3, 0x1.b0dcc60000000p-3, 0x1.648a580000000p-5, -0x1.b5a7600000000p-4, 0x1.47c7e20000000p-4, -0x1.4a72fe0000000p-4, 0x1.59959c0000000p-5, 0x1.8fa4560000000p-4, 0x1.47b41c0000000p-3, 0x1.cba2d40000000p-4, 0x1.d4e7fe0000000p-4, 0x1.2460fe0000000p-5, 0x1.da4c8e0000000p-4, 0x1.f194700000000p-6, -0x1.cba4100000000p-5, -0x1.c029b00000000p-4, -0x1.51dec60000000p-3, -0x1.2bfcc80000000p-5, 0x1.1c22ee0000000p-2, -0x1.38bc7c0000000p-4}
, {0x1.0b674e0000000p-3, 0x1.6c30ec0000000p-6, 0x1.a497b20000000p-6, 0x1.81f1c80000000p-5, 0x1.b0817a0000000p-8, -0x1.16cb960000000p-3, 0x1.6ad41c0000000p-5, 0x1.a9ac780000000p-6, -0x1.e4cf840000000p-3, 0x1.cdff760000000p-4, 0x1.9c4cd20000000p-5, 0x1.b6d58a0000000p-5, -0x1.2b86d00000000p-3, 0x1.91823c0000000p-4, -0x1.86a72e0000000p-4, 0x1.ef8a6c0000000p-8, 0x1.d140280000000p-4, -0x1.bac4e20000000p-4, -0x1.30acbc0000000p-5, -0x1.ed3c6e0000000p-6, 0x1.a3bf1c0000000p-6, -0x1.92e8be0000000p-5, 0x1.bdba960000000p-5, 0x1.8aa7a80000000p-7, 0x1.5a47c00000000p-4, 0x1.465cca0000000p-4, -0x1.f0d5e20000000p-12, 0x1.0c3b4c0000000p-3, 0x1.1446340000000p-5, 0x1.4a940a0000000p-6, 0x1.8866f20000000p-3, 0x1.2208740000000p-3, 0x1.10bf7a0000000p-3, 0x1.153a740000000p-5, 0x1.146c280000000p-4, -0x1.989fe60000000p-3, 0x1.73c9ee0000000p-4, 0x1.dcce6a0000000p-7, 0x1.0fe3580000000p-5, 0x1.7fd4d40000000p-6, 0x1.faadb20000000p-4, 0x1.d942240000000p-4, 0x1.d184cc0000000p-6, 0x1.a368c00000000p-5, 0x1.daf05a0000000p-5, 0x1.93ea660000000p-9, -0x1.69b8860000000p-5, -0x1.107a320000000p-4, 0x1.52627e0000000p-5, 0x1.7607800000000p-3, 0x1.0cbfdc0000000p-3, -0x1.712fee0000000p-4, 0x1.baca1a0000000p-5, 0x1.1a098e0000000p-4, 0x1.6e9ba20000000p-6, -0x1.39253e0000000p-3, -0x1.e2d2140000000p-5, -0x1.01da780000000p-2, -0x1.0aab9a0000000p-7, -0x1.c1051a0000000p-6, -0x1.b18b780000000p-9, 0x1.73314a0000000p-4, 0x1.7be3840000000p-3, -0x1.6224560000000p-6}
, {0x1.c7052a0000000p-6, 0x1.d650d80000000p-4, 0x1.bd8a6c0000000p-5, -0x1.9898240000000p-4, 0x1.301e760000000p-4, 0x1.6e18f80000000p-3, -0x1.14bc900000000p-3, 0x1.eb564a0000000p-5, 0x1.11dc1a0000000p-5, 0x1.b214ac0000000p-3, 0x1.aa0cb20000000p-3, 0x1.430f6a0000000p-3, 0x1.e1a95e0000000p-4, -0x1.663fb40000000p-4, 0x1.0a242a0000000p-4, 0x1.5014d60000000p-6, 0x1.59c5760000000p-3, 0x1.89aacc0000000p-3, -0x1.05c5a00000000p-5, -0x1.21bb940000000p-2, 0x1.41b98c0000000p-5, -0x1.d045520000000p-9, 0x1.0667f40000000p-3, 0x1.f8fd9c0000000p-6, -0x1.0bec840000000p-4, 0x1.3389140000000p-3, 0x1.5a0a0e0000000p-4, 0x1.15e7f20000000p-5, 0x1.231ba60000000p-2, 0x1.3fb6aa0000000p-5, -0x1.c295480000000p-5, -0x1.84d2ec0000000p-4, -0x1.37f83a0000000p-7, 0x1.84ec320000000p-7, -0x1.5befec0000000p-4, -0x1.1b08760000000p-8, 0x1.4320fa0000000p-4, -0x1.e9e5640000000p-4, 0x1.423b6c0000000p-2, 0x1.4c23f60000000p-6, 0x1.491cac0000000p-6, -0x1.ad20a60000000p-4, 0x1.5a9a6e0000000p-3, -0x1.4518840000000p-5, 0x1.bf827e0000000p-4, 0x1.3cede40000000p-5, 0x1.8fd57a0000000p-4, -0x1.0d89920000000p-3, 0x1.c1bbac0000000p-4, -0x1.604ba20000000p-4, 0x1.29639a0000000p-4, 0x1.b1167e0000000p-4, 0x1.3989e20000000p-4, 0x1.9925c60000000p-4, 0x1.45089c0000000p-3, -0x1.36eaaa0000000p-3, 0x1.542b080000000p-2, -0x1.7be06e0000000p-4, 0x1.5375880000000p-4, -0x1.76a6f00000000p-4, -0x1.8613da0000000p-6, -0x1.cf55840000000p-7, 0x1.6103cc0000000p-3, 0x1.20fc4c0000000p-2}
}
, {{-0x1.6fb3de0000000p-9, 0x1.1a7aec0000000p-2, 0x1.a927c40000000p-4, -0x1.3bf3380000000p-3, 0x1.012f0e0000000p-4, 0x1.7416080000000p-3, 0x1.256db80000000p-4, 0x1.0f7b380000000p-3, -0x1.7d77fc0000000p-3, -0x1.2f150e0000000p-5, 0x1.3748720000000p-3, 0x1.3eabfc0000000p-4, -0x1.1ecb080000000p-4, -0x1.c0606a0000000p-4, -0x1.0b25de0000000p-2, 0x1.1b6a300000000p-4, 0x1.4f31de0000000p-3, -0x1.8894ca0000000p-3, 0x1.7363680000000p-4, 0x1.b989840000000p-3, 0x1.6461540000000p-5, 0x1.a9061e0000000p-3, -0x1.8440620000000p-3, 0x1.2596680000000p-3, -0x1.2ae2760000000p-2, 0x1.9bd6120000000p-4, -0x1.4d725c0000000p-4, 0x1.560e4a0000000p-3, 0x1.2a09040000000p-4, 0x1.c19f680000000p-5, 0x1.3fca6c0000000p-10, 0x1.14b42e0000000p-3, -0x1.b867de0000000p-3, -0x1.33f2c20000000p-4, 0x1.fc729e0000000p-4, 0x1.13e80c0000000p-6, -0x1.d142ce0000000p-4, -0x1.af96380000000p-4, -0x1.efe2660000000p-6, -0x1.c7024a0000000p-5, 0x1.0d6d8c0000000p-2, 0x1.67d6bc0000000p-6, -0x1.f81ef00000000p-4, -0x1.9a94140000000p-4, -0x1.7c874a0000000p-5, -0x1.ad0d600000000p-4, 0x1.dcdd180000000p-3, -0x1.36c6460000000p-3, -0x1.045f9c0000000p-3, -0x1.61b1080000000p-6, 0x1.76157c0000000p-5, 0x1.8cbb680000000p-5, 0x1.c2ce660000000p-3, 0x1.d718fa0000000p-3, -0x1.254ab80000000p-6, 0x1.d2248a0000000p-4, 0x1.ce47e60000000p-5, 0x1.43a1b40000000p-3, 0x1.2cc5340000000p-2, -0x1.6af41a0000000p-4, -0x1.30c2180000000p-5, 0x1.7c3e660000000p-3, 0x1.9277d40000000p-5, -0x1.718b8e0000000p-3}
, {-0x1.b24dfa0000000p-4, -0x1.ac7f1e0000000p-4, 0x1.b8d6700000000p-3, -0x1.9dbc100000000p-4, -0x1.6b4d560000000p-8, 0x1.e59be60000000p-6, -0x1.5e1a480000000p-5, 0x1.17aa320000000p-3, 0x1.652d980000000p-4, -0x1.0a65e60000000p-5, 0x1.10f55c0000000p-4, 0x1.023e100000000p-4, 0x1.77086e0000000p-3, -0x1.e0d8620000000p-4, 0x1.a941700000000p-4, 0x1.659f9c0000000p-3, 0x1.63832a0000000p-5, 0x1.23ea600000000p-6, 0x1.4075140000000p-5, 0x1.606ae20000000p-6, 0x1.4740420000000p-3, 0x1.41a7140000000p-3, 0x1.bb6f080000000p-5, -0x1.07ca060000000p-10, 0x1.043c340000000p-6, 0x1.bc9cb40000000p-6, 0x1.0f07ec0000000p-4, 0x1.e2d6a60000000p-6, 0x1.38fc0a0000000p-4, 0x1.ee87180000000p-4, 0x1.8ebd9a0000000p-6, -0x1.ee4d840000000p-4, -0x1.2373fc0000000p-3, 0x1.0aa0e20000000p-4, 0x1.1cc04e0000000p-4, -0x1.16f4a40000000p-8, -0x1.cd8e120000000p-3, -0x1.5a278c0000000p-5, 0x1.d8a8200000000p-8, 0x1.e444f80000000p-7, 0x1.8b54440000000p-6, -0x1.5818ee0000000p-5, -0x1.7533f40000000p-4, -0x1.91c42c0000000p-3, 0x1.457bde0000000p-3, 0x1.3a34060000000p-5, 0x1.bc381e0000000p-4, -0x1.df58100000000p-5, -0x1.7590900000000p-6, 0x1.dc3bca0000000p-4, 0x1.f244100000000p-5, -0x1.2e64b80000000p-6, -0x1.d39e360000000p-5, -0x1.12292c0000000p-3, -0x1.3dccc00000000p-6, -0x1.66135c0000000p-7, 0x1.4c494e0000000p-8, 0x1.d52cf00000000p-3, -0x1.e392fa0000000p-5, -0x1.94fc1e0000000p-5, -0x1.4bcf2e0000000p-4, -0x1.79057c0000000p-4, -0x1.e353c60000000p-9, -0x1.9bd7e40000000p-5}
, {-0x1.3c20340000000p-3, -0x1.12dc7c0000000p-5, -0x1.1de7a00000000p-6, -0x1.abc2100000000p-4, -0x1.f094840000000p-4, -0x1.74b62e0000000p-6, -0x1.126a880000000p-9, -0x1.9eed300000000p-4, -0x1.ad4ea20000000p-3, -0x1.898ea80000000p-3, -0x1.86e68a0000000p-6, 0x1.17b6d80000000p-4, -0x1.31728c0000000p-3, -0x1.d31aac0000000p-5, 0x1.af2ca00000000p-4, -0x1.c682ca0000000p-4, 0x1.8d723a0000000p-3, 0x1.8b74840000000p-7, -0x1.4802da0000000p-2, -0x1.61c6040000000p-4, -0x1.7367880000000p-3, -0x1.7a14ba0000000p-3, -0x1.c3d8620000000p-3, -0x1.55ecbe0000000p-3, -0x1.db9aec0000000p-4, -0x1.464ebe0000000p-5, 0x1.0e701c0000000p-4, -0x1.2a6a3c0000000p-3, -0x1.de79680000000p-7, -0x1.f37d620000000p-8, 0x1.3221420000000p-4, 0x1.fa0e0c0000000p-3, 0x1.8205b00000000p-2, -0x1.0498f20000000p-3, -0x1.ed36fa0000000p-4, 0x1.0f31020000000p-3, 0x1.d2c0860000000p-8, -0x1.e552500000000p-4, -0x1.1c33460000000p-2, -0x1.874fce0000000p-5, 0x1.df25600000000p-7, 0x1.181dea0000000p-5, -0x1.7024320000000p-3, -0x1.6fac6c0000000p-5, -0x1.468ee40000000p-3, 0x1.ecbef80000000p-5, -0x1.f63dc80000000p-5, -0x1.b313120000000p-9, -0x1.38437e0000000p-4, 0x1.1dc4300000000p-2, -0x1.b439800000000p-4, -0x1.309b440000000p-2, -0x1.c066980000000p-5, 0x1.983dde0000000p-4, -0x1.1742760000000p-2, 0x1.3a2adc0000000p-2, -0x1.d9fa360000000p-7, 0x1.f7862c0000000p-6, -0x1.2e85e40000000p-3, -0x1.9cbd7a0000000p-6, 0x1.2912160000000p-2, 0x1.93b3e60000000p-4, 0x1.0edec20000000p-3, 0x1.d17cf40000000p-10}
}
, {{0x1.53e33a0000000p-2, 0x1.522cc40000000p-3, -0x1.390ce40000000p-3, -0x1.4d901a0000000p-4, 0x1.a065b40000000p-5, 0x1.8825620000000p-4, -0x1.88bc460000000p-3, 0x1.be0bb00000000p-9, -0x1.2f60c40000000p-4, -0x1.1cdb160000000p-3, 0x1.a4070a0000000p-4, 0x1.530e640000000p-4, -0x1.630e8e0000000p-3, -0x1.bff5f60000000p-6, -0x1.867c260000000p-4, -0x1.6af6500000000p-4, -0x1.054e460000000p-3, 0x1.ae962a0000000p-5, 0x1.bb9a040000000p-4, 0x1.8f1a840000000p-5, 0x1.25c21c0000000p-5, -0x1.c948e40000000p-4, -0x1.f195140000000p-6, 0x1.e40f8c0000000p-5, -0x1.e89bac0000000p-6, -0x1.2a61ea0000000p-4, -0x1.0060ec0000000p-6, 0x1.0626a60000000p-4, -0x1.e4a0600000000p-4, -0x1.463d160000000p-3, 0x1.1560320000000p-3, 0x1.02b8980000000p-3, 0x1.e9b53a0000000p-5, -0x1.af89220000000p-5, 0x1.13751e0000000p-6, -0x1.d39ef00000000p-4, 0x1.009cee0000000p-5, -0x1.4075a40000000p-6, 0x1.469f360000000p-6, 0x1.8ded9e0000000p-5, -0x1.5563ea0000000p-4, -0x1.c2ac9e0000000p-5, -0x1.ad10d60000000p-6, -0x1.29ee180000000p-6, 0x1.488a300000000p-3, -0x1.35967a0000000p-2, -0x1.c7b3f20000000p-3, -0x1.374ac00000000p-3, -0x1.9b553e0000000p-5, 0x1.fd9e4a0000000p-3, -0x1.0bdcb20000000p-4, 0x1.80e8d60000000p-5, -0x1.2f47820000000p-4, -0x1.bd8f460000000p-8, 0x1.28f9a20000000p-5, -0x1.f19da60000000p-4, -0x1.9e21de0000000p-4, 0x1.01b7660000000p-4, 0x1.de4eaa0000000p-7, -0x1.e997d80000000p-4, -0x1.48f2da0000000p-4, -0x1.abdf260000000p-5, -0x1.bc5f5e0000000p-5, -0x1.096e980000000p-4}
, {-0x1.1a420c0000000p-8, 0x1.c1a3260000000p-7, 0x1.3716b20000000p-3, 0x1.5de6900000000p-6, 0x1.007b160000000p-4, -0x1.3e36380000000p-5, -0x1.1474bc0000000p-3, -0x1.302e9c0000000p-5, 0x1.1b403a0000000p-3, 0x1.2571c60000000p-4, 0x1.a179800000000p-5, 0x1.6cc4f80000000p-7, -0x1.2914100000000p-6, -0x1.59aed60000000p-3, 0x1.f571320000000p-4, 0x1.35e17a0000000p-4, 0x1.a427840000000p-9, 0x1.6b62180000000p-3, 0x1.7e7db60000000p-5, -0x1.d5f7040000000p-6, 0x1.a0e6d60000000p-6, -0x1.a4c5e60000000p-6, -0x1.9856b40000000p-8, 0x1.82ac180000000p-5, -0x1.e8db680000000p-5, -0x1.4533900000000p-3, 0x1.90f4060000000p-10, -0x1.02864a0000000p-4, 0x1.6be2a60000000p-6, 0x1.5267fa0000000p-6, 0x1.39721c0000000p-5, -0x1.97f6f40000000p-3, -0x1.9437c20000000p-9, 0x1.55b1960000000p-6, 0x1.69e8700000000p-4, 0x1.fb0ef80000000p-4, 0x1.896c360000000p-9, 0x1.bf6fe60000000p-4, -0x1.35fae00000000p-3, 0x1.d73be20000000p-7, -0x1.09aa940000000p-3, -0x1.aad79e0000000p-5, 0x1.52ab2e0000000p-5, -0x1.0789ee0000000p-5, 0x1.ffb1260000000p-4, 0x1.0507600000000p-4, -0x1.37c7e80000000p-3, 0x1.15d1940000000p-3, -0x1.9373d20000000p-9, -0x1.2f0ef00000000p-3, -0x1.2fdfa60000000p-4, 0x1.0ede8c0000000p-5, 0x1.d1b1be0000000p-5, 0x1.0f02080000000p-4, 0x1.c1ad200000000p-7, -0x1.43127c0000000p-4, 0x1.cbe39c0000000p-7, 0x1.973dfe0000000p-4, -0x1.86b5100000000p-4, -0x1.12ddcc0000000p-3, 0x1.2bc30c0000000p-4, 0x1.6b43720000000p-8, -0x1.a4485e0000000p-6, -0x1.28339c0000000p-3}
, {0x1.cd0bc00000000p-5, -0x1.11a08a0000000p-3, -0x1.8035ae0000000p-6, 0x1.256a900000000p-4, -0x1.44e6880000000p-7, 0x1.30925c0000000p-7, 0x1.62f87e0000000p-4, -0x1.c629620000000p-4, -0x1.bddf0a0000000p-4, -0x1.9e73cc0000000p-4, -0x1.70da760000000p-4, -0x1.df5df60000000p-10, -0x1.22cec60000000p-3, -0x1.6de9d00000000p-4, 0x1.5df2c20000000p-8, 0x1.1527b20000000p-3, 0x1.3cfd3c0000000p-4, 0x1.7252e20000000p-5, -0x1.eb41820000000p-5, 0x1.33098c0000000p-7, -0x1.b65d9a0000000p-3, -0x1.197d6c0000000p-5, -0x1.24a2720000000p-4, -0x1.8f755a0000000p-6, -0x1.c7ef7c0000000p-4, -0x1.cf62ce0000000p-3, 0x1.9c43aa0000000p-6, 0x1.58865a0000000p-6, -0x1.4f55900000000p-3, 0x1.cf96ea0000000p-5, 0x1.bcff9a0000000p-6, 0x1.e6734a0000000p-4, 0x1.35fc4a0000000p-4, 0x1.ead60a0000000p-9, -0x1.52888a0000000p-4, 0x1.fac6ca0000000p-6, 0x1.03217a0000000p-2, -0x1.9f80180000000p-4, -0x1.bf36e80000000p-7, 0x1.db9dd00000000p-5, -0x1.4b28f20000000p-6, -0x1.3551d60000000p-6, -0x1.4fc1c00000000p-4, -0x1.07640a0000000p-4, -0x1.109e580000000p-3, 0x1.7a75380000000p-5, -0x1.8834b80000000p-3, -0x1.acaf980000000p-7, -0x1.582a060000000p-5, 0x1.31cfa60000000p-3, 0x1.5a401c0000000p-4, -0x1.092cde0000000p-3, 0x1.b91c820000000p-5, 0x1.db70940000000p-4, -0x1.3dec400000000p-4, 0x1.b7883c0000000p-4, -0x1.778a060000000p-3, -0x1.74f7200000000p-3, 0x1.38f9900000000p-3, 0x1.a257840000000p-6, 0x1.0cf7760000000p-3, -0x1.4cab700000000p-9, 0x1.06ec8e0000000p-3, -0x1.7c639a0000000p-3}
}
, {{-0x1.52b2160000000p-3, -0x1.6831b40000000p-3, -0x1.89af4a0000000p-7, 0x1.0a42f80000000p-3, 0x1.1409460000000p-3, 0x1.1c820a0000000p-6, 0x1.bc6df80000000p-10, 0x1.27b92a0000000p-8, 0x1.cbdc500000000p-5, 0x1.f1fd440000000p-8, 0x1.4668e60000000p-4, -0x1.e2f15a0000000p-5, -0x1.e9257e0000000p-5, 0x1.c899b60000000p-4, -0x1.5bdeb20000000p-3, -0x1.3c0bea0000000p-4, -0x1.45f21a0000000p-4, -0x1.5254360000000p-3, 0x1.19cca80000000p-5, -0x1.cc1a4a0000000p-3, 0x1.1b19580000000p-3, 0x1.f2be060000000p-6, 0x1.33b6a00000000p-7, -0x1.0ce0a20000000p-3, -0x1.fbb5400000000p-4, -0x1.07666c0000000p-6, 0x1.897cf60000000p-7, 0x1.106fe80000000p-3, 0x1.20e11e0000000p-5, 0x1.14d0f60000000p-5, -0x1.8ac91c0000000p-4, -0x1.3eee260000000p-3, 0x1.7477960000000p-4, 0x1.383efe0000000p-6, 0x1.25d05e0000000p-8, -0x1.975c860000000p-7, -0x1.5104fc0000000p-5, 0x1.cf5ef00000000p-7, -0x1.7da2280000000p-8, -0x1.723a780000000p-9, -0x1.f9ba180000000p-4, -0x1.c89f420000000p-6, 0x1.22b4c00000000p-4, -0x1.ef6e5e0000000p-5, 0x1.25ab280000000p-3, 0x1.3bcf500000000p-3, 0x1.ebc54c0000000p-4, 0x1.f522800000000p-6, 0x1.bb7cc80000000p-4, -0x1.f65dc60000000p-3, -0x1.3e091a0000000p-3, -0x1.b46c8c0000000p-4, -0x1.7b63e40000000p-5, 0x1.806a0a0000000p-3, 0x1.285afa0000000p-4, -0x1.b5dfc40000000p-4, 0x1.bbe6460000000p-4, -0x1.70a4280000000p-7, 0x1.1323980000000p-6, -0x1.b741cc0000000p-4, -0x1.1e92a20000000p-3, -0x1.4594c80000000p-4, -0x1.16af920000000p-4, 0x1.5e4a900000000p-4}
, {-0x1.3ec6dc0000000p-4, 0x1.2874680000000p-6, -0x1.8015c60000000p-4, -0x1.60ac760000000p-5, 0x1.2975720000000p-5, 0x1.c67c240000000p-4, -0x1.2456de0000000p-3, -0x1.c01f260000000p-4, 0x1.02745a0000000p-3, 0x1.96825a0000000p-5, 0x1.00575e0000000p-3, 0x1.83b0920000000p-3, 0x1.6518580000000p-3, 0x1.30f1720000000p-4, -0x1.73de3e0000000p-6, 0x1.2b99000000000p-5, 0x1.9240d00000000p-3, -0x1.d2d9c80000000p-6, 0x1.f4ad100000000p-4, -0x1.15321c0000000p-3, -0x1.f9c2d20000000p-5, 0x1.6c7a7e0000000p-4, -0x1.1e565a0000000p-3, 0x1.7a2ccc0000000p-3, 0x1.14c1920000000p-5, 0x1.6925e20000000p-4, 0x1.a9818e0000000p-4, 0x1.cfa2f60000000p-4, -0x1.69895e0000000p-8, 0x1.11bf020000000p-3, -0x1.65dac20000000p-6, 0x1.7198ac0000000p-3, 0x1.0a5b8a0000000p-4, 0x1.1c4e1a0000000p-4, 0x1.b0ba920000000p-3, 0x1.1688220000000p-4, -0x1.de0b120000000p-7, 0x1.8a9a420000000p-5, -0x1.ce27200000000p-10, -0x1.1c2d820000000p-5, -0x1.50ea200000000p-5, 0x1.0ec17a0000000p-3, -0x1.04f3f00000000p-3, -0x1.233f700000000p-3, -0x1.6a11540000000p-4, 0x1.62feea0000000p-6, 0x1.2005c20000000p-6, 0x1.e5e4b60000000p-4, 0x1.a3b6f20000000p-9, -0x1.d953e20000000p-5, 0x1.2d6e840000000p-4, -0x1.4b87a80000000p-5, -0x1.145a380000000p-6, 0x1.4d27020000000p-4, 0x1.060a9a0000000p-4, 0x1.b82a4e0000000p-3, 0x1.b33c220000000p-4, 0x1.73479c0000000p-5, 0x1.75edd80000000p-3, 0x1.32d8880000000p-7, 0x1.96d7ec0000000p-3, 0x1.8e59980000000p-4, 0x1.7e22ac0000000p-4, 0x1.8312180000000p-5}
, {-0x1.31b1a20000000p-5, 0x1.61d46c0000000p-3, 0x1.fb91100000000p-5, 0x1.401b620000000p-7, -0x1.814de20000000p-5, 0x1.c8113a0000000p-3, -0x1.b1ed840000000p-6, 0x1.192f4c0000000p-6, 0x1.1cc13e0000000p-3, 0x1.27dbca0000000p-3, 0x1.8ca44e0000000p-3, -0x1.18f3ae0000000p-5, 0x1.a4d8260000000p-6, -0x1.f393480000000p-6, -0x1.dc2c580000000p-4, -0x1.a541a60000000p-5, 0x1.2e01940000000p-5, -0x1.9761680000000p-5, -0x1.d543aa0000000p-6, -0x1.e8a01e0000000p-4, 0x1.fe24be0000000p-11, -0x1.2f6ada0000000p-3, -0x1.033b9a0000000p-3, 0x1.71395a0000000p-4, -0x1.5a2eae0000000p-5, -0x1.1c7ec20000000p-5, 0x1.7292fc0000000p-4, 0x1.edcb3c0000000p-4, 0x1.04c2e60000000p-3, 0x1.49bc7c0000000p-5, -0x1.bd76b40000000p-4, -0x1.e9d3ce0000000p-8, 0x1.3f0c1e0000000p-5, 0x1.495a960000000p-4, -0x1.e35a440000000p-8, 0x1.49fede0000000p-4, 0x1.3bdbc00000000p-5, -0x1.5299420000000p-4, 0x1.f40b880000000p-3, -0x1.62e3cc0000000p-3, -0x1.33cbd80000000p-4, 0x1.697df00000000p-4, 0x1.b9f01a0000000p-6, 0x1.77f10c0000000p-4, -0x1.36d7de0000000p-5, -0x1.42c6a60000000p-14, 0x1.f512c00000000p-4, 0x1.0865180000000p-3, -0x1.99e6620000000p-5, -0x1.a2e4460000000p-4, 0x1.9ebaa60000000p-5, 0x1.5c3ee80000000p-4, 0x1.af1e980000000p-4, 0x1.a6d1d40000000p-3, 0x1.a0c77e0000000p-3, 0x1.0f073e0000000p-3, 0x1.15c9b80000000p-3, 0x1.1737fe0000000p-6, 0x1.151d3e0000000p-4, 0x1.11b7720000000p-7, -0x1.a5724a0000000p-7, -0x1.2612b20000000p-4, 0x1.6854c00000000p-4, 0x1.06be840000000p-3}
}
, {{-0x1.f8b5180000000p-4, -0x1.9e7d780000000p-5, 0x1.ba5dc20000000p-3, -0x1.908f640000000p-3, -0x1.5d45a40000000p-4, 0x1.df4dc20000000p-3, -0x1.47eeac0000000p-4, 0x1.d64aec0000000p-3, 0x1.58b7fe0000000p-5, -0x1.194e880000000p-3, 0x1.52829e0000000p-3, 0x1.54dc180000000p-2, -0x1.ccae1c0000000p-5, -0x1.5d3fea0000000p-3, -0x1.24d1a00000000p-4, -0x1.4fcfc40000000p-3, 0x1.86ff680000000p-3, -0x1.78eeb60000000p-5, -0x1.19837c0000000p-3, -0x1.9527520000000p-4, 0x1.fe7f360000000p-6, 0x1.de9e680000000p-3, -0x1.24a0300000000p-2, -0x1.9c6b700000000p-4, -0x1.dab20c0000000p-3, -0x1.a4d0100000000p-4, 0x1.02efbe0000000p-2, -0x1.03b5000000000p-4, -0x1.f8bcc20000000p-5, 0x1.04643e0000000p-3, -0x1.4a212e0000000p-5, 0x1.e0ea7a0000000p-4, -0x1.2364360000000p-4, 0x1.58b1100000000p-3, 0x1.1a26940000000p-3, 0x1.c2cef00000000p-3, -0x1.37ec000000000p-3, -0x1.57f58c0000000p-3, -0x1.84f9a80000000p-3, 0x1.cd68720000000p-4, -0x1.1dd4ee0000000p-5, 0x1.8e31980000000p-5, -0x1.6b39b60000000p-4, -0x1.46ecb20000000p-3, -0x1.26f4ca0000000p-3, 0x1.0685440000000p-2, 0x1.488e880000000p-6, 0x1.bf3f000000000p-3, 0x1.01b3ac0000000p-2, 0x1.2f86a60000000p-3, 0x1.8cd9840000000p-4, 0x1.f4de3e0000000p-3, 0x1.74cb380000000p-2, 0x1.4b4b620000000p-4, -0x1.e2af5a0000000p-3, 0x1.47e4680000000p-3, 0x1.c276f80000000p-4, 0x1.10c3160000000p-4, -0x1.e3819c0000000p-4, -0x1.449e7c0000000p-3, 0x1.4080400000000p-7, 0x1.34457c0000000p-2, 0x1.d48f8c0000000p-5, -0x1.835b660000000p-3}
, {-0x1.4efdb40000000p-4, -0x1.bb3ad00000000p-4, -0x1.010db80000000p-6, -0x1.52acaa0000000p-5, 0x1.47b17c0000000p-4, 0x1.9486c80000000p-9, 0x1.a0dee20000000p-4, -0x1.476e180000000p-4, -0x1.37f0ea0000000p-5, -0x1.23b7300000000p-3, 0x1.e03a380000000p-6, 0x1.322d3a0000000p-2, 0x1.8f0a020000000p-4, 0x1.27eea20000000p-3, -0x1.3fabb80000000p-3, -0x1.bf80160000000p-3, -0x1.29fc820000000p-3, 0x1.000dc20000000p-3, -0x1.91bff00000000p-3, -0x1.0cedd80000000p-7, 0x1.ce1c100000000p-6, -0x1.35729a0000000p-4, -0x1.6598880000000p-4, -0x1.208fe80000000p-4, -0x1.6c69e60000000p-3, 0x1.6cd7be0000000p-4, 0x1.5a7ba60000000p-5, -0x1.a3fb760000000p-3, -0x1.21897e0000000p-3, 0x1.ed15a00000000p-3, -0x1.731cae0000000p-4, 0x1.d368640000000p-5, 0x1.1fbc800000000p-3, -0x1.38a9940000000p-3, -0x1.86b2940000000p-5, -0x1.e003240000000p-4, 0x1.c1e18a0000000p-4, -0x1.acf46c0000000p-3, 0x1.3d32aa0000000p-9, -0x1.e452600000000p-6, 0x1.bcad0e0000000p-4, -0x1.ff35ba0000000p-4, -0x1.11c14c0000000p-5, -0x1.3394380000000p-3, 0x1.9810b80000000p-8, 0x1.dfd9de0000000p-6, -0x1.c9fb1e0000000p-4, -0x1.6fb76e0000000p-4, -0x1.ba81060000000p-4, 0x1.94c5720000000p-5, 0x1.36873c0000000p-4, 0x1.dd2a9e0000000p-4, -0x1.1325ba0000000p-4, 0x1.3dcce40000000p-4, 0x1.71615e0000000p-5, 0x1.4c79240000000p-3, 0x1.6a42a80000000p-3, 0x1.8a27680000000p-4, -0x1.1bef860000000p-3, 0x1.5583ec0000000p-4, 0x1.6f1e940000000p-4, 0x1.3eca8e0000000p-3, -0x1.0c66480000000p-2, 0x1.38f84e0000000p-6}
, {0x1.1dc9600000000p-2, -0x1.7135580000000p-4, -0x1.c576fc0000000p-4, 0x1.1de8b80000000p-2, 0x1.1493620000000p-5, -0x1.1ad8ea0000000p-3, -0x1.3d00200000000p-5, 0x1.2329ac0000000p-3, -0x1.7fbf7e0000000p-10, 0x1.0b43160000000p-2, -0x1.19346a0000000p-7, 0x1.35a5600000000p-5, -0x1.1ed15c0000000p-3, 0x1.a918660000000p-4, -0x1.917b4a0000000p-6, -0x1.f435260000000p-3, -0x1.174bd60000000p-3, -0x1.8f0c8c0000000p-5, -0x1.28f2d60000000p-4, 0x1.5fcdd00000000p-12, -0x1.5019cc0000000p-4, -0x1.8a78fa0000000p-3, 0x1.aaaab60000000p-4, 0x1.0ff7220000000p-3, -0x1.0cf7760000000p-3, 0x1.4ba2da0000000p-6, 0x1.2812c40000000p-4, 0x1.8db4300000000p-4, -0x1.8b04d60000000p-2, -0x1.1a54200000000p-2, 0x1.1b3f980000000p-2, 0x1.97b95e0000000p-4, 0x1.00135e0000000p-6, -0x1.7099540000000p-6, -0x1.5cf9480000000p-2, -0x1.5bb3820000000p-2, 0x1.39223c0000000p-3, -0x1.f953e60000000p-3, 0x1.7c84d40000000p-4, -0x1.0504bc0000000p-3, 0x1.a72f880000000p-4, 0x1.50e4560000000p-3, 0x1.9f0c620000000p-4, -0x1.324ca60000000p-2, 0x1.deb6ee0000000p-5, -0x1.983efe0000000p-6, -0x1.0b32a40000000p-4, 0x1.e802720000000p-7, -0x1.3e8b880000000p-2, 0x1.08cdc00000000p-7, 0x1.07ffe20000000p-3, -0x1.078c680000000p-5, -0x1.921f8e0000000p-3, -0x1.3702680000000p-5, -0x1.b416ec0000000p-3, 0x1.31028c0000000p-4, -0x1.0f8dba0000000p-2, -0x1.7c6a300000000p-5, -0x1.100cc80000000p-3, 0x1.ded75e0000000p-3, -0x1.a9ff240000000p-6, 0x1.d61da60000000p-3, 0x1.065ac60000000p-3, 0x1.cd64f00000000p-6}
}
, {{-0x1.bd4c600000000p-7, 0x1.26ea740000000p-3, -0x1.5e7d240000000p-4, -0x1.21b8160000000p-8, -0x1.ba221a0000000p-5, -0x1.f70c3e0000000p-4, 0x1.b9cb100000000p-5, -0x1.a50c4c0000000p-8, -0x1.6532b00000000p-2, -0x1.c8d3a80000000p-7, -0x1.e3372a0000000p-4, 0x1.84a37c0000000p-3, -0x1.ca81a00000000p-7, -0x1.f8116a0000000p-4, -0x1.8307a00000000p-3, 0x1.27c05c0000000p-4, 0x1.9db60c0000000p-6, 0x1.8157760000000p-4, -0x1.4c70460000000p-3, 0x1.f1a95c0000000p-4, -0x1.680ee80000000p-5, -0x1.3420ce0000000p-5, 0x1.7a4ae20000000p-5, -0x1.0c977a0000000p-5, 0x1.797c480000000p-6, 0x1.4da96e0000000p-5, -0x1.8c0aa00000000p-6, 0x1.51344e0000000p-10, 0x1.dfea140000000p-4, 0x1.f378400000000p-6, 0x1.20723e0000000p-2, 0x1.1c02c80000000p-3, -0x1.bf82d80000000p-13, -0x1.f439480000000p-6, 0x1.d1989e0000000p-5, 0x1.20c1da0000000p-8, 0x1.458c2c0000000p-6, -0x1.a8519e0000000p-5, -0x1.f7b0e80000000p-4, -0x1.8458f20000000p-8, -0x1.6f91a40000000p-7, -0x1.a058ea0000000p-4, -0x1.ed2ea40000000p-3, -0x1.fd5f2e0000000p-3, -0x1.d089e20000000p-5, -0x1.8d5bde0000000p-3, -0x1.034e8c0000000p-3, -0x1.0280f00000000p-2, -0x1.efa3de0000000p-4, -0x1.5460080000000p-4, -0x1.3790b60000000p-9, 0x1.f9d2140000000p-7, 0x1.aac7100000000p-6, -0x1.72b76c0000000p-5, -0x1.f5406c0000000p-3, 0x1.17628e0000000p-7, 0x1.c30be40000000p-5, 0x1.cb58a00000000p-4, 0x1.0ce3ea0000000p-4, 0x1.6f19660000000p-5, 0x1.fe779a0000000p-3, 0x1.b2e7fa0000000p-4, -0x1.d19c860000000p-5, -0x1.4f4cc80000000p-3}
, {0x1.d9232a0000000p-5, -0x1.c5d2fa0000000p-7, -0x1.a87d600000000p-9, -0x1.36ac760000000p-5, -0x1.ad3f420000000p-5, 0x1.3905cc0000000p-3, -0x1.55895e0000000p-5, -0x1.39d2620000000p-3, 0x1.cb777a0000000p-7, 0x1.3ac4920000000p-4, -0x1.27d3e60000000p-3, -0x1.2e46120000000p-6, 0x1.73a8b80000000p-5, 0x1.5377f20000000p-4, -0x1.b623ba0000000p-6, -0x1.de7f540000000p-5, -0x1.634f040000000p-5, 0x1.2518aa0000000p-6, 0x1.4344700000000p-3, -0x1.0e10a20000000p-3, 0x1.0ff2cc0000000p-3, -0x1.755f140000000p-7, 0x1.7aa7600000000p-4, 0x1.b1054c0000000p-10, 0x1.ed517e0000000p-6, -0x1.e8335c0000000p-7, -0x1.be63dc0000000p-4, -0x1.c0603a0000000p-7, 0x1.7da4980000000p-7, -0x1.8e36800000000p-8, -0x1.2beb900000000p-3, -0x1.16f3660000000p-4, -0x1.94990e0000000p-3, -0x1.68f03a0000000p-5, 0x1.41cdae0000000p-4, -0x1.85f3140000000p-6, -0x1.4a5d5a0000000p-5, 0x1.bacc560000000p-4, 0x1.b6cbb00000000p-6, -0x1.5d5dda0000000p-5, -0x1.3059760000000p-3, -0x1.01e76a0000000p-4, -0x1.6ef8c60000000p-6, -0x1.21062c0000000p-3, -0x1.7727580000000p-5, -0x1.36c2be0000000p-3, -0x1.6698920000000p-5, -0x1.d662860000000p-7, 0x1.0b29220000000p-4, 0x1.686aa60000000p-7, -0x1.6c72220000000p-6, 0x1.a6a4900000000p-9, 0x1.b438480000000p-6, -0x1.59028c0000000p-5, -0x1.a9bd400000000p-5, -0x1.469ad40000000p-3, -0x1.1c14ae0000000p-3, 0x1.d6368c0000000p-4, 0x1.531d920000000p-4, 0x1.40a8300000000p-7, -0x1.a106e80000000p-4, -0x1.e0bc3e0000000p-4, -0x1.99586a0000000p-5, -0x1.3313f60000000p-3}
, {-0x1.c63b900000000p-3, -0x1.bbbb3a0000000p-4, -0x1.80ab960000000p-4, -0x1.c602020000000p-4, -0x1.7d58080000000p-4, -0x1.9d0c2e0000000p-5, 0x1.6a85aa0000000p-7, -0x1.971bce0000000p-6, -0x1.2d40580000000p-3, 0x1.fb2ab20000000p-4, -0x1.7a7c860000000p-3, 0x1.67aa800000000p-3, -0x1.db02700000000p-4, -0x1.f3c00a0000000p-5, -0x1.49ce460000000p-4, -0x1.7554200000000p-3, -0x1.aea21c0000000p-4, -0x1.301b980000000p-6, 0x1.625c420000000p-6, 0x1.0aa7860000000p-2, 0x1.b19bb60000000p-4, 0x1.8b31a20000000p-3, 0x1.60d23c0000000p-4, 0x1.dd5c600000000p-5, -0x1.eb95260000000p-7, -0x1.2fd03e0000000p-2, -0x1.d505660000000p-4, -0x1.b7311a0000000p-4, -0x1.c61c6c0000000p-8, -0x1.50153a0000000p-5, 0x1.4d6a500000000p-4, 0x1.5c85240000000p-3, 0x1.447bc60000000p-4, 0x1.d113220000000p-10, -0x1.b7ee3e0000000p-3, -0x1.0415d00000000p-2, 0x1.74aa560000000p-4, -0x1.f34b8a0000000p-3, 0x1.a2abec0000000p-6, 0x1.ac3a780000000p-5, -0x1.00c05c0000000p-3, 0x1.3195b40000000p-5, -0x1.6cfc8a0000000p-6, 0x1.10aa000000000p-6, -0x1.5135aa0000000p-4, -0x1.015aba0000000p-5, -0x1.46c9b20000000p-6, -0x1.8367be0000000p-5, -0x1.95df580000000p-3, 0x1.857aba0000000p-4, 0x1.2715120000000p-3, 0x1.bc36480000000p-5, -0x1.19b4600000000p-5, 0x1.5c55da0000000p-4, 0x1.6e093e0000000p-6, -0x1.0a68d00000000p-3, -0x1.0f523e0000000p-3, 0x1.fd36180000000p-5, -0x1.3fc2960000000p-3, 0x1.cdab760000000p-3, 0x1.3527b20000000p-3, 0x1.1154720000000p-3, -0x1.fcae140000000p-3, -0x1.65c9a20000000p-4}
}
, {{-0x1.39bc9a0000000p-3, -0x1.433e820000000p-4, -0x1.98b43e0000000p-6, 0x1.d1c23a0000000p-8, -0x1.857aa00000000p-4, 0x1.daf4080000000p-5, 0x1.776cba0000000p-4, 0x1.0f72c60000000p-5, 0x1.8faea80000000p-5, 0x1.8dea080000000p-3, 0x1.3dbe980000000p-5, -0x1.6b01100000000p-7, 0x1.5351e40000000p-6, 0x1.a90f400000000p-7, 0x1.9459c80000000p-5, 0x1.375dce0000000p-4, 0x1.467d1a0000000p-6, -0x1.ce09840000000p-4, -0x1.2640ee0000000p-4, -0x1.ece4500000000p-7, 0x1.1d2de20000000p-3, -0x1.41a4720000000p-3, 0x1.b2f6580000000p-6, 0x1.5ec1600000000p-6, -0x1.44b8980000000p-4, 0x1.a781ca0000000p-5, 0x1.83b0000000000p-4, 0x1.721a780000000p-5, 0x1.ca7b1c0000000p-5, 0x1.2bf89a0000000p-3, 0x1.7c0cd20000000p-4, -0x1.888b740000000p-3, 0x1.2f50200000000p-4, -0x1.3f48800000000p-4, -0x1.7713be0000000p-3, -0x1.b6a9f00000000p-3, 0x1.770dbc0000000p-4, -0x1.8b845c0000000p-6, -0x1.0076be0000000p-3, -0x1.1afc300000000p-6, 0x1.a49e780000000p-5, 0x1.3aed860000000p-5, 0x1.0a4bca0000000p-5, 0x1.fd2ca00000000p-5, 0x1.4980dc0000000p-5, 0x1.eec6720000000p-5, 0x1.387c600000000p-5, 0x1.22e2580000000p-5, -0x1.193d520000000p-4, -0x1.ccbf620000000p-5, 0x1.9bacc40000000p-7, -0x1.b376b80000000p-5, -0x1.bcc85a0000000p-6, 0x1.d3a8720000000p-4, -0x1.370c260000000p-4, 0x1.feef460000000p-3, -0x1.7301be0000000p-4, 0x1.60e9400000000p-5, 0x1.6ecaf20000000p-7, 0x1.541fba0000000p-4, 0x1.ac4a0c0000000p-5, -0x1.31428a0000000p-3, 0x1.1926c20000000p-4, 0x1.36a6560000000p-3}
, {0x1.4992220000000p-6, 0x1.7423780000000p-5, -0x1.df28e20000000p-6, 0x1.90cea40000000p-5, -0x1.12c55c0000000p-8, 0x1.adda000000000p-5, 0x1.adbaa40000000p-5, 0x1.14402c0000000p-3, -0x1.271a8a0000000p-4, 0x1.11ff100000000p-3, 0x1.160a1e0000000p-4, -0x1.46ec660000000p-7, 0x1.1bf3560000000p-4, -0x1.3e592a0000000p-6, -0x1.1ec9240000000p-4, 0x1.0dd6500000000p-4, -0x1.1e68560000000p-3, -0x1.6046540000000p-4, 0x1.7543820000000p-5, 0x1.c664340000000p-5, 0x1.7eb7fa0000000p-3, 0x1.ea539c0000000p-6, 0x1.1ebf9c0000000p-3, 0x1.31728e0000000p-3, -0x1.6d54b60000000p-5, -0x1.3e4dd80000000p-5, 0x1.a2c8400000000p-5, -0x1.24c7980000000p-4, 0x1.0bf6300000000p-5, -0x1.63a5a00000000p-3, 0x1.ec8e3e0000000p-4, 0x1.193bb20000000p-3, 0x1.300f340000000p-3, -0x1.8603280000000p-3, -0x1.edee380000000p-4, 0x1.c636040000000p-5, 0x1.1577660000000p-2, -0x1.b88d840000000p-5, -0x1.404c860000000p-7, -0x1.0b59780000000p-4, 0x1.21ffb00000000p-3, 0x1.170dd00000000p-3, 0x1.3e58d00000000p-4, -0x1.3879c20000000p-3, 0x1.d27b640000000p-3, 0x1.bf4fc20000000p-4, 0x1.17c4620000000p-2, 0x1.1ee5f60000000p-7, 0x1.db704e0000000p-4, -0x1.5823280000000p-5, 0x1.e222000000000p-7, 0x1.25077a0000000p-5, -0x1.9d21960000000p-4, 0x1.9d8b9c0000000p-7, -0x1.2373da0000000p-4, -0x1.5163f00000000p-5, 0x1.60c8a40000000p-3, -0x1.4ca2840000000p-3, -0x1.93de100000000p-5, 0x1.2d90c80000000p-3, 0x1.d0b7d80000000p-7, -0x1.8e4c640000000p-6, 0x1.5379b20000000p-4, 0x1.44e70c0000000p-6}
, {-0x1.2ada5a0000000p-3, -0x1.0a63640000000p-3, -0x1.0a66580000000p-2, -0x1.6db8440000000p-4, 0x1.cd28640000000p-5, 0x1.8208920000000p-4, -0x1.81db060000000p-3, -0x1.c0abe00000000p-4, -0x1.bf8f1a0000000p-3, -0x1.e49dd80000000p-4, 0x1.5af5e20000000p-4, -0x1.d020f00000000p-8, -0x1.a40a480000000p-6, 0x1.a658d40000000p-6, -0x1.a152080000000p-3, -0x1.2e3cd40000000p-4, 0x1.fdfbaa0000000p-4, 0x1.14dba20000000p-3, -0x1.da0cf60000000p-3, -0x1.023da20000000p-4, -0x1.9692000000000p-3, -0x1.2123dc0000000p-5, 0x1.a237460000000p-3, -0x1.453e580000000p-3, 0x1.13b9a60000000p-4, -0x1.058dca0000000p-2, 0x1.50abd00000000p-7, 0x1.f330e20000000p-5, 0x1.d4bfb20000000p-5, 0x1.faee6c0000000p-5, -0x1.14810c0000000p-3, 0x1.9b21240000000p-4, 0x1.fcd87c0000000p-3, 0x1.adb8fe0000000p-5, 0x1.355e000000000p-4, 0x1.1af9960000000p-2, 0x1.66b78e0000000p-3, -0x1.07e8da0000000p-2, 0x1.13a6160000000p-4, 0x1.3553f40000000p-4, 0x1.b4b1220000000p-3, -0x1.6f7e9a0000000p-4, -0x1.c4215e0000000p-10, -0x1.1d6bd20000000p-5, -0x1.5ca62c0000000p-3, 0x1.485dce0000000p-9, 0x1.1e2e000000000p-4, 0x1.f4f0760000000p-4, -0x1.490f060000000p-8, -0x1.e14aaa0000000p-5, 0x1.3121dc0000000p-5, -0x1.dffe1c0000000p-4, 0x1.1bd6e00000000p-5, 0x1.a7c5a20000000p-3, -0x1.20abca0000000p-5, 0x1.307b940000000p-4, 0x1.c544e60000000p-3, -0x1.308ab00000000p-2, 0x1.75b73e0000000p-4, 0x1.3d3bba0000000p-3, 0x1.4af5520000000p-4, 0x1.38655e0000000p-5, -0x1.5b54ae0000000p-5, 0x1.815d660000000p-9}
}
, {{0x1.c891a40000000p-5, 0x1.48ef760000000p-3, 0x1.bcca9a0000000p-3, -0x1.5136980000000p-7, -0x1.293a380000000p-3, 0x1.2d05960000000p-11, -0x1.1518ea0000000p-3, 0x1.447c280000000p-3, 0x1.8920aa0000000p-5, 0x1.a320d40000000p-9, -0x1.f216ae0000000p-6, 0x1.4793300000000p-4, 0x1.0d504c0000000p-4, -0x1.e811e60000000p-3, 0x1.56e9840000000p-13, -0x1.162d760000000p-4, -0x1.ed11460000000p-5, -0x1.86ffd00000000p-4, -0x1.207c860000000p-4, 0x1.626fb20000000p-5, -0x1.6a73f40000000p-8, 0x1.b24e8c0000000p-5, 0x1.17b3540000000p-5, 0x1.7bd80e0000000p-3, 0x1.ebee660000000p-5, -0x1.24079c0000000p-6, 0x1.e3993a0000000p-3, -0x1.8fe0480000000p-4, -0x1.c0fa940000000p-5, -0x1.fe79520000000p-9, -0x1.7b798c0000000p-4, -0x1.00d1940000000p-3, 0x1.3c5e660000000p-3, 0x1.1728240000000p-2, 0x1.a13fa60000000p-3, 0x1.1fb8980000000p-4, 0x1.cb51a80000000p-4, -0x1.2b08200000000p-5, -0x1.fd20de0000000p-4, 0x1.a691720000000p-2, -0x1.46ab2e0000000p-3, 0x1.4395fe0000000p-7, -0x1.0becb40000000p-4, -0x1.29f50e0000000p-5, -0x1.003ee40000000p-3, 0x1.ef6d260000000p-4, -0x1.7fe1d00000000p-7, 0x1.8e64f20000000p-5, 0x1.71202c0000000p-8, -0x1.3b22460000000p-7, 0x1.42413e0000000p-2, -0x1.55a5460000000p-6, 0x1.d793ea0000000p-5, 0x1.d770360000000p-5, -0x1.467da60000000p-6, 0x1.b9800a0000000p-6, -0x1.447ba40000000p-3, -0x1.924efc0000000p-4, 0x1.0726ea0000000p-3, 0x1.cfd6500000000p-4, -0x1.b635dc0000000p-4, -0x1.68b75a0000000p-5, -0x1.b4ddf80000000p-7, 0x1.06b9ee0000000p-4}
, {-0x1.d148f00000000p-4, 0x1.3102560000000p-4, 0x1.742afe0000000p-6, -0x1.9c8fbc0000000p-4, -0x1.f464f60000000p-6, 0x1.57324e0000000p-4, -0x1.120b600000000p-3, 0x1.8b77320000000p-4, 0x1.843d7e0000000p-7, -0x1.f1999c0000000p-4, 0x1.96d47a0000000p-4, 0x1.5a98fa0000000p-3, 0x1.4250700000000p-7, 0x1.9aee900000000p-6, -0x1.7577b80000000p-5, -0x1.0aee460000000p-5, 0x1.7990b60000000p-4, -0x1.baf43c0000000p-8, -0x1.a655ce0000000p-5, 0x1.e587260000000p-7, 0x1.cfde540000000p-5, 0x1.e4285c0000000p-6, -0x1.0e304c0000000p-3, 0x1.8493ba0000000p-4, -0x1.6682840000000p-4, 0x1.c4a8520000000p-7, 0x1.9324240000000p-3, -0x1.c38e3c0000000p-4, 0x1.6fb5f20000000p-10, 0x1.3f1d460000000p-4, -0x1.a3202a0000000p-4, 0x1.087cac0000000p-6, 0x1.195bde0000000p-4, -0x1.fcacd40000000p-7, 0x1.1f079c0000000p-4, 0x1.2fca960000000p-4, 0x1.3df0de0000000p-5, -0x1.2addf60000000p-5, -0x1.0370d20000000p-5, 0x1.0f0c6a0000000p-4, -0x1.7597260000000p-6, -0x1.09a8a60000000p-4, -0x1.5375ee0000000p-5, -0x1.92ad680000000p-3, -0x1.f5bbae0000000p-4, -0x1.a979e60000000p-4, -0x1.235ca80000000p-3, -0x1.2dbc080000000p-6, 0x1.6485b00000000p-5, -0x1.f420dc0000000p-5, 0x1.f2bfe40000000p-5, 0x1.00e88e0000000p-4, 0x1.dd82780000000p-5, 0x1.46f2aa0000000p-5, 0x1.a01a120000000p-6, 0x1.59d5c60000000p-9, -0x1.d506780000000p-7, 0x1.9cb2560000000p-5, -0x1.3d60d80000000p-5, -0x1.2600f20000000p-2, 0x1.48228a0000000p-6, -0x1.f622140000000p-6, -0x1.0c3c3a0000000p-3, -0x1.af998a0000000p-5}
, {-0x1.85e1fe0000000p-5, 0x1.3cf34c0000000p-4, 0x1.e27c000000000p-3, 0x1.8805c20000000p-6, 0x1.ccf9620000000p-5, 0x1.8f6bd20000000p-3, 0x1.a093a60000000p-10, -0x1.2854460000000p-5, -0x1.6768920000000p-5, 0x1.bfb7ee0000000p-4, -0x1.8dc3920000000p-4, -0x1.35b7f60000000p-6, 0x1.82a55c0000000p-5, 0x1.8644c60000000p-3, 0x1.911d260000000p-5, 0x1.22064a0000000p-3, 0x1.91acd20000000p-3, 0x1.2719740000000p-4, -0x1.fd1fea0000000p-7, 0x1.c2c5a80000000p-6, 0x1.788cc40000000p-5, -0x1.8c588a0000000p-4, -0x1.0eb6680000000p-3, 0x1.df8be00000000p-4, -0x1.1d82340000000p-3, 0x1.6972b80000000p-5, 0x1.8377140000000p-4, -0x1.1e6dea0000000p-7, -0x1.4bf6780000000p-4, -0x1.9d19ca0000000p-5, -0x1.b416d60000000p-4, -0x1.a1484e0000000p-9, 0x1.6e4cae0000000p-3, -0x1.3326580000000p-4, -0x1.e5a1f60000000p-6, 0x1.5b13fe0000000p-3, 0x1.ecdf840000000p-5, 0x1.0beee80000000p-4, 0x1.14cc6e0000000p-4, -0x1.315d520000000p-6, -0x1.58f4540000000p-4, -0x1.5849160000000p-4, 0x1.2706b20000000p-6, -0x1.4bea2c0000000p-4, -0x1.ed164e0000000p-3, 0x1.7778ac0000000p-4, -0x1.2efe5c0000000p-3, 0x1.62e6a60000000p-5, 0x1.49c7780000000p-3, -0x1.9fe58e0000000p-6, 0x1.c0797a0000000p-3, 0x1.ac681c0000000p-4, 0x1.0b7d260000000p-4, 0x1.6d818a0000000p-5, 0x1.53df360000000p-4, 0x1.86028c0000000p-3, 0x1.5358e80000000p-5, 0x1.2c6cf20000000p-3, 0x1.67cf140000000p-5, 0x1.a877360000000p-8, 0x1.83fbd00000000p-3, -0x1.6078100000000p-4, 0x1.6bf7fe0000000p-7, 0x1.ce51540000000p-3}
}
, {{-0x1.a1ee6a0000000p-7, -0x1.5674ae0000000p-7, -0x1.3c92a20000000p-4, 0x1.6de2f20000000p-4, -0x1.70b2880000000p-3, -0x1.8c35d20000000p-3, 0x1.bdd5a60000000p-5, 0x1.a394340000000p-4, -0x1.76a4360000000p-9, 0x1.5765980000000p-4, 0x1.13611c0000000p-3, -0x1.7675fa0000000p-4, 0x1.03a1400000000p-3, 0x1.f6aae40000000p-5, 0x1.ab935c0000000p-4, 0x1.aad0ae0000000p-3, 0x1.0203c00000000p-4, 0x1.edc0260000000p-4, 0x1.e548780000000p-5, 0x1.c012a40000000p-3, 0x1.a706c00000000p-4, -0x1.d98c7a0000000p-4, -0x1.b69aa00000000p-5, -0x1.a938900000000p-4, 0x1.5336bc0000000p-4, -0x1.9f05480000000p-5, -0x1.c1cacc0000000p-10, -0x1.3f30dc0000000p-6, 0x1.4c33000000000p-3, 0x1.19ac100000000p-4, 0x1.6a793c0000000p-4, -0x1.ee83f60000000p-7, -0x1.f2343e0000000p-4, -0x1.0666c20000000p-4, -0x1.00d92e0000000p-4, -0x1.c531f60000000p-4, 0x1.56d0780000000p-8, 0x1.d99be20000000p-7, -0x1.9b09340000000p-4, -0x1.773d820000000p-6, -0x1.3c45fa0000000p-8, 0x1.c42d2a0000000p-6, 0x1.43dc2c0000000p-5, 0x1.0ba96c0000000p-5, -0x1.a2c2aa0000000p-4, 0x1.09651c0000000p-3, -0x1.704ede0000000p-6, -0x1.24e7b00000000p-5, -0x1.9aa8e20000000p-4, 0x1.ff89c00000000p-5, -0x1.7066920000000p-2, -0x1.00785e0000000p-4, -0x1.8dad760000000p-5, 0x1.e5a2da0000000p-4, 0x1.5bc95e0000000p-4, 0x1.eed0160000000p-6, -0x1.ff94dc0000000p-5, 0x1.672fac0000000p-4, -0x1.67a1d80000000p-3, -0x1.9750c80000000p-3, -0x1.d6f0340000000p-6, -0x1.eb837e0000000p-5, -0x1.142d920000000p-4, -0x1.296d2a0000000p-3}
, {0x1.24bb4c0000000p-5, -0x1.8689800000000p-3, 0x1.02b9080000000p-4, 0x1.d2b16e0000000p-5, 0x1.37182a0000000p-7, -0x1.c323ae0000000p-4, 0x1.9d6e260000000p-5, 0x1.099c840000000p-3, -0x1.1ca5e60000000p-3, 0x1.a48dae0000000p-8, 0x1.0e27c60000000p-8, 0x1.faf1a00000000p-12, 0x1.64318a0000000p-3, -0x1.139c040000000p-4, 0x1.f221560000000p-3, 0x1.38d5480000000p-3, -0x1.4b04640000000p-3, 0x1.3da64a0000000p-4, 0x1.67ba760000000p-3, 0x1.7ecbd80000000p-3, 0x1.fd02220000000p-3, -0x1.ffa4ac0000000p-4, 0x1.2d497c0000000p-4, 0x1.31163e0000000p-3, 0x1.abff8e0000000p-5, -0x1.0f39ee0000000p-2, 0x1.4a83f80000000p-4, 0x1.51f63c0000000p-4, -0x1.19e8ce0000000p-3, 0x1.73eaa60000000p-4, -0x1.6bc46e0000000p-4, -0x1.36aa7e0000000p-5, 0x1.ae26d80000000p-4, 0x1.5f14180000000p-5, 0x1.0419ca0000000p-4, 0x1.c469b60000000p-9, 0x1.5930820000000p-8, -0x1.0ffaee0000000p-4, -0x1.048a600000000p-4, -0x1.0732e80000000p-2, -0x1.0ce3020000000p-2, 0x1.3ecd500000000p-5, -0x1.4208a00000000p-5, -0x1.0e2e020000000p-2, 0x1.3706e40000000p-5, 0x1.f2dec80000000p-4, -0x1.8462c00000000p-3, 0x1.74c7d40000000p-6, 0x1.034a700000000p-9, -0x1.f73ca80000000p-6, 0x1.43d1460000000p-11, 0x1.c104980000000p-4, -0x1.5007fc0000000p-4, -0x1.6745bc0000000p-4, -0x1.f8eab80000000p-6, 0x1.8f4c0a0000000p-5, -0x1.9fb8bc0000000p-4, 0x1.25facc0000000p-7, 0x1.23c7c80000000p-5, 0x1.e1aa2a0000000p-7, 0x1.3742dc0000000p-5, -0x1.4852000000000p-7, 0x1.2ee8be0000000p-3, -0x1.3aed520000000p-4}
, {-0x1.4941c00000000p-5, -0x1.3ff04a0000000p-4, -0x1.e9e57e0000000p-2, -0x1.31add40000000p-3, 0x1.1d53220000000p-2, 0x1.46bb660000000p-2, -0x1.b087820000000p-3, -0x1.0b67ba0000000p-2, -0x1.3069520000000p-1, -0x1.9833a80000000p-2, -0x1.00b5ea0000000p-3, -0x1.a4649e0000000p-5, -0x1.876c400000000p-5, 0x1.4eb5ee0000000p-7, -0x1.3bfebc0000000p-2, -0x1.38cf5c0000000p-2, -0x1.592c300000000p-3, 0x1.5308580000000p-3, -0x1.2b89920000000p-2, 0x1.1de3720000000p-3, -0x1.4e25d20000000p-3, 0x1.7b57700000000p-5, 0x1.222c280000000p-2, -0x1.4c82ce0000000p-2, 0x1.5f7c940000000p-3, -0x1.5478900000000p-1, -0x1.19d04a0000000p-2, -0x1.03f9240000000p-5, 0x1.46e7040000000p-3, 0x1.1972300000000p-2, -0x1.624a360000000p-6, 0x1.c5261e0000000p-3, 0x1.667d000000000p-3, 0x1.60ae940000000p-4, -0x1.25ab560000000p-5, 0x1.4102b80000000p-3, 0x1.57e8ac0000000p-3, -0x1.6b1ca60000000p-4, -0x1.ce5d920000000p-4, 0x1.16da200000000p-3, 0x1.8563500000000p-2, -0x1.eb30ea0000000p-6, -0x1.7cd0b80000000p-4, -0x1.19b2360000000p-2, -0x1.77a6120000000p-3, -0x1.9ce0800000000p-2, 0x1.3c90300000000p-5, -0x1.3bb1f20000000p-3, 0x1.bfa4aa0000000p-4, 0x1.6196c60000000p-4, -0x1.3079e60000000p-3, -0x1.c2d6120000000p-3, 0x1.a2e74e0000000p-3, 0x1.40bbd40000000p-3, -0x1.ebbcd40000000p-6, 0x1.fd847a0000000p-2, 0x1.2bda840000000p-4, -0x1.45b9480000000p-1, 0x1.5c87740000000p-2, 0x1.1ce04a0000000p-2, 0x1.8310160000000p-3, -0x1.4047380000000p-3, 0x1.fbf0680000000p-4, 0x1.57a2d60000000p-2}
}
, {{0x1.9450dc0000000p-3, 0x1.6082160000000p-4, -0x1.082dc80000000p-4, -0x1.2276bc0000000p-5, -0x1.f56d7e0000000p-5, -0x1.7c5b560000000p-3, -0x1.6751800000000p-6, 0x1.580db20000000p-4, -0x1.cedf9a0000000p-9, -0x1.3c74c40000000p-4, 0x1.45ba580000000p-4, 0x1.8797860000000p-3, 0x1.2076840000000p-3, -0x1.84d7240000000p-3, -0x1.3fde600000000p-3, 0x1.fd3b480000000p-4, -0x1.fcd5b00000000p-3, -0x1.06b5fa0000000p-5, 0x1.8fec860000000p-5, 0x1.4239d60000000p-4, 0x1.13c91a0000000p-6, 0x1.0d1c600000000p-4, 0x1.ad6fe00000000p-4, 0x1.82a8380000000p-3, -0x1.47b8880000000p-3, 0x1.7ae87e0000000p-4, -0x1.0e88c60000000p-6, -0x1.b444aa0000000p-3, -0x1.5045260000000p-4, -0x1.8d6ffe0000000p-4, 0x1.9c4a8e0000000p-4, -0x1.64f8fe0000000p-6, 0x1.750b8c0000000p-5, -0x1.2691120000000p-4, 0x1.be32460000000p-4, -0x1.d7ef340000000p-4, -0x1.1bea080000000p-4, -0x1.64eb900000000p-4, -0x1.e58bd00000000p-5, 0x1.0c52d60000000p-3, 0x1.8e66880000000p-6, 0x1.b824c20000000p-4, -0x1.84093a0000000p-7, -0x1.c3960e0000000p-4, -0x1.95c8fa0000000p-3, -0x1.d479800000000p-5, 0x1.7d10020000000p-5, -0x1.418ccc0000000p-2, 0x1.e56d8e0000000p-4, 0x1.4995140000000p-5, 0x1.6085700000000p-5, -0x1.1245040000000p-4, 0x1.0828580000000p-8, -0x1.87671a0000000p-4, -0x1.8363900000000p-3, -0x1.bc190c0000000p-3, -0x1.a672a40000000p-4, 0x1.66fc820000000p-5, 0x1.1f6c700000000p-4, -0x1.31f19e0000000p-5, 0x1.6223e00000000p-5, 0x1.c594bc0000000p-5, -0x1.5cec9a0000000p-4, -0x1.56429c0000000p-3}
, {-0x1.82ecde0000000p-9, -0x1.9f29ac0000000p-5, 0x1.e83e7a0000000p-6, 0x1.e36b600000000p-6, 0x1.721e720000000p-5, -0x1.2512480000000p-4, -0x1.3b05480000000p-7, 0x1.4ad99a0000000p-5, 0x1.7631cc0000000p-6, -0x1.0f16880000000p-16, -0x1.cdfbee0000000p-7, -0x1.77300c0000000p-5, 0x1.eebdc60000000p-5, -0x1.7015740000000p-5, -0x1.a0c1d00000000p-4, -0x1.6dec500000000p-6, -0x1.3e250a0000000p-5, 0x1.74fb1e0000000p-5, 0x1.e0cb9e0000000p-5, 0x1.bf63760000000p-6, -0x1.c524e80000000p-4, 0x1.266dd60000000p-3, 0x1.1f48320000000p-3, 0x1.5a39400000000p-4, -0x1.bff7120000000p-4, 0x1.d54b7e0000000p-6, 0x1.4270400000000p-6, -0x1.0c37360000000p-4, -0x1.bdad900000000p-4, -0x1.c7e6340000000p-4, -0x1.dcb9c60000000p-3, -0x1.7142220000000p-4, -0x1.0a61e80000000p-3, 0x1.a7b44e0000000p-6, 0x1.1561800000000p-4, 0x1.2b54980000000p-6, 0x1.addfb40000000p-4, -0x1.131f680000000p-3, 0x1.8a65d00000000p-5, 0x1.f720860000000p-6, 0x1.bf41ea0000000p-4, 0x1.edaff00000000p-4, -0x1.94e4160000000p-4, -0x1.bde5060000000p-4, 0x1.a89d520000000p-5, 0x1.4f26be0000000p-4, 0x1.6f74900000000p-7, -0x1.51d1440000000p-5, 0x1.8314d20000000p-7, 0x1.42a4000000000p-4, -0x1.a9d16e0000000p-5, -0x1.22e4740000000p-3, 0x1.061e760000000p-3, 0x1.9de08a0000000p-6, -0x1.11d4160000000p-9, -0x1.bca2ba0000000p-5, 0x1.827c280000000p-5, 0x1.379baa0000000p-3, -0x1.6169220000000p-6, 0x1.334a9c0000000p-3, -0x1.fe9c020000000p-4, 0x1.4ae7600000000p-4, -0x1.7d76e40000000p-6, -0x1.ff4eda0000000p-6}
, {-0x1.80a5420000000p-6, -0x1.22f1ba0000000p-5, -0x1.52b6900000000p-5, 0x1.db844e0000000p-5, -0x1.17d1080000000p-3, -0x1.163e3c0000000p-4, 0x1.dd26b60000000p-5, 0x1.4abfca0000000p-6, 0x1.0fc8c80000000p-3, -0x1.1c97020000000p-3, -0x1.2c092e0000000p-5, -0x1.2c0edc0000000p-4, 0x1.c5f5f80000000p-7, -0x1.0e84e60000000p-4, -0x1.775ec60000000p-4, 0x1.0aaf440000000p-4, -0x1.7edde20000000p-3, -0x1.5effb60000000p-3, 0x1.41e6d80000000p-3, 0x1.4edbf20000000p-3, -0x1.afc0280000000p-4, 0x1.44fa0c0000000p-3, 0x1.4b8e980000000p-4, -0x1.433c420000000p-5, 0x1.e3c1f80000000p-7, -0x1.a038840000000p-4, -0x1.fd3c4c0000000p-4, 0x1.29b8b20000000p-6, -0x1.bb50400000000p-8, -0x1.863ae80000000p-5, 0x1.778e800000000p-4, 0x1.2d2b580000000p-6, -0x1.4a2b4a0000000p-2, -0x1.3482580000000p-3, 0x1.70c5b20000000p-4, 0x1.6f36820000000p-5, -0x1.ebae060000000p-7, -0x1.e71f0c0000000p-8, 0x1.0e14560000000p-9, -0x1.32c6280000000p-9, 0x1.02afae0000000p-4, -0x1.1248660000000p-3, -0x1.51a35a0000000p-3, -0x1.4b08980000000p-3, 0x1.c8d8820000000p-5, -0x1.46da420000000p-5, 0x1.84350e0000000p-4, -0x1.1d9dd60000000p-3, -0x1.065d200000000p-6, -0x1.31ef180000000p-4, 0x1.4508720000000p-7, -0x1.711a9e0000000p-3, -0x1.0bd5600000000p-3, -0x1.720c040000000p-3, -0x1.b115d60000000p-3, -0x1.d867d00000000p-4, -0x1.12d0520000000p-6, -0x1.5731580000000p-4, 0x1.1b9b8c0000000p-7, 0x1.970b840000000p-5, 0x1.ddc8200000000p-6, -0x1.b3c38e0000000p-9, -0x1.18b7a80000000p-2, -0x1.b5acd20000000p-4}
}
, {{-0x1.79e3560000000p-4, 0x1.f6fe3c0000000p-5, -0x1.646d600000000p-4, -0x1.0a6ee20000000p-6, 0x1.34762a0000000p-6, 0x1.43f4000000000p-2, -0x1.005f380000000p-4, -0x1.c03f8a0000000p-5, -0x1.bcc84e0000000p-4, 0x1.3092fe0000000p-4, 0x1.28843e0000000p-10, 0x1.2a7b360000000p-5, 0x1.07e7f40000000p-5, 0x1.7e650a0000000p-7, -0x1.0c01780000000p-3, -0x1.4a245a0000000p-3, 0x1.0932740000000p-5, 0x1.4c66340000000p-3, 0x1.ebe6a20000000p-4, -0x1.86e5ee0000000p-3, 0x1.8ef80c0000000p-3, -0x1.0f53740000000p-9, -0x1.cca5240000000p-7, 0x1.5674ae0000000p-3, 0x1.d485d60000000p-4, -0x1.a593640000000p-4, -0x1.039a080000000p-7, 0x1.42a82a0000000p-3, -0x1.bc1c260000000p-6, 0x1.5206080000000p-2, -0x1.79cf1a0000000p-5, -0x1.7573e00000000p-6, -0x1.6269780000000p-3, -0x1.591b020000000p-3, 0x1.5ae7d00000000p-3, -0x1.fdf2b00000000p-4, 0x1.3a3b280000000p-5, 0x1.c221a60000000p-6, 0x1.7f18520000000p-3, 0x1.583e900000000p-4, 0x1.707e640000000p-4, -0x1.30b4dc0000000p-5, 0x1.44443e0000000p-4, 0x1.8415400000000p-4, 0x1.bc51f40000000p-4, -0x1.9461f60000000p-3, 0x1.ee970a0000000p-3, 0x1.6f46ee0000000p-4, -0x1.cfe67c0000000p-11, -0x1.d6d8d00000000p-3, -0x1.a20d9c0000000p-7, -0x1.1e08aa0000000p-3, -0x1.1d0fc20000000p-4, 0x1.15b1800000000p-4, 0x1.c802280000000p-3, 0x1.90dcde0000000p-5, -0x1.3c22720000000p-9, -0x1.a1271e0000000p-7, 0x1.6f3fc00000000p-2, -0x1.410e720000000p-4, -0x1.3431b00000000p-7, -0x1.f758260000000p-5, 0x1.c5be680000000p-4, 0x1.2442560000000p-5}
, {-0x1.5b30200000000p-5, 0x1.57fea60000000p-5, 0x1.75c5160000000p-3, -0x1.da5d0e0000000p-4, -0x1.7f56780000000p-4, 0x1.339cca0000000p-2, 0x1.abc20c0000000p-4, 0x1.ccb59a0000000p-4, 0x1.26213e0000000p-4, 0x1.7b717c0000000p-4, -0x1.e55e300000000p-7, 0x1.c83e160000000p-4, 0x1.2102d00000000p-2, 0x1.8447780000000p-7, -0x1.03e0e00000000p-3, 0x1.e7ff3c0000000p-3, 0x1.4d99740000000p-7, -0x1.26c0760000000p-3, 0x1.6222f40000000p-4, -0x1.4bbbb20000000p-3, 0x1.c942da0000000p-3, 0x1.4159b20000000p-3, -0x1.58fb5e0000000p-4, 0x1.e05f1c0000000p-3, -0x1.c666220000000p-4, -0x1.49ecc00000000p-4, 0x1.d266d60000000p-4, -0x1.36bfae0000000p-3, -0x1.4c44ec0000000p-3, 0x1.ab6f020000000p-5, -0x1.a1b1ee0000000p-4, -0x1.ca60360000000p-5, -0x1.f0f0fc0000000p-4, 0x1.625a3c0000000p-4, 0x1.db4de60000000p-4, 0x1.9641b00000000p-3, -0x1.06739a0000000p-7, 0x1.eff13a0000000p-4, 0x1.c199540000000p-5, 0x1.4ecbb80000000p-3, 0x1.35619a0000000p-5, 0x1.14ddb80000000p-2, -0x1.a0c2ba0000000p-6, 0x1.767afc0000000p-8, -0x1.57bd2a0000000p-5, 0x1.b97b720000000p-3, 0x1.2eed7a0000000p-4, 0x1.01e0c40000000p-3, 0x1.6163820000000p-4, -0x1.396a4e0000000p-4, -0x1.2452820000000p-4, -0x1.61ae8e0000000p-3, 0x1.2afd5c0000000p-6, 0x1.cd80440000000p-6, 0x1.4776be0000000p-5, -0x1.58018a0000000p-3, -0x1.50c7860000000p-7, 0x1.6f75e40000000p-3, 0x1.0aeae20000000p-3, 0x1.025b7c0000000p-4, -0x1.0c3f8a0000000p-4, -0x1.2914780000000p-4, -0x1.9ce7860000000p-5, -0x1.5169920000000p-7}
, {-0x1.f7f0160000000p-4, -0x1.bdb2420000000p-7, 0x1.e6d2f20000000p-5, 0x1.3dd4560000000p-4, -0x1.65920a0000000p-4, 0x1.99cc020000000p-4, 0x1.1702d40000000p-4, -0x1.524a240000000p-9, -0x1.f6f2cc0000000p-5, -0x1.0b25c80000000p-6, 0x1.7b8f840000000p-3, 0x1.2bf4560000000p-5, 0x1.acedaa0000000p-4, 0x1.421f680000000p-4, -0x1.6216da0000000p-5, 0x1.29420c0000000p-2, 0x1.3de7a20000000p-3, 0x1.2db99c0000000p-3, -0x1.2d07ea0000000p-6, -0x1.02035c0000000p-2, 0x1.5ccea60000000p-3, -0x1.3233380000000p-5, -0x1.516c060000000p-2, 0x1.2adbf40000000p-4, -0x1.d2234e0000000p-4, 0x1.01d8840000000p-4, 0x1.e102aa0000000p-9, -0x1.9d46a20000000p-4, -0x1.eada520000000p-5, 0x1.13881c0000000p-8, -0x1.6339cc0000000p-4, 0x1.798a2e0000000p-4, -0x1.8d5b880000000p-5, 0x1.99ccd80000000p-5, -0x1.e690e00000000p-5, -0x1.01ecae0000000p-4, 0x1.08dd120000000p-3, 0x1.61b2760000000p-6, -0x1.0033600000000p-6, -0x1.9190e80000000p-5, -0x1.b939c40000000p-3, 0x1.5b4eec0000000p-6, -0x1.0154260000000p-3, -0x1.645a240000000p-3, -0x1.cfa5d20000000p-6, 0x1.4a9b0e0000000p-3, -0x1.748e560000000p-4, 0x1.fbfef40000000p-6, -0x1.6fe4d40000000p-4, -0x1.be48c80000000p-5, -0x1.73d7100000000p-3, -0x1.9400f20000000p-3, 0x1.317ba40000000p-4, 0x1.08ab380000000p-3, 0x1.1492e60000000p-3, 0x1.28189e0000000p-4, -0x1.a58dac0000000p-6, 0x1.d78aea0000000p-4, -0x1.5c13940000000p-5, -0x1.5ce71c0000000p-3, -0x1.4872dc0000000p-3, 0x1.09db9e0000000p-5, -0x1.b01d680000000p-5, -0x1.9868c20000000p-5}
}
, {{0x1.b0c4ce0000000p-5, 0x1.231fca0000000p-4, 0x1.5269ea0000000p-5, 0x1.7138900000000p-5, 0x1.c3a1900000000p-4, 0x1.876c0e0000000p-3, -0x1.535eda0000000p-5, -0x1.d23c020000000p-5, -0x1.06a7fe0000000p-4, 0x1.22b11a0000000p-5, 0x1.0b2c940000000p-4, 0x1.b09bf80000000p-4, -0x1.b657f60000000p-5, 0x1.79667e0000000p-4, -0x1.8605bc0000000p-5, 0x1.c80c920000000p-3, 0x1.745b4c0000000p-3, 0x1.33b5da0000000p-5, -0x1.7f5e120000000p-6, 0x1.1300440000000p-6, -0x1.0a95640000000p-5, -0x1.33cd940000000p-5, 0x1.fb19c40000000p-8, -0x1.c68b660000000p-11, 0x1.2829840000000p-12, -0x1.7a050a0000000p-5, 0x1.26ef4a0000000p-5, 0x1.69923c0000000p-7, 0x1.1f89ec0000000p-3, 0x1.a6ee1a0000000p-6, -0x1.13fc900000000p-5, 0x1.c1685a0000000p-6, 0x1.40e09e0000000p-4, 0x1.002f0a0000000p-4, 0x1.a929e80000000p-4, 0x1.18d3a80000000p-3, 0x1.a3a49a0000000p-4, -0x1.5cbcc20000000p-9, -0x1.31d59a0000000p-6, 0x1.0d40ae0000000p-7, -0x1.786ef60000000p-6, 0x1.3f16ae0000000p-3, -0x1.882ab80000000p-4, -0x1.ae90160000000p-3, -0x1.3ddf340000000p-7, -0x1.1011320000000p-4, 0x1.b726420000000p-5, 0x1.1970160000000p-5, 0x1.4c2e320000000p-3, -0x1.7cb0e20000000p-6, -0x1.1f80500000000p-4, -0x1.9515580000000p-4, -0x1.3472be0000000p-6, 0x1.411a320000000p-3, -0x1.5542940000000p-8, 0x1.0256320000000p-3, 0x1.c6442e0000000p-3, -0x1.f983f00000000p-8, 0x1.665ba80000000p-3, 0x1.1585f20000000p-4, 0x1.f8220c0000000p-4, 0x1.113e940000000p-5, 0x1.2ec1d20000000p-7, -0x1.c4f2cc0000000p-5}
, {-0x1.dbf23e0000000p-8, 0x1.d7e6820000000p-5, -0x1.2be2920000000p-4, -0x1.3edf680000000p-4, -0x1.c41bd00000000p-4, 0x1.41c45e0000000p-3, 0x1.d06e000000000p-4, -0x1.7593b00000000p-4, -0x1.0376740000000p-3, 0x1.422f4a0000000p-3, 0x1.0439280000000p-3, -0x1.e744480000000p-5, 0x1.7a70bc0000000p-4, 0x1.3bdc100000000p-4, 0x1.7199a40000000p-4, 0x1.65d9920000000p-4, 0x1.324ab00000000p-3, -0x1.aae09a0000000p-7, 0x1.6629740000000p-6, -0x1.006f8e0000000p-4, -0x1.0be9740000000p-3, 0x1.c0d5b20000000p-5, -0x1.f4ec800000000p-7, -0x1.8851040000000p-4, -0x1.5772820000000p-3, -0x1.9afa700000000p-6, -0x1.7b8c5e0000000p-4, 0x1.74d5fc0000000p-3, 0x1.e835b80000000p-5, 0x1.b60e8c0000000p-3, 0x1.b001ac0000000p-4, 0x1.fcdef60000000p-6, 0x1.eae7920000000p-5, 0x1.df8dcc0000000p-4, 0x1.0280c40000000p-2, 0x1.b017b00000000p-3, -0x1.aef6760000000p-5, 0x1.e363500000000p-4, 0x1.1389ba0000000p-6, 0x1.44781c0000000p-3, -0x1.7ff7b60000000p-4, 0x1.8aa1720000000p-5, -0x1.d2967e0000000p-4, 0x1.d672f40000000p-4, -0x1.a6b4d80000000p-4, 0x1.04c0ea0000000p-4, -0x1.e8b56a0000000p-5, 0x1.bf0afe0000000p-4, 0x1.e159500000000p-4, -0x1.0bbb500000000p-4, 0x1.41fc1e0000000p-4, 0x1.1ae2e80000000p-5, -0x1.7a0a740000000p-5, 0x1.0b44f60000000p-3, 0x1.0be3540000000p-3, 0x1.fbc6620000000p-5, 0x1.3e1a5a0000000p-6, 0x1.2ac50e0000000p-4, 0x1.5d48980000000p-5, 0x1.9657480000000p-6, 0x1.00da940000000p-3, -0x1.0881680000000p-5, 0x1.4714140000000p-4, 0x1.fe63460000000p-4}
, {-0x1.0aafb00000000p-6, 0x1.265bbc0000000p-5, -0x1.4430820000000p-7, 0x1.705d020000000p-5, 0x1.5419600000000p-4, 0x1.73f2920000000p-5, 0x1.c14b4e0000000p-4, -0x1.2db56c0000000p-9, -0x1.f8dfae0000000p-8, 0x1.9e21ec0000000p-3, 0x1.28dcda0000000p-4, 0x1.3608e80000000p-3, -0x1.9dcc260000000p-6, 0x1.423b2c0000000p-3, -0x1.0d4f780000000p-6, 0x1.94fb580000000p-4, 0x1.052d860000000p-4, 0x1.edfc8e0000000p-5, -0x1.aa72220000000p-4, -0x1.7707f20000000p-3, -0x1.5a442a0000000p-7, 0x1.704d2c0000000p-4, -0x1.4011040000000p-4, 0x1.2da8820000000p-5, -0x1.b3e5ea0000000p-4, -0x1.2de0c00000000p-8, -0x1.8a9e8e0000000p-5, 0x1.cba1d60000000p-5, 0x1.4534760000000p-9, 0x1.6dc21a0000000p-4, 0x1.f90ff40000000p-8, 0x1.da89920000000p-4, 0x1.ac59240000000p-4, 0x1.4d29060000000p-5, 0x1.3024860000000p-4, 0x1.983cd60000000p-3, -0x1.4f8d200000000p-4, -0x1.70af1a0000000p-6, -0x1.5aac3a0000000p-4, 0x1.1ba5b20000000p-6, -0x1.6a51060000000p-5, 0x1.1618e80000000p-3, 0x1.c7fb160000000p-6, 0x1.8f02fe0000000p-4, -0x1.02a17c0000000p-3, 0x1.e452de0000000p-4, 0x1.370ec80000000p-6, 0x1.3b7d1c0000000p-4, 0x1.326f800000000p-4, -0x1.96d0900000000p-4, 0x1.53ae380000000p-3, -0x1.084a8c0000000p-4, -0x1.756d880000000p-5, 0x1.937b580000000p-3, -0x1.67c3d00000000p-6, 0x1.06ba2e0000000p-2, 0x1.ae24960000000p-5, 0x1.df4b920000000p-4, -0x1.6c05cc0000000p-4, -0x1.08fe700000000p-6, 0x1.fbfa800000000p-5, 0x1.1c1cbc0000000p-5, 0x1.8ae3820000000p-4, 0x1.113c5a0000000p-4}
}
, {{0x1.4656d20000000p-4, 0x1.78be8c0000000p-3, -0x1.f0220a0000000p-6, 0x1.0b21940000000p-6, 0x1.19ba5c0000000p-4, -0x1.fc76600000000p-5, 0x1.e1f22a0000000p-4, 0x1.bbb3400000000p-3, 0x1.9f10000000000p-3, 0x1.31fad40000000p-4, 0x1.5498860000000p-3, 0x1.9a946c0000000p-4, 0x1.011cac0000000p-4, 0x1.93d4da0000000p-3, -0x1.b3e3ec0000000p-3, 0x1.1794820000000p-3, 0x1.48804c0000000p-6, -0x1.b81fee0000000p-4, 0x1.2bdf620000000p-4, -0x1.07d69e0000000p-3, -0x1.5487260000000p-4, 0x1.46081e0000000p-6, -0x1.6576560000000p-4, 0x1.3882760000000p-5, 0x1.3ab7ec0000000p-5, 0x1.b9ae5a0000000p-4, -0x1.b4037a0000000p-7, 0x1.28826c0000000p-3, 0x1.425ea60000000p-5, -0x1.1e30640000000p-4, -0x1.58100c0000000p-4, 0x1.07a3360000000p-3, 0x1.ff24b40000000p-4, 0x1.91d4bc0000000p-4, -0x1.4a0fa60000000p-4, -0x1.89e5f20000000p-4, -0x1.557a000000000p-6, -0x1.0681fa0000000p-8, 0x1.af2bb20000000p-2, 0x1.3402840000000p-4, 0x1.8356fc0000000p-4, 0x1.b0b2fa0000000p-4, 0x1.43ad280000000p-4, 0x1.f449620000000p-7, -0x1.17062e0000000p-3, -0x1.4dae240000000p-5, -0x1.6421ee0000000p-8, -0x1.5f8c7a0000000p-3, -0x1.06020c0000000p-5, -0x1.d23e0c0000000p-3, 0x1.9c1f480000000p-3, 0x1.3f13200000000p-2, 0x1.cf2f3e0000000p-5, 0x1.b2660a0000000p-4, 0x1.18fa280000000p-3, -0x1.26b8c40000000p-7, 0x1.67294c0000000p-9, 0x1.152d380000000p-3, 0x1.2e6f5e0000000p-2, -0x1.36e4680000000p-3, -0x1.3b39e60000000p-4, -0x1.14b3060000000p-7, -0x1.64793c0000000p-4, 0x1.5896c40000000p-3}
, {0x1.479fe20000000p-4, 0x1.54a7620000000p-6, 0x1.6aaa3a0000000p-4, 0x1.32e0c20000000p-3, 0x1.3135600000000p-3, 0x1.2558540000000p-3, 0x1.1344460000000p-4, 0x1.4729340000000p-3, 0x1.9724ae0000000p-4, 0x1.693d940000000p-3, 0x1.46e4ba0000000p-3, 0x1.ab54100000000p-10, -0x1.11faa20000000p-3, 0x1.98cb2a0000000p-7, -0x1.423f100000000p-7, -0x1.71518e0000000p-5, 0x1.60a8ee0000000p-6, -0x1.0658000000000p-3, -0x1.3602b20000000p-4, -0x1.3901960000000p-4, 0x1.b8b1360000000p-4, 0x1.046bda0000000p-4, -0x1.b31e980000000p-4, 0x1.489f660000000p-3, -0x1.b2dd380000000p-5, 0x1.066a020000000p-4, 0x1.c1385e0000000p-3, -0x1.4021b00000000p-4, 0x1.e4b0ee0000000p-4, 0x1.f310f20000000p-4, -0x1.9ed91e0000000p-4, 0x1.7cad400000000p-4, 0x1.cb6e180000000p-4, 0x1.37d5a20000000p-5, 0x1.06a9d80000000p-7, 0x1.9d599c0000000p-3, -0x1.4d96100000000p-7, 0x1.bd3e940000000p-4, 0x1.31a2ca0000000p-5, 0x1.4cb9fa0000000p-3, -0x1.dc82900000000p-4, 0x1.f285be0000000p-4, 0x1.da9cae0000000p-3, -0x1.e7423e0000000p-7, 0x1.31d6ac0000000p-5, 0x1.5781120000000p-3, 0x1.2aefa20000000p-3, 0x1.5bbd840000000p-2, -0x1.a4bdaa0000000p-7, -0x1.4ceb5c0000000p-2, 0x1.5dc3740000000p-3, 0x1.2d704e0000000p-5, 0x1.47232e0000000p-5, 0x1.9a808e0000000p-3, 0x1.220f0e0000000p-4, -0x1.1548de0000000p-4, 0x1.1bf24a0000000p-4, 0x1.edeef20000000p-5, 0x1.6491720000000p-3, 0x1.5261220000000p-5, -0x1.2f73bc0000000p-4, 0x1.458e200000000p-4, 0x1.ee5aca0000000p-4, 0x1.84f0d80000000p-6}
, {-0x1.8435aa0000000p-4, 0x1.d85aa00000000p-5, 0x1.94dfca0000000p-3, 0x1.b240e00000000p-5, -0x1.ad1b540000000p-4, -0x1.afa03a0000000p-5, 0x1.22afea0000000p-3, 0x1.6aaebc0000000p-4, 0x1.99a91c0000000p-6, -0x1.d9572c0000000p-4, -0x1.22b8ea0000000p-4, 0x1.a9577a0000000p-5, 0x1.214d800000000p-3, 0x1.e8ed860000000p-4, -0x1.c2cf9e0000000p-6, 0x1.2788040000000p-4, -0x1.4849500000000p-6, -0x1.0c53ea0000000p-6, 0x1.669c7c0000000p-7, -0x1.313ad80000000p-3, 0x1.195aa20000000p-3, -0x1.a714860000000p-4, -0x1.362ce00000000p-5, -0x1.7700460000000p-6, 0x1.9b3e7a0000000p-4, 0x1.031b300000000p-3, 0x1.5e82d80000000p-4, 0x1.106a4e0000000p-4, -0x1.49e63a0000000p-5, 0x1.5a53c20000000p-5, -0x1.6162ac0000000p-3, -0x1.114b540000000p-3, 0x1.aa4cb00000000p-3, 0x1.638c8e0000000p-6, 0x1.8693960000000p-7, -0x1.04346e0000000p-4, 0x1.788cfe0000000p-4, 0x1.0a6bba0000000p-4, -0x1.f4edb20000000p-5, 0x1.3fa6c80000000p-3, -0x1.d52c660000000p-6, 0x1.7f09420000000p-3, -0x1.06bd6c0000000p-3, -0x1.153ac00000000p-6, -0x1.45f97c0000000p-3, 0x1.1ee8ba0000000p-7, 0x1.2bc46a0000000p-2, 0x1.76989e0000000p-4, -0x1.86c2560000000p-6, -0x1.e024360000000p-3, 0x1.7cec900000000p-6, 0x1.9b362c0000000p-4, 0x1.a7befe0000000p-6, 0x1.321c020000000p-4, -0x1.9b540c0000000p-4, -0x1.7287f80000000p-5, 0x1.8a21060000000p-4, 0x1.a482ec0000000p-3, -0x1.02acc60000000p-5, -0x1.efcb180000000p-9, -0x1.1e63d60000000p-2, -0x1.9729c60000000p-3, -0x1.aa9a3e0000000p-3, -0x1.89da0c0000000p-7}
}
, {{-0x1.f0ef9e0000000p-4, 0x1.14b2080000000p-4, -0x1.9778440000000p-3, 0x1.ac2a540000000p-7, 0x1.41bc5e0000000p-4, 0x1.b350240000000p-6, 0x1.1706f00000000p-4, -0x1.22902c0000000p-3, -0x1.44cfd00000000p-3, -0x1.d5a6cc0000000p-4, -0x1.6034fe0000000p-5, -0x1.2960f00000000p-3, 0x1.0342940000000p-3, 0x1.1e5f980000000p-3, 0x1.4b9af00000000p-4, 0x1.1a55b00000000p-3, 0x1.e154da0000000p-5, -0x1.8cf5c20000000p-3, -0x1.0ad0460000000p-3, -0x1.a49c2c0000000p-4, 0x1.ca660c0000000p-5, -0x1.c4e65a0000000p-5, -0x1.9330ae0000000p-3, 0x1.366fd60000000p-10, 0x1.361b1a0000000p-4, 0x1.af54c60000000p-7, 0x1.21f0e80000000p-7, -0x1.22e16c0000000p-3, 0x1.3683d80000000p-5, 0x1.c9d4920000000p-5, -0x1.a5eb9a0000000p-5, 0x1.de81380000000p-3, 0x1.6520e60000000p-6, 0x1.21e2960000000p-5, -0x1.10d8980000000p-6, 0x1.2cd4000000000p-4, 0x1.765a8a0000000p-6, 0x1.d29cdc0000000p-4, -0x1.a650ec0000000p-5, -0x1.81c85e0000000p-4, -0x1.c5fb200000000p-5, 0x1.86e9b60000000p-7, 0x1.8d6f9a0000000p-3, -0x1.2c58b40000000p-3, -0x1.05a7400000000p-4, 0x1.fab7d40000000p-4, 0x1.10b8020000000p-5, 0x1.35ea0a0000000p-6, 0x1.fcfe7a0000000p-4, -0x1.8b11800000000p-5, 0x1.7dffbc0000000p-4, 0x1.3aa6c80000000p-6, 0x1.0f00900000000p-4, 0x1.e668c00000000p-4, 0x1.aac6fe0000000p-4, -0x1.287d2a0000000p-8, -0x1.d9ab6c0000000p-4, -0x1.30220a0000000p-6, 0x1.98d8060000000p-5, -0x1.05b2140000000p-5, 0x1.b58c120000000p-7, -0x1.4da7920000000p-3, -0x1.e974ac0000000p-4, 0x1.6152cc0000000p-3}
, {-0x1.6707ba0000000p-3, 0x1.d149980000000p-4, 0x1.82dbba0000000p-6, -0x1.14762a0000000p-3, 0x1.7f74ee0000000p-4, 0x1.fb98040000000p-6, 0x1.f733e20000000p-3, 0x1.2e775c0000000p-4, 0x1.a76bfc0000000p-5, 0x1.2731cc0000000p-4, 0x1.ba39660000000p-5, -0x1.7c40f00000000p-3, -0x1.90bc540000000p-4, 0x1.897c140000000p-8, 0x1.02b2ca0000000p-4, 0x1.471eac0000000p-5, 0x1.c3f4f80000000p-3, -0x1.b5c1a40000000p-4, 0x1.48636c0000000p-6, -0x1.15a1d80000000p-7, -0x1.1b530c0000000p-3, 0x1.f2e7980000000p-6, -0x1.2c724a0000000p-3, 0x1.64584a0000000p-3, -0x1.14450e0000000p-5, -0x1.f329100000000p-4, -0x1.45f2560000000p-9, 0x1.92e50c0000000p-4, 0x1.afa0e80000000p-5, -0x1.9569720000000p-6, -0x1.53aaee0000000p-4, 0x1.0773060000000p-3, 0x1.c676d00000000p-4, -0x1.a06f6a0000000p-6, 0x1.267c0a0000000p-7, 0x1.0ef8000000000p-9, 0x1.06586c0000000p-5, -0x1.0abe5a0000000p-4, 0x1.f645680000000p-4, -0x1.5c42a20000000p-7, 0x1.4ab4540000000p-3, -0x1.5e9edc0000000p-4, -0x1.19dfec0000000p-4, -0x1.5515260000000p-4, -0x1.7d4cc60000000p-3, 0x1.5fc63e0000000p-5, -0x1.d25ee20000000p-5, -0x1.9db46a0000000p-3, 0x1.3b81c80000000p-5, -0x1.7cbcb00000000p-3, 0x1.06f72c0000000p-5, 0x1.68aee40000000p-3, 0x1.3ff1ae0000000p-3, 0x1.4dcc160000000p-3, 0x1.fe74cc0000000p-8, 0x1.eb6d0c0000000p-4, -0x1.a4677a0000000p-4, -0x1.a13a460000000p-6, 0x1.c5721e0000000p-5, 0x1.5dc4fe0000000p-3, -0x1.2b795c0000000p-4, 0x1.5d13880000000p-6, 0x1.b129360000000p-7, 0x1.4d2abe0000000p-4}
, {-0x1.a0fed80000000p-4, 0x1.5530ce0000000p-3, -0x1.b89fa60000000p-4, -0x1.7101560000000p-3, -0x1.52320c0000000p-6, 0x1.cc2dac0000000p-5, 0x1.333cb60000000p-3, -0x1.1497040000000p-4, -0x1.a8702a0000000p-4, -0x1.ac175e0000000p-4, 0x1.4f20fa0000000p-4, -0x1.98d9e20000000p-3, 0x1.2770fc0000000p-4, -0x1.0ad1a20000000p-4, -0x1.af18300000000p-4, 0x1.6784f40000000p-4, 0x1.9eee620000000p-5, -0x1.3566b00000000p-3, -0x1.15b98a0000000p-6, -0x1.8a12740000000p-4, -0x1.2b18aa0000000p-4, 0x1.12c3f40000000p-3, -0x1.0265180000000p-3, -0x1.752db40000000p-4, 0x1.4304c20000000p-5, 0x1.126b520000000p-4, -0x1.3daeba0000000p-5, -0x1.6c0c760000000p-3, 0x1.af8b460000000p-3, 0x1.06051e0000000p-3, -0x1.baaaac0000000p-5, 0x1.d72f000000000p-5, 0x1.1c099a0000000p-5, 0x1.24c43a0000000p-12, 0x1.f943c00000000p-4, 0x1.3867d40000000p-3, 0x1.122b960000000p-3, 0x1.f1caaa0000000p-4, 0x1.79b1860000000p-5, 0x1.d18faa0000000p-3, 0x1.5bcd640000000p-4, -0x1.7b67520000000p-6, 0x1.da7d880000000p-4, -0x1.24baa00000000p-3, -0x1.dbcb5c0000000p-5, -0x1.00ec7a0000000p-3, 0x1.dfb2720000000p-6, -0x1.6c39e00000000p-3, 0x1.b8c7280000000p-4, 0x1.b9d4780000000p-5, 0x1.4141c80000000p-5, 0x1.33d93a0000000p-4, -0x1.ce232c0000000p-5, 0x1.bb14740000000p-5, 0x1.2b6f140000000p-4, 0x1.383c300000000p-3, -0x1.08d8440000000p-4, -0x1.04c8e60000000p-3, 0x1.0428640000000p-5, -0x1.d2aca80000000p-6, 0x1.58f47a0000000p-4, -0x1.a296720000000p-4, -0x1.24dc4e0000000p-3, 0x1.7ad92c0000000p-4}
}
, {{-0x1.9538c80000000p-4, -0x1.7899d40000000p-4, -0x1.98f88e0000000p-3, -0x1.402ba60000000p-5, -0x1.26bfe20000000p-4, 0x1.a4c1ae0000000p-2, -0x1.46a0260000000p-3, 0x1.de9a5e0000000p-3, -0x1.17b1540000000p-2, -0x1.8de1680000000p-6, 0x1.7e18f20000000p-7, 0x1.e4aa460000000p-5, -0x1.83ff460000000p-3, 0x1.cb9b2c0000000p-5, -0x1.ac814a0000000p-5, 0x1.4744360000000p-4, 0x1.376cd80000000p-5, 0x1.645fae0000000p-2, 0x1.c771560000000p-3, 0x1.c8fe040000000p-5, -0x1.3db73e0000000p-4, -0x1.6b66560000000p-4, -0x1.2d4c740000000p-4, -0x1.2872180000000p-5, 0x1.eb22260000000p-8, -0x1.a4d9e80000000p-3, -0x1.25efb40000000p-5, 0x1.305f260000000p-7, -0x1.b4fed80000000p-4, 0x1.980e700000000p-6, -0x1.3361c20000000p-3, 0x1.6fe43a0000000p-4, 0x1.e227980000000p-7, -0x1.edc0c00000000p-3, 0x1.e2cb180000000p-3, 0x1.ab90600000000p-2, 0x1.91bf460000000p-3, -0x1.0d13fc0000000p-3, -0x1.d13fa20000000p-6, 0x1.0e99420000000p-6, -0x1.127a520000000p-4, 0x1.0ea27e0000000p-3, 0x1.e208e60000000p-7, -0x1.6075040000000p-4, -0x1.984cfa0000000p-3, -0x1.d034b40000000p-5, 0x1.ddd7d60000000p-4, 0x1.4ed7340000000p-3, 0x1.bffb6a0000000p-5, 0x1.eed2c60000000p-5, -0x1.d51e520000000p-6, -0x1.5af01c0000000p-2, 0x1.0b95620000000p-2, 0x1.304c6a0000000p-3, -0x1.46e5a40000000p-3, 0x1.13ae980000000p-2, -0x1.5c72020000000p-4, -0x1.140bec0000000p-2, -0x1.3235240000000p-4, -0x1.4718160000000p-2, 0x1.524ca40000000p-4, 0x1.0f6b9c0000000p-2, -0x1.c984760000000p-4, -0x1.5b9cd40000000p-6}
, {0x1.2575260000000p-6, -0x1.4002740000000p-5, -0x1.535cdc0000000p-4, 0x1.364c2c0000000p-3, -0x1.1588020000000p-3, 0x1.69ac8e0000000p-3, -0x1.bc01c20000000p-4, -0x1.819be00000000p-4, -0x1.cde0980000000p-4, 0x1.4b72780000000p-4, -0x1.d3c6d20000000p-5, 0x1.2fd9840000000p-5, 0x1.072ea80000000p-6, -0x1.2497400000000p-5, -0x1.92dd7c0000000p-6, 0x1.04c72c0000000p-5, -0x1.0ada500000000p-5, 0x1.c237f40000000p-4, -0x1.b682100000000p-4, 0x1.3305a20000000p-5, 0x1.543a020000000p-3, 0x1.0098b80000000p-5, 0x1.77febe0000000p-3, -0x1.e5db880000000p-3, -0x1.bb87f20000000p-6, 0x1.535c2a0000000p-5, -0x1.7ce8d00000000p-3, 0x1.c9b7660000000p-4, -0x1.c4e2bc0000000p-4, 0x1.19011e0000000p-5, 0x1.07f3100000000p-4, -0x1.1c1de20000000p-4, -0x1.26017a0000000p-4, 0x1.0787c00000000p-7, 0x1.02eec80000000p-4, 0x1.9430240000000p-4, 0x1.52d6520000000p-4, -0x1.b7d1b60000000p-6, 0x1.aa6fe00000000p-5, -0x1.8ebd9e0000000p-5, 0x1.285b240000000p-5, -0x1.f9512a0000000p-4, -0x1.7b99400000000p-3, -0x1.fd76d80000000p-5, -0x1.9e2a1c0000000p-5, -0x1.14808a0000000p-2, -0x1.30181c0000000p-4, 0x1.8f0ef60000000p-5, 0x1.2484840000000p-6, -0x1.c37c920000000p-6, 0x1.1061cc0000000p-3, -0x1.528b400000000p-3, -0x1.16fe2c0000000p-3, 0x1.848ca60000000p-4, 0x1.f20e800000000p-7, 0x1.f66c7a0000000p-7, -0x1.309d3a0000000p-5, -0x1.45fdb80000000p-3, 0x1.4b55f00000000p-4, -0x1.94cf640000000p-4, 0x1.9263fc0000000p-5, 0x1.f877ae0000000p-7, -0x1.2ff6640000000p-5, -0x1.02eb200000000p-4}
, {0x1.98da4e0000000p-5, -0x1.8fe1420000000p-9, -0x1.046a660000000p-3, 0x1.49890a0000000p-13, -0x1.3a81740000000p-2, 0x1.7bf1540000000p-3, -0x1.60ce1a0000000p-4, -0x1.72a4740000000p-3, 0x1.2270ca0000000p-4, 0x1.1ac3720000000p-3, -0x1.adc9be0000000p-6, 0x1.f403240000000p-3, 0x1.05b3460000000p-3, -0x1.422d140000000p-4, 0x1.37db280000000p-5, 0x1.e04ffa0000000p-4, -0x1.727b020000000p-3, 0x1.26daec0000000p-3, 0x1.9fd68a0000000p-6, 0x1.74baf80000000p-4, 0x1.4af12c0000000p-4, 0x1.09426c0000000p-7, -0x1.9182fe0000000p-5, -0x1.4e1fb00000000p-4, -0x1.8161680000000p-6, -0x1.81a38a0000000p-3, -0x1.427bc60000000p-4, -0x1.63dfbc0000000p-2, -0x1.a139e60000000p-5, 0x1.8d5a4a0000000p-4, -0x1.63fe060000000p-8, -0x1.b4bebc0000000p-4, -0x1.4702e40000000p-2, -0x1.4a45cc0000000p-7, 0x1.4f96420000000p-2, 0x1.4218640000000p-7, -0x1.c4696e0000000p-3, 0x1.3e0af20000000p-3, -0x1.fd89500000000p-3, -0x1.59b4180000000p-4, -0x1.d877d60000000p-4, -0x1.a262200000000p-3, -0x1.fe03860000000p-3, -0x1.6698780000000p-5, 0x1.e75ea00000000p-6, -0x1.2ee7ce0000000p-5, -0x1.6db51a0000000p-7, 0x1.d398680000000p-5, -0x1.adc6200000000p-4, -0x1.b4d7440000000p-5, 0x1.88dbd60000000p-5, -0x1.1a25a60000000p-3, -0x1.15fd220000000p-4, -0x1.31ed8a0000000p-4, -0x1.fb10a20000000p-9, -0x1.2c0f200000000p-4, -0x1.a7e1d00000000p-4, 0x1.d31cc00000000p-4, -0x1.2ac4c80000000p-4, -0x1.02013c0000000p-2, -0x1.e71c720000000p-7, 0x1.13fc9a0000000p-3, -0x1.23442e0000000p-2, -0x1.5a50140000000p-4}
}
, {{-0x1.9062fc0000000p-6, 0x1.175c8a0000000p-6, 0x1.94c4f80000000p-3, 0x1.644f1e0000000p-4, 0x1.48c4700000000p-4, 0x1.b22b380000000p-5, 0x1.3a335a0000000p-5, 0x1.b8301c0000000p-5, -0x1.99d3200000000p-6, 0x1.b2b2640000000p-5, 0x1.d6d85c0000000p-5, 0x1.29b0bc0000000p-3, 0x1.21a4880000000p-5, 0x1.7017540000000p-3, -0x1.91db280000000p-4, 0x1.cb3d2e0000000p-3, 0x1.1bcc080000000p-4, -0x1.a156f40000000p-4, 0x1.cb711e0000000p-5, 0x1.95370c0000000p-6, 0x1.c7504a0000000p-5, -0x1.04bb0c0000000p-3, 0x1.5931e40000000p-4, 0x1.209b5c0000000p-3, 0x1.6d91420000000p-4, 0x1.d6f4580000000p-3, 0x1.39f13e0000000p-3, -0x1.0123d80000000p-3, 0x1.d74d160000000p-5, -0x1.0069b80000000p-4, 0x1.fd9e600000000p-5, 0x1.276be80000000p-2, 0x1.3a5abc0000000p-3, 0x1.5d290e0000000p-3, 0x1.971f600000000p-4, 0x1.1609880000000p-7, 0x1.c00d9e0000000p-4, -0x1.461d680000000p-5, 0x1.5d6efc0000000p-5, 0x1.a876980000000p-4, -0x1.acc3920000000p-7, 0x1.ac917a0000000p-4, 0x1.c5d91a0000000p-4, -0x1.9a49040000000p-4, -0x1.e46f0a0000000p-4, 0x1.0b08a60000000p-3, -0x1.8e8e8e0000000p-7, 0x1.11e3e80000000p-3, 0x1.ed24fa0000000p-4, 0x1.41cdf00000000p-6, 0x1.09753a0000000p-3, 0x1.ef57e80000000p-4, 0x1.4a945a0000000p-10, 0x1.afd2c20000000p-5, -0x1.10d56a0000000p-3, 0x1.7425720000000p-11, -0x1.09822c0000000p-2, 0x1.444b5e0000000p-3, 0x1.983e9a0000000p-4, 0x1.9d6db20000000p-3, 0x1.0221440000000p-3, 0x1.82d4820000000p-3, -0x1.fcc7440000000p-5, 0x1.1e89ec0000000p-3}
, {-0x1.0ea4220000000p-6, 0x1.ffb4960000000p-4, -0x1.5faace0000000p-9, 0x1.906bda0000000p-4, 0x1.4b681c0000000p-4, 0x1.2cf8c00000000p-4, 0x1.264d160000000p-7, 0x1.be67260000000p-5, 0x1.868b4a0000000p-3, -0x1.0eceb00000000p-5, 0x1.59f1220000000p-4, -0x1.747ea80000000p-4, -0x1.0131b80000000p-8, 0x1.89f8920000000p-3, 0x1.19c7fa0000000p-6, -0x1.bf30b20000000p-4, -0x1.53e4de0000000p-6, -0x1.3beae60000000p-3, -0x1.70e1ec0000000p-5, -0x1.398e460000000p-4, 0x1.e76a4a0000000p-7, -0x1.12d2660000000p-3, 0x1.0cbd020000000p-4, -0x1.73d1280000000p-6, 0x1.37cccc0000000p-4, 0x1.f9f86c0000000p-4, 0x1.5cacb60000000p-5, 0x1.e9275e0000000p-5, 0x1.04d86c0000000p-3, 0x1.e87f400000000p-4, -0x1.0c2dc20000000p-3, 0x1.92404e0000000p-5, 0x1.14e0ea0000000p-2, 0x1.8c7b7c0000000p-5, -0x1.9a56f00000000p-5, -0x1.7f86240000000p-7, -0x1.32ab940000000p-6, 0x1.3a68e80000000p-3, 0x1.d5030a0000000p-4, 0x1.7a04e00000000p-3, -0x1.216b180000000p-3, -0x1.e96d380000000p-6, 0x1.47abb40000000p-4, -0x1.d288c20000000p-10, -0x1.3c322a0000000p-4, 0x1.a97b4c0000000p-4, -0x1.132bd80000000p-5, 0x1.041a2e0000000p-7, 0x1.4a79d40000000p-4, -0x1.114cb00000000p-4, 0x1.5268bc0000000p-4, 0x1.fc48040000000p-5, 0x1.3e924a0000000p-5, -0x1.2178cc0000000p-5, 0x1.0ee4a20000000p-3, 0x1.0251800000000p-3, -0x1.79281a0000000p-4, 0x1.613c120000000p-3, 0x1.4a1aa80000000p-5, 0x1.023e920000000p-3, 0x1.5db1340000000p-3, 0x1.37fe6a0000000p-4, 0x1.74da280000000p-5, 0x1.6962c80000000p-3}
, {-0x1.3bd26c0000000p-3, 0x1.86e36e0000000p-4, 0x1.23d9060000000p-3, 0x1.7861a20000000p-3, 0x1.a2a8220000000p-4, 0x1.5fb8ae0000000p-3, 0x1.babc560000000p-6, 0x1.06d52a0000000p-4, 0x1.2776ba0000000p-2, 0x1.3498bc0000000p-4, 0x1.645aa40000000p-5, -0x1.97dda20000000p-6, 0x1.fb977c0000000p-3, -0x1.1ba7920000000p-4, -0x1.70d67a0000000p-7, 0x1.5a4be60000000p-5, 0x1.1dff220000000p-6, -0x1.070d8e0000000p-3, 0x1.3b77060000000p-4, -0x1.ef0c040000000p-6, 0x1.e3d0be0000000p-3, -0x1.68a1620000000p-4, 0x1.83881a0000000p-3, 0x1.230f6e0000000p-3, 0x1.d6f6f00000000p-4, 0x1.e5a2c60000000p-6, 0x1.24c7a60000000p-9, 0x1.d0e1e80000000p-7, 0x1.1617160000000p-4, 0x1.18ab440000000p-3, -0x1.23b0cc0000000p-3, 0x1.12d3e20000000p-2, 0x1.1925f40000000p-2, 0x1.fbbff00000000p-5, 0x1.7319460000000p-3, 0x1.443d0e0000000p-3, -0x1.5ae6d40000000p-3, 0x1.1698b40000000p-2, 0x1.7ef8560000000p-5, 0x1.0fe73c0000000p-2, -0x1.6316fc0000000p-3, -0x1.394ff80000000p-5, 0x1.75b4c40000000p-5, -0x1.39466a0000000p-3, -0x1.5e4c040000000p-5, 0x1.c6557e0000000p-4, 0x1.fe66000000000p-5, -0x1.2b0e840000000p-7, 0x1.1aa7000000000p-4, -0x1.f970100000000p-4, 0x1.54f6e60000000p-3, 0x1.d6a0760000000p-6, 0x1.9af4920000000p-4, 0x1.234e240000000p-4, 0x1.4e5ca20000000p-3, -0x1.7eb9220000000p-5, -0x1.867aac0000000p-3, 0x1.235d280000000p-2, 0x1.58b7c80000000p-3, 0x1.50b6440000000p-3, 0x1.15c5860000000p-3, -0x1.e8e9980000000p-6, 0x1.783eb80000000p-5, -0x1.7beb540000000p-6}
}
, {{0x1.d1cda60000000p-4, -0x1.875eb00000000p-4, -0x1.11079a0000000p-3, 0x1.e072c40000000p-5, 0x1.0c25200000000p-4, -0x1.2b74c20000000p-3, -0x1.c09f420000000p-5, 0x1.9284180000000p-3, -0x1.acad280000000p-6, -0x1.4425900000000p-4, 0x1.1c307a0000000p-3, 0x1.076d380000000p-6, -0x1.c1bcd60000000p-5, -0x1.7945cc0000000p-5, -0x1.78fdec0000000p-3, 0x1.5f60260000000p-4, -0x1.07b57c0000000p-7, -0x1.5fbe360000000p-3, 0x1.de19da0000000p-4, 0x1.64021a0000000p-4, 0x1.fcb4920000000p-5, -0x1.d834760000000p-4, 0x1.3bbb820000000p-2, -0x1.8f302a0000000p-5, -0x1.11c3600000000p-5, 0x1.0d5e780000000p-5, -0x1.180a9c0000000p-4, -0x1.b2d5e80000000p-3, 0x1.9feb580000000p-6, -0x1.735bd00000000p-4, 0x1.12beee0000000p-5, 0x1.cb50f40000000p-7, 0x1.b1aaae0000000p-7, -0x1.0629980000000p-8, -0x1.369e180000000p-4, 0x1.90585e0000000p-6, 0x1.1664ac0000000p-5, 0x1.0ae65c0000000p-5, -0x1.5f1d840000000p-8, 0x1.60f5100000000p-5, -0x1.dd4c240000000p-3, 0x1.5d94660000000p-3, -0x1.2802640000000p-4, 0x1.5dbe200000000p-5, -0x1.970bac0000000p-3, -0x1.590f420000000p-5, -0x1.a43f440000000p-5, -0x1.4b454c0000000p-4, 0x1.531e800000000p-3, 0x1.5f66a00000000p-4, -0x1.4186d00000000p-3, -0x1.bcabf80000000p-5, -0x1.0744680000000p-5, -0x1.2229100000000p-6, 0x1.c8a8720000000p-6, -0x1.2b02260000000p-8, -0x1.7f45d60000000p-4, 0x1.1155840000000p-5, -0x1.7d7cd80000000p-3, -0x1.cf3fbe0000000p-3, -0x1.8a20b80000000p-3, 0x1.1a4f140000000p-2, -0x1.2ff89a0000000p-3, -0x1.25ac5a0000000p-4}
, {0x1.9fdf100000000p-5, -0x1.b8b0d60000000p-4, -0x1.c905760000000p-4, -0x1.0d7db80000000p-10, 0x1.d11cc80000000p-8, -0x1.325ff40000000p-2, 0x1.21a93e0000000p-3, -0x1.2cd3f20000000p-3, 0x1.bf886e0000000p-5, -0x1.b06ea60000000p-5, 0x1.ddafca0000000p-5, 0x1.2569220000000p-10, 0x1.80dbec0000000p-5, -0x1.3be2260000000p-4, 0x1.35b9500000000p-5, -0x1.4abef80000000p-5, -0x1.50ee0c0000000p-11, -0x1.abfd740000000p-6, 0x1.054d120000000p-4, 0x1.4b18b40000000p-3, -0x1.2ee5500000000p-6, -0x1.7067ac0000000p-4, 0x1.e380ea0000000p-4, 0x1.56a9ec0000000p-4, -0x1.9e2c340000000p-7, -0x1.316c2c0000000p-5, -0x1.1313820000000p-4, -0x1.28360c0000000p-3, 0x1.a08a080000000p-6, 0x1.d8a9980000000p-6, 0x1.523a440000000p-4, -0x1.9c1a060000000p-8, 0x1.099b080000000p-3, -0x1.2785060000000p-4, -0x1.9e7fae0000000p-6, 0x1.aedb480000000p-7, 0x1.75c6fe0000000p-3, 0x1.c534860000000p-5, 0x1.5936b20000000p-5, -0x1.e1b2b40000000p-4, -0x1.0e869c0000000p-3, -0x1.67dc1a0000000p-3, 0x1.410d0a0000000p-4, -0x1.6e7aae0000000p-5, -0x1.0d0a400000000p-4, 0x1.0a0c320000000p-4, -0x1.d44c7e0000000p-4, 0x1.d141a80000000p-7, -0x1.0fe9340000000p-4, 0x1.1e375e0000000p-6, -0x1.dcfb120000000p-6, -0x1.49a1bc0000000p-3, -0x1.137a900000000p-5, -0x1.fd43cc0000000p-4, 0x1.ef8bd80000000p-3, -0x1.35b6c20000000p-3, -0x1.9cbe020000000p-5, 0x1.e7605e0000000p-4, -0x1.6bc89c0000000p-3, 0x1.01c9720000000p-4, -0x1.1b25440000000p-3, -0x1.0704f20000000p-2, -0x1.4d53880000000p-7, -0x1.cf68340000000p-4}
, {0x1.ad205c0000000p-4, -0x1.743c680000000p-4, -0x1.be43380000000p-6, 0x1.cd1bfc0000000p-5, 0x1.34ace00000000p-3, -0x1.bef50c0000000p-4, -0x1.294ef40000000p-4, -0x1.73478a0000000p-4, 0x1.14fb3c0000000p-11, 0x1.7bbc440000000p-3, 0x1.28920e0000000p-3, 0x1.56596a0000000p-3, -0x1.84c5900000000p-8, -0x1.0b15da0000000p-5, 0x1.e523280000000p-4, -0x1.1101020000000p-3, -0x1.89242e0000000p-3, 0x1.03936a0000000p-2, -0x1.b571d80000000p-4, -0x1.af2a1a0000000p-5, -0x1.d072320000000p-5, -0x1.fcb01a0000000p-4, 0x1.6cef6c0000000p-3, 0x1.17bc840000000p-5, -0x1.f5f7080000000p-6, -0x1.bde8b80000000p-5, -0x1.3f41c40000000p-6, 0x1.0cfc180000000p-3, -0x1.94ab520000000p-3, -0x1.d85f440000000p-4, 0x1.3632880000000p-5, -0x1.e77aa60000000p-6, 0x1.84febe0000000p-3, -0x1.37c7e20000000p-6, -0x1.6a8b380000000p-3, -0x1.a585fa0000000p-9, 0x1.81c29c0000000p-5, -0x1.0b09300000000p-4, -0x1.59a1aa0000000p-3, -0x1.38f4c60000000p-3, -0x1.a1e5ca0000000p-3, -0x1.14643e0000000p-3, 0x1.01cb340000000p-8, -0x1.1624c20000000p-5, -0x1.f6cc060000000p-6, -0x1.c18e4e0000000p-8, -0x1.c1cb880000000p-3, 0x1.112b300000000p-2, -0x1.286e720000000p-2, 0x1.e3f82e0000000p-5, -0x1.4fbbda0000000p-3, -0x1.86875e0000000p-4, -0x1.c6dc4c0000000p-4, -0x1.cfb0380000000p-5, 0x1.cf8aa40000000p-4, -0x1.78c6ca0000000p-4, -0x1.1d4b840000000p-5, 0x1.2cef3c0000000p-3, -0x1.6398e40000000p-3, 0x1.2d824e0000000p-4, -0x1.7efdaa0000000p-4, 0x1.8db8360000000p-7, 0x1.6455420000000p-4, 0x1.8b353c0000000p-4}
}
, {{-0x1.5cbb120000000p-3, -0x1.461a660000000p-10, -0x1.3e6b020000000p-4, 0x1.602dd20000000p-6, 0x1.6833760000000p-7, 0x1.1b652a0000000p-4, -0x1.7966d20000000p-6, -0x1.909ef40000000p-4, -0x1.926e020000000p-3, -0x1.f64e300000000p-5, 0x1.2b30d60000000p-3, -0x1.07f22e0000000p-7, -0x1.f605ee0000000p-4, 0x1.52eb4a0000000p-4, -0x1.91825e0000000p-3, 0x1.6491d40000000p-4, 0x1.9361ee0000000p-4, 0x1.143ae40000000p-7, 0x1.2b115c0000000p-5, 0x1.0a02240000000p-6, -0x1.8ca7ae0000000p-5, 0x1.1915b80000000p-5, 0x1.f1f0f60000000p-8, 0x1.2f17ac0000000p-3, 0x1.6778900000000p-5, -0x1.8890fa0000000p-7, -0x1.01f7e60000000p-4, -0x1.4c375c0000000p-4, -0x1.aa7a540000000p-4, -0x1.e36f880000000p-3, -0x1.12f3880000000p-3, 0x1.4cd2520000000p-4, -0x1.c04ff80000000p-4, -0x1.e131020000000p-4, -0x1.6b793e0000000p-6, -0x1.33954a0000000p-5, 0x1.54dd9c0000000p-4, 0x1.44ddaa0000000p-3, 0x1.02c0be0000000p-3, -0x1.79ed4a0000000p-4, -0x1.1a45d60000000p-3, -0x1.1f7fc20000000p-5, -0x1.320f0e0000000p-6, -0x1.4850600000000p-4, 0x1.df1dd80000000p-10, -0x1.d225580000000p-4, -0x1.5ef4de0000000p-7, 0x1.82306e0000000p-4, 0x1.e1c6cc0000000p-7, -0x1.1138d80000000p-6, -0x1.11da1a0000000p-3, 0x1.06b1720000000p-5, -0x1.9868700000000p-4, -0x1.bd24a80000000p-4, -0x1.44f3880000000p-4, 0x1.71bd6c0000000p-8, -0x1.99697c0000000p-7, -0x1.c6ed120000000p-3, 0x1.bdb3f80000000p-4, -0x1.57cc980000000p-3, 0x1.e936a20000000p-6, 0x1.4b95780000000p-5, -0x1.03c2b60000000p-3, -0x1.4c8b7a0000000p-6}
, {-0x1.d8a50c0000000p-4, 0x1.f6ceee0000000p-5, 0x1.33a4960000000p-6, -0x1.7fa4aa0000000p-3, 0x1.3e0fe80000000p-4, 0x1.07c0d20000000p-4, 0x1.3778f00000000p-4, -0x1.0b7f960000000p-3, -0x1.aed03c0000000p-4, 0x1.3678740000000p-5, 0x1.1d26220000000p-3, 0x1.e3a5de0000000p-11, 0x1.cefb4c0000000p-4, 0x1.de80e60000000p-5, -0x1.30b1240000000p-4, -0x1.65f7840000000p-3, 0x1.28a6a20000000p-4, -0x1.7f3af00000000p-4, -0x1.6f3c0a0000000p-5, -0x1.38e3120000000p-3, -0x1.f68a340000000p-8, -0x1.59ae460000000p-6, -0x1.78cb280000000p-5, 0x1.d960b00000000p-4, 0x1.44a5380000000p-5, -0x1.a434f00000000p-4, -0x1.2464760000000p-5, -0x1.942c9e0000000p-5, 0x1.3395640000000p-4, -0x1.01ef3e0000000p-6, -0x1.273b540000000p-8, 0x1.94504a0000000p-3, 0x1.f3bb020000000p-8, -0x1.21af180000000p-4, 0x1.eeaa8e0000000p-6, 0x1.c824540000000p-5, -0x1.71693e0000000p-5, -0x1.4a86d40000000p-7, 0x1.e998be0000000p-4, 0x1.a7afba0000000p-4, -0x1.7cfedc0000000p-5, 0x1.0a5cd00000000p-5, -0x1.7bc2f00000000p-5, -0x1.0bfa760000000p-4, 0x1.732f8e0000000p-5, -0x1.b64a980000000p-7, -0x1.94d7d40000000p-4, -0x1.5977140000000p-4, 0x1.73bf9c0000000p-6, -0x1.130cf40000000p-4, 0x1.b418980000000p-3, 0x1.d059a40000000p-4, 0x1.2adad40000000p-3, -0x1.57d9940000000p-5, 0x1.739ac60000000p-5, -0x1.28f3b00000000p-7, -0x1.6c2ba00000000p-4, -0x1.059c880000000p-5, 0x1.e50f7a0000000p-5, 0x1.d833380000000p-7, -0x1.d475320000000p-4, -0x1.ce7a5c0000000p-4, 0x1.5364180000000p-6, -0x1.11e1820000000p-3}
, {-0x1.a196340000000p-5, 0x1.8a71d40000000p-3, 0x1.f6c50e0000000p-3, 0x1.4911740000000p-4, 0x1.89927a0000000p-4, -0x1.a4c24c0000000p-6, 0x1.d079660000000p-3, -0x1.a2175c0000000p-4, 0x1.8bf9500000000p-5, 0x1.7f7f3e0000000p-3, 0x1.7214b40000000p-3, 0x1.e789b00000000p-6, 0x1.4b722a0000000p-4, 0x1.cfb8060000000p-4, -0x1.4ec87e0000000p-6, -0x1.d142880000000p-4, 0x1.fb8ebe0000000p-4, -0x1.e09f3c0000000p-4, -0x1.ac5d2e0000000p-4, 0x1.6c39260000000p-6, -0x1.39d7320000000p-7, 0x1.5ff1040000000p-7, 0x1.f2c8d20000000p-5, 0x1.74d70a0000000p-3, -0x1.98033e0000000p-3, 0x1.2456fc0000000p-2, 0x1.062fd60000000p-2, 0x1.0727220000000p-2, 0x1.18b1720000000p-5, -0x1.05f1820000000p-5, -0x1.468a340000000p-4, 0x1.0edf2c0000000p-4, 0x1.2f4cc00000000p-4, 0x1.8754e60000000p-5, 0x1.c1899a0000000p-5, -0x1.5135120000000p-10, 0x1.0d74380000000p-3, 0x1.7df97e0000000p-7, 0x1.930cc60000000p-5, 0x1.b27fe60000000p-3, -0x1.12dee00000000p-5, 0x1.60a53e0000000p-3, 0x1.81e3dc0000000p-2, -0x1.27aa160000000p-3, 0x1.c20df00000000p-4, 0x1.a8006e0000000p-3, 0x1.d795000000000p-6, -0x1.5d04c60000000p-6, 0x1.a589f60000000p-5, -0x1.f89e7e0000000p-3, 0x1.3c74e00000000p-2, 0x1.c9a8480000000p-6, 0x1.5f6ab80000000p-3, -0x1.9fc6c00000000p-9, 0x1.ff0eb00000000p-4, -0x1.3a17e60000000p-4, 0x1.a716700000000p-5, 0x1.50ad800000000p-2, 0x1.bef2f00000000p-5, 0x1.d6d7ca0000000p-3, 0x1.6d26460000000p-6, -0x1.d2ec7a0000000p-5, -0x1.13b2ae0000000p-6, 0x1.9e66880000000p-4}
}
, {{-0x1.ae89d40000000p-4, 0x1.ab8a820000000p-4, 0x1.30e6a60000000p-3, 0x1.a377260000000p-5, -0x1.0fda2a0000000p-4, 0x1.bf3dc40000000p-6, -0x1.8c3ac40000000p-5, -0x1.094eb80000000p-5, 0x1.3a3ba40000000p-3, -0x1.34b8a20000000p-3, -0x1.2a81140000000p-3, -0x1.9c1a9e0000000p-3, 0x1.fd9dfa0000000p-5, 0x1.173a900000000p-3, 0x1.3d5ebc0000000p-2, -0x1.0fd48c0000000p-4, -0x1.eb36000000000p-5, 0x1.0d25ea0000000p-5, -0x1.3fce7c0000000p-4, -0x1.df22460000000p-3, -0x1.031b820000000p-2, -0x1.d966740000000p-3, -0x1.6a67980000000p-7, -0x1.15bc140000000p-3, -0x1.19a6a60000000p-3, 0x1.cb33760000000p-8, -0x1.77b0ea0000000p-3, 0x1.2a2bfa0000000p-5, -0x1.0e86ce0000000p-4, 0x1.2b715c0000000p-6, 0x1.c268a00000000p-6, 0x1.01cc100000000p-6, 0x1.f6c6b80000000p-4, 0x1.4ebf160000000p-10, -0x1.909fda0000000p-4, 0x1.751b100000000p-7, -0x1.857f440000000p-4, 0x1.46a88c0000000p-8, -0x1.4ad5020000000p-5, 0x1.e9c3620000000p-8, 0x1.bfe82a0000000p-6, 0x1.30e6500000000p-13, -0x1.3fde3c0000000p-4, -0x1.a82d3a0000000p-5, 0x1.321da80000000p-3, 0x1.530e0c0000000p-4, -0x1.8883520000000p-3, 0x1.2948f00000000p-3, -0x1.ca74ca0000000p-4, -0x1.fab9440000000p-7, -0x1.05c1880000000p-3, -0x1.dd16e00000000p-4, -0x1.c5f8ce0000000p-4, -0x1.0dc7000000000p-3, -0x1.57e1fa0000000p-3, 0x1.0d9f260000000p-5, -0x1.0815d00000000p-5, -0x1.fffc4e0000000p-4, -0x1.dabf5a0000000p-9, 0x1.35da9c0000000p-7, -0x1.0951cc0000000p-3, -0x1.3c3c340000000p-2, -0x1.8c9c260000000p-4, -0x1.419ea40000000p-6}
, {0x1.631ab80000000p-6, -0x1.1e0d6e0000000p-4, -0x1.0907200000000p-6, -0x1.5725960000000p-4, -0x1.e92c6e0000000p-4, 0x1.0e87340000000p-3, -0x1.7593680000000p-5, -0x1.d42fd60000000p-6, -0x1.221e300000000p-3, 0x1.1073240000000p-4, 0x1.79506e0000000p-4, -0x1.0264d00000000p-3, 0x1.6b3b860000000p-8, -0x1.1f584a0000000p-3, -0x1.673d5a0000000p-4, -0x1.b0a0c80000000p-5, -0x1.25f0d60000000p-8, 0x1.5700d40000000p-3, 0x1.3cb4a40000000p-5, -0x1.bcdc560000000p-5, -0x1.bd1f740000000p-3, -0x1.1b67900000000p-3, 0x1.1f51460000000p-4, 0x1.ed535c0000000p-5, -0x1.0fc4380000000p-4, 0x1.0480060000000p-3, 0x1.056c020000000p-7, -0x1.9573160000000p-5, -0x1.a165d00000000p-5, -0x1.e68f3e0000000p-4, 0x1.f25a300000000p-5, 0x1.8ea5660000000p-7, 0x1.01861c0000000p-4, 0x1.9bb8640000000p-5, -0x1.76b03a0000000p-5, -0x1.cf63980000000p-5, -0x1.700c680000000p-6, -0x1.437c020000000p-5, 0x1.8b20980000000p-4, 0x1.6d519e0000000p-4, 0x1.b892c00000000p-3, 0x1.271f020000000p-6, -0x1.5200900000000p-3, -0x1.54892e0000000p-4, -0x1.963d180000000p-4, -0x1.9e8bf80000000p-5, -0x1.65adcc0000000p-4, -0x1.e31c640000000p-6, 0x1.e87b360000000p-6, 0x1.8dfaa20000000p-3, 0x1.20e1bc0000000p-3, -0x1.836e4c0000000p-4, -0x1.2d296c0000000p-3, -0x1.e25fa80000000p-4, -0x1.6b52da0000000p-4, 0x1.0c37ca0000000p-3, -0x1.84cc2a0000000p-8, -0x1.5698160000000p-4, 0x1.f85ecc0000000p-5, -0x1.b5b0300000000p-8, 0x1.bdb9ae0000000p-3, -0x1.7ddb1c0000000p-4, -0x1.c4fd720000000p-4, -0x1.8223da0000000p-5}
, {-0x1.64fcd40000000p-5, -0x1.eef6c40000000p-5, 0x1.505c720000000p-4, -0x1.3bdcec0000000p-8, 0x1.1657ae0000000p-5, -0x1.7bd59a0000000p-4, 0x1.3d8e360000000p-4, 0x1.86bd5c0000000p-5, 0x1.3caab60000000p-4, -0x1.3233f20000000p-4, -0x1.f9aad60000000p-5, -0x1.d35cdc0000000p-4, -0x1.24d46a0000000p-4, -0x1.74526e0000000p-4, 0x1.8818c20000000p-4, -0x1.5416f80000000p-3, -0x1.43c1f60000000p-3, -0x1.fb23040000000p-11, -0x1.32083c0000000p-7, 0x1.d264780000000p-6, 0x1.1831a00000000p-5, -0x1.eccb260000000p-5, 0x1.9097f00000000p-4, -0x1.30953e0000000p-3, -0x1.4511480000000p-6, 0x1.fe5de00000000p-3, -0x1.85bef40000000p-6, -0x1.0915560000000p-3, -0x1.0ddcf20000000p-3, -0x1.f5e68c0000000p-5, 0x1.0289500000000p-4, -0x1.2e1e2e0000000p-7, -0x1.3a32800000000p-6, -0x1.dd66420000000p-4, -0x1.03779c0000000p-3, 0x1.ba59ee0000000p-5, -0x1.85f4260000000p-4, -0x1.5bf97c0000000p-4, 0x1.68c62a0000000p-3, -0x1.555b7e0000000p-5, 0x1.6d61240000000p-4, -0x1.61488a0000000p-4, -0x1.5fb2140000000p-4, -0x1.83896c0000000p-3, 0x1.0114040000000p-4, 0x1.33615c0000000p-5, -0x1.59c93e0000000p-3, -0x1.b369e40000000p-4, 0x1.1edf3e0000000p-4, 0x1.3ed23a0000000p-5, -0x1.73a55a0000000p-3, 0x1.3c23040000000p-4, -0x1.ef45cc0000000p-5, 0x1.4b7bea0000000p-4, 0x1.0e42360000000p-8, -0x1.3772dc0000000p-4, -0x1.da5a520000000p-6, 0x1.85866c0000000p-3, -0x1.1672bc0000000p-5, 0x1.cdeb280000000p-6, -0x1.6b7e4a0000000p-7, -0x1.5133e40000000p-4, -0x1.4e61140000000p-7, 0x1.bc58320000000p-9}
}
, {{0x1.05398a0000000p-4, -0x1.b3bcbc0000000p-4, -0x1.196bf00000000p-1, -0x1.44e5020000000p-6, 0x1.b80ab60000000p-6, 0x1.1e74740000000p-1, -0x1.a22bd80000000p-3, -0x1.ed9d300000000p-3, -0x1.d9923e0000000p-2, -0x1.8508560000000p-2, -0x1.0510700000000p-2, -0x1.1609aa0000000p-8, -0x1.a49adc0000000p-7, -0x1.5bf3180000000p-4, -0x1.e4b37a0000000p-3, 0x1.95ebf80000000p-4, -0x1.7ff8f80000000p-7, 0x1.27c9f20000000p-3, 0x1.e04ed20000000p-6, 0x1.386b380000000p-10, 0x1.16438a0000000p-6, 0x1.b96e140000000p-4, 0x1.d4a9400000000p-5, -0x1.a635f40000000p-6, 0x1.d585b20000000p-3, -0x1.80dd300000000p-3, -0x1.5b51680000000p-2, 0x1.ef6c0e0000000p-5, -0x1.4fbc480000000p-4, 0x1.e68e060000000p-3, 0x1.4f54c80000000p-4, 0x1.6c55440000000p-4, 0x1.6cb4020000000p-3, -0x1.6aaf640000000p-3, 0x1.f919340000000p-2, -0x1.742d640000000p-3, -0x1.0b8e1a0000000p-4, 0x1.37f3da0000000p-5, 0x1.54abf60000000p-4, -0x1.c31ba60000000p-4, 0x1.b0060e0000000p-3, -0x1.42fd4e0000000p-2, 0x1.9cebf00000000p-3, -0x1.25917a0000000p-2, 0x1.d81efe0000000p-8, -0x1.65c4980000000p-2, -0x1.df808a0000000p-9, -0x1.0285de0000000p-1, -0x1.06aebe0000000p-5, 0x1.13e95e0000000p-2, -0x1.2ca8240000000p-4, -0x1.9604360000000p-3, -0x1.50ebd20000000p-3, -0x1.90c2c00000000p-5, 0x1.23a1b20000000p-3, 0x1.e41f140000000p-3, -0x1.6574f60000000p-7, -0x1.24fe740000000p-2, -0x1.49105e0000000p-4, -0x1.93a0a00000000p-7, 0x1.673a0a0000000p-3, 0x1.5abae60000000p-4, -0x1.4ae14e0000000p-4, 0x1.afce6a0000000p-4}
, {0x1.c0e0da0000000p-4, -0x1.32f9220000000p-3, -0x1.9b149e0000000p-4, 0x1.58f0280000000p-8, 0x1.96584c0000000p-9, -0x1.0cce620000000p-3, 0x1.2a4c5c0000000p-3, 0x1.78da600000000p-3, -0x1.d57c660000000p-6, -0x1.a1ecc80000000p-3, -0x1.3fc8720000000p-8, -0x1.6693260000000p-6, 0x1.43c5fc0000000p-4, 0x1.20a95a0000000p-5, -0x1.9702200000000p-4, 0x1.fb10780000000p-9, -0x1.aaaf020000000p-5, -0x1.d36dec0000000p-4, -0x1.8c539c0000000p-4, 0x1.a4f9360000000p-4, 0x1.ec057e0000000p-4, -0x1.2912f20000000p-3, -0x1.f1dbc60000000p-5, 0x1.3eefc60000000p-4, 0x1.a0b9420000000p-7, 0x1.11f96e0000000p-3, 0x1.02b97a0000000p-8, -0x1.de05bc0000000p-4, 0x1.94426a0000000p-4, -0x1.2da9420000000p-4, -0x1.7549900000000p-3, 0x1.b4de940000000p-4, -0x1.8cfb9a0000000p-3, 0x1.4a46f80000000p-3, 0x1.e74c0c0000000p-4, -0x1.62718e0000000p-3, -0x1.5b34d60000000p-4, 0x1.61ea0c0000000p-5, -0x1.277b9a0000000p-3, 0x1.5b70140000000p-7, 0x1.781f9c0000000p-4, 0x1.db68900000000p-6, 0x1.78af5c0000000p-8, -0x1.d316440000000p-5, -0x1.020f220000000p-6, -0x1.a48db00000000p-8, -0x1.c8994c0000000p-5, 0x1.3e8e300000000p-9, -0x1.07a8460000000p-9, 0x1.cc9a160000000p-5, -0x1.0c13620000000p-3, -0x1.f41d6c0000000p-6, 0x1.03c21c0000000p-4, 0x1.2bf7ba0000000p-4, 0x1.2a176c0000000p-4, 0x1.a8b5ae0000000p-7, 0x1.cae7800000000p-6, -0x1.46cc920000000p-5, -0x1.05d2640000000p-2, -0x1.e830e60000000p-4, -0x1.f350440000000p-5, 0x1.83ac780000000p-3, -0x1.12196c0000000p-4, -0x1.8cbd3c0000000p-4}
, {0x1.d703240000000p-5, 0x1.3a40d80000000p-5, -0x1.f4b50a0000000p-3, -0x1.1f78000000000p-3, -0x1.644aea0000000p-4, -0x1.695cb00000000p-6, -0x1.b853960000000p-6, -0x1.90ef480000000p-5, -0x1.0b99980000000p-4, 0x1.3063f00000000p-4, 0x1.c671ba0000000p-5, 0x1.84fe000000000p-3, -0x1.9980060000000p-5, 0x1.633f520000000p-6, -0x1.84f3080000000p-3, 0x1.50208c0000000p-3, -0x1.86e01a0000000p-5, 0x1.224bf80000000p-2, -0x1.01e99e0000000p-4, 0x1.f950a40000000p-4, -0x1.09c45a0000000p-4, 0x1.2834380000000p-7, -0x1.c1cdc40000000p-4, -0x1.2366a00000000p-6, 0x1.033ea60000000p-5, -0x1.0fa28a0000000p-2, -0x1.0fa0560000000p-2, -0x1.743ba00000000p-4, 0x1.fd55a40000000p-8, 0x1.7d54080000000p-4, 0x1.7ca6720000000p-4, 0x1.4ecfda0000000p-3, 0x1.f796da0000000p-4, -0x1.a470ce0000000p-4, -0x1.2467d40000000p-4, -0x1.bcd4100000000p-4, 0x1.cd17440000000p-3, 0x1.5635740000000p-7, -0x1.d67df80000000p-3, -0x1.938f2a0000000p-5, 0x1.deb6120000000p-3, 0x1.a027120000000p-4, -0x1.7297160000000p-3, -0x1.cfed800000000p-6, -0x1.92b3640000000p-4, 0x1.17ccb40000000p-4, -0x1.0768c60000000p-3, -0x1.66520e0000000p-2, -0x1.e3b3240000000p-3, 0x1.4d3bbe0000000p-3, 0x1.712fca0000000p-6, -0x1.95a5bc0000000p-5, 0x1.776d940000000p-4, 0x1.a979a60000000p-3, 0x1.3df8500000000p-6, 0x1.15595a0000000p-3, -0x1.db61920000000p-3, -0x1.1871800000000p-3, -0x1.5ddf1a0000000p-4, -0x1.49ac260000000p-6, 0x1.4ead580000000p-3, 0x1.f013e40000000p-4, 0x1.0cc0b40000000p-3, -0x1.53cf2a0000000p-4}
}
, {{0x1.aaa37a0000000p-5, -0x1.149b160000000p-3, -0x1.b59e920000000p-3, -0x1.9d0a140000000p-3, -0x1.dab5ba0000000p-4, 0x1.92187e0000000p-3, -0x1.b9dff00000000p-4, 0x1.9f68ee0000000p-3, -0x1.4d697e0000000p-3, -0x1.8bbbfe0000000p-4, 0x1.10ca720000000p-3, 0x1.60c1200000000p-3, -0x1.0c88020000000p-4, -0x1.40af500000000p-3, 0x1.1718da0000000p-4, -0x1.0617fc0000000p-2, 0x1.19ff0a0000000p-3, 0x1.ce9b9c0000000p-3, -0x1.8fbb240000000p-4, -0x1.3489b00000000p-9, -0x1.db67c80000000p-8, 0x1.cb99480000000p-4, 0x1.f0b4340000000p-5, 0x1.b42e5c0000000p-4, -0x1.6536400000000p-3, 0x1.aef0260000000p-4, -0x1.99435a0000000p-4, -0x1.1259da0000000p-6, -0x1.44fcec0000000p-6, -0x1.2b9b780000000p-3, 0x1.a930e00000000p-3, 0x1.f7d0280000000p-4, -0x1.530d760000000p-4, -0x1.4d59d40000000p-5, -0x1.9d387a0000000p-8, 0x1.2db8aa0000000p-3, -0x1.72ea700000000p-9, -0x1.82281e0000000p-3, 0x1.1359b40000000p-6, 0x1.5e82020000000p-4, 0x1.fddbf20000000p-4, -0x1.521aa00000000p-6, -0x1.b24b100000000p-7, -0x1.bd714e0000000p-4, -0x1.45c23a0000000p-5, 0x1.436e940000000p-3, 0x1.534d4e0000000p-4, 0x1.645b000000000p-5, 0x1.1906b00000000p-3, 0x1.c809060000000p-4, -0x1.5b71c80000000p-6, -0x1.433e1e0000000p-4, 0x1.47b4820000000p-3, -0x1.78924e0000000p-5, -0x1.601ff40000000p-4, 0x1.3ec84c0000000p-4, -0x1.14a12c0000000p-3, -0x1.2525a60000000p-4, 0x1.50e6ae0000000p-7, -0x1.9ccc780000000p-4, -0x1.c45a340000000p-8, 0x1.31da8e0000000p-2, -0x1.5e02820000000p-3, 0x1.45a7400000000p-5}
, {-0x1.2becac0000000p-4, 0x1.e5cd020000000p-4, -0x1.17de1e0000000p-3, 0x1.9b0b220000000p-5, -0x1.7fde080000000p-7, 0x1.dc55e20000000p-5, -0x1.7c9aea0000000p-5, -0x1.5b42100000000p-5, -0x1.d47c360000000p-5, 0x1.2cf3400000000p-3, 0x1.f0067c0000000p-4, 0x1.a6dae60000000p-11, -0x1.4f07a60000000p-4, 0x1.5ec68c0000000p-4, 0x1.2a771a0000000p-6, -0x1.668d7e0000000p-4, -0x1.01892a0000000p-5, 0x1.79d6960000000p-7, -0x1.f140880000000p-4, 0x1.ecbffc0000000p-4, 0x1.a1ccb40000000p-5, 0x1.50c7860000000p-3, 0x1.06d1560000000p-3, -0x1.39d9200000000p-5, -0x1.ddfaf60000000p-6, 0x1.5815da0000000p-6, -0x1.e725340000000p-6, -0x1.b505580000000p-7, -0x1.8041640000000p-4, -0x1.5dd21a0000000p-4, 0x1.65a3dc0000000p-3, 0x1.60fcee0000000p-5, -0x1.9d0db00000000p-6, 0x1.984dd60000000p-6, -0x1.0355860000000p-3, -0x1.529ea60000000p-3, 0x1.5e52ba0000000p-5, -0x1.5b892c0000000p-3, 0x1.7c3c7e0000000p-5, -0x1.5dd02e0000000p-6, 0x1.7ac21c0000000p-4, -0x1.13baa00000000p-7, -0x1.37ad2c0000000p-4, 0x1.5dbc400000000p-4, 0x1.deffda0000000p-5, -0x1.49db720000000p-5, -0x1.0830720000000p-3, -0x1.092c180000000p-3, -0x1.6bd0720000000p-3, 0x1.a6a1880000000p-5, -0x1.640c740000000p-4, -0x1.e94cc80000000p-5, -0x1.7b22920000000p-3, -0x1.7a72280000000p-3, 0x1.16a4620000000p-4, 0x1.c0effc0000000p-4, -0x1.678f060000000p-4, 0x1.11a9160000000p-6, -0x1.247bea0000000p-8, -0x1.5ef0d60000000p-7, -0x1.4b19f40000000p-4, -0x1.7343da0000000p-3, -0x1.6b55640000000p-5, 0x1.88e5360000000p-5}
, {0x1.26d91c0000000p-9, -0x1.976be80000000p-6, 0x1.de63fe0000000p-5, 0x1.d3f59c0000000p-4, -0x1.63280e0000000p-5, 0x1.af89940000000p-5, -0x1.27e2820000000p-4, 0x1.61ba880000000p-7, -0x1.2846d00000000p-4, 0x1.e9b5e40000000p-3, -0x1.0375c80000000p-3, 0x1.77a5b60000000p-6, -0x1.0554920000000p-4, -0x1.b1dc320000000p-7, 0x1.3692900000000p-5, -0x1.d50e1a0000000p-5, -0x1.244c9c0000000p-3, -0x1.3af9fa0000000p-3, 0x1.62b62e0000000p-5, 0x1.756ba20000000p-3, -0x1.40b5fa0000000p-5, -0x1.0a9d620000000p-4, 0x1.e5e1300000000p-3, 0x1.1e0d980000000p-4, -0x1.0b57860000000p-4, -0x1.0bb48a0000000p-5, 0x1.877b740000000p-3, 0x1.822f600000000p-4, -0x1.a416660000000p-3, -0x1.146aec0000000p-2, 0x1.9434ca0000000p-4, 0x1.71c9a20000000p-4, -0x1.4fac860000000p-3, 0x1.2f77bc0000000p-6, 0x1.a40f820000000p-7, -0x1.7f85360000000p-3, -0x1.0636a80000000p-3, 0x1.16e5fa0000000p-3, 0x1.0992e80000000p-5, -0x1.7df6180000000p-5, -0x1.909cd40000000p-4, -0x1.16755c0000000p-3, 0x1.70a5ce0000000p-3, -0x1.31999a0000000p-4, 0x1.23cd220000000p-2, -0x1.6c166e0000000p-4, -0x1.186c300000000p-7, 0x1.4b7b8e0000000p-10, -0x1.0ebd8e0000000p-3, 0x1.288f680000000p-11, -0x1.091d300000000p-5, -0x1.3e44c80000000p-5, -0x1.1ad4460000000p-5, -0x1.126eb80000000p-2, -0x1.dc8aa40000000p-4, -0x1.1c06d20000000p-2, -0x1.f34c380000000p-4, -0x1.eabce40000000p-5, 0x1.2e8d540000000p-3, 0x1.4995f60000000p-4, -0x1.099eca0000000p-5, -0x1.b644d80000000p-4, -0x1.9bdf9a0000000p-6, 0x1.10ba320000000p-3}
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

typedef float batch_normalization_3_output_type[2][64];

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
#define WEIGHTS_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define NUMBER_T float
#define LONG_NUMBER_T float


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

const float batch_normalization_3_bias[64] = {-0x1.18394c0000000p-1, -0x1.0a340a0000000p-2, -0x1.0924fc0000000p+0, -0x1.882c340000000p-2, -0x1.2074200000000p-3, -0x1.57532c0000000p-2, -0x1.b94dc80000000p-1, -0x1.c9a0560000000p-1, -0x1.5281dc0000000p-2, -0x1.bef2c00000000p-3, -0x1.ae88d20000000p-2, -0x1.8d1a680000000p-2, -0x1.2055460000000p-1, -0x1.b80fd60000000p-2, -0x1.228d280000000p-3, -0x1.1bd4060000000p+0, -0x1.b31e420000000p-3, -0x1.67acd40000000p-1, -0x1.d7c0200000000p-4, -0x1.3ac8780000000p+0, 0x1.6215400000000p-5, -0x1.0386c80000000p+0, -0x1.0cde640000000p-2, -0x1.2a73e80000000p-1, -0x1.6344860000000p-1, -0x1.2088320000000p+0, -0x1.c0effe0000000p-1, -0x1.ddb4840000000p-1, -0x1.2432c40000000p-1, -0x1.de9c420000000p-1, -0x1.3463e40000000p+0, -0x1.c2b8780000000p-1, -0x1.8e53360000000p-1, -0x1.9fd6440000000p-3, -0x1.80f0740000000p+0, -0x1.1fbcb80000000p+0, -0x1.f3ae520000000p-1, -0x1.265d580000000p-2, -0x1.c6271c0000000p-2, -0x1.e56e920000000p-1, -0x1.05bb600000000p+0, -0x1.c7fe000000000p-2, -0x1.e509480000000p-4, -0x1.59cbca0000000p-1, -0x1.6e859c0000000p-2, -0x1.45a8de0000000p-2, -0x1.05a3760000000p+0, -0x1.64c3b20000000p-1, 0x1.bb08400000000p-6, -0x1.6595a20000000p-1, -0x1.0ebc8a0000000p+0, -0x1.22eeb20000000p-2, -0x1.1a89100000000p-3, -0x1.ec71380000000p-1, -0x1.1486720000000p+0, -0x1.0cdbca0000000p+0, -0x1.f445a60000000p-1, -0x1.401e5a0000000p-2, -0x1.fd65f00000000p-1, -0x1.b916560000000p-3, -0x1.00ca540000000p+0, -0x1.9181040000000p-2, 0x1.d6ec700000000p-5, -0x1.6f87f80000000p-2}
;
const float batch_normalization_3_kernel[64] = {0x1.6abfda0000000p-2, 0x1.9bc7d60000000p-2, 0x1.e843f20000000p-3, 0x1.ea28260000000p-2, 0x1.b6e5080000000p-1, 0x1.9e2b120000000p-1, 0x1.809c0e0000000p-2, 0x1.be7d100000000p-3, 0x1.6dabe40000000p-2, 0x1.46d7ec0000000p-1, 0x1.c9fe1c0000000p-2, 0x1.85e16a0000000p-1, 0x1.05c36c0000000p-1, 0x1.2265580000000p-2, 0x1.b3b9840000000p-1, 0x1.ffb3c00000000p-3, 0x1.1da8120000000p+0, 0x1.7999d80000000p-1, 0x1.205eae0000000p-1, 0x1.1e52a80000000p-2, 0x1.ce0b820000000p-2, 0x1.f352dc0000000p-2, 0x1.e26e3e0000000p-1, 0x1.8903120000000p-2, 0x1.4d9d880000000p-2, 0x1.bfef320000000p-3, 0x1.0351460000000p-2, 0x1.a3196c0000000p-3, 0x1.8a20640000000p-2, 0x1.6bd1560000000p-2, 0x1.f00d220000000p-3, 0x1.a2de440000000p-3, 0x1.7793e40000000p-2, 0x1.d5d2bc0000000p-1, 0x1.e280e40000000p-3, 0x1.2e9c680000000p-2, 0x1.03601e0000000p-2, 0x1.03de320000000p-1, 0x1.97e9520000000p-2, 0x1.76843c0000000p-3, 0x1.4045720000000p-2, 0x1.94011e0000000p-1, 0x1.46667c0000000p-1, 0x1.2818fa0000000p-2, 0x1.a808320000000p-2, 0x1.9a89420000000p-1, 0x1.d506e00000000p-3, 0x1.b8cc1e0000000p-2, 0x1.1a304a0000000p+0, 0x1.cdbfc20000000p-3, 0x1.1994840000000p-2, 0x1.02b33e0000000p-1, 0x1.38d9460000000p+0, 0x1.c1247e0000000p-3, 0x1.8e60f20000000p-3, 0x1.a3e1040000000p-3, 0x1.b4e33c0000000p-3, 0x1.749b400000000p-1, 0x1.215b600000000p-3, 0x1.d148d20000000p-2, 0x1.f058460000000p-3, 0x1.7b07780000000p-1, 0x1.05d1000000000p+0, 0x1.e494320000000p-1}
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

typedef float max_pooling1d_3_output_type[POOL_LENGTH][INPUT_CHANNELS];

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
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define NUMBER_T float
#define LONG_NUMBER_T float


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

typedef float flatten_output_type[OUTPUT_DIM];

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

#define NUMBER_T float
#define LONG_NUMBER_T float

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

typedef float dense_output_type[FC_UNITS];

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
#define WEIGHTS_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define NUMBER_T float
#define LONG_NUMBER_T float


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


#error "Data type unsupported by CMSIS-NN"

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


const float dense_bias[FC_UNITS] = {-0x1.7193e40000000p-2, 0x1.1805ea0000000p-5, 0x1.8559b60000000p-7, -0x1.37cce00000000p-2, -0x1.018a080000000p-3, -0x1.2d3eac0000000p-3, 0x1.982b100000000p-7, 0x1.e4c1e60000000p-7, 0x1.b3c36e0000000p-6, -0x1.25db0a0000000p-2, -0x1.54ab180000000p-5, -0x1.16d3760000000p-2, -0x1.3199f20000000p-2, -0x1.9e944a0000000p-3, -0x1.05731e0000000p-3, -0x1.0974a20000000p-3}
;

const float dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{-0x1.5732000000000p-4, -0x1.9197a20000000p-5, -0x1.d78e600000000p-3, -0x1.69fa380000000p-4, -0x1.12258c0000000p-3, 0x1.5804be0000000p-3, -0x1.c55f280000000p-6, 0x1.c08cfc0000000p-4, 0x1.8241dc0000000p-4, -0x1.cde2ca0000000p-4, -0x1.c3e0e80000000p-4, 0x1.b25a5c0000000p-5, 0x1.45de900000000p-4, -0x1.4f9bd20000000p-3, -0x1.02abae0000000p-3, -0x1.213dec0000000p-5, -0x1.71ed460000000p-5, 0x1.bfbfc20000000p-5, 0x1.a300de0000000p-3, -0x1.9a2a9c0000000p-5, -0x1.ccd36a0000000p-4, -0x1.d7d9d40000000p-5, -0x1.9133600000000p-7, -0x1.5575b20000000p-3, -0x1.7b653e0000000p-9, 0x1.2d38900000000p-3, 0x1.adb89a0000000p-6, -0x1.1e214e0000000p-2, -0x1.2c8ce60000000p-4, -0x1.19844e0000000p-2, 0x1.0c5c180000000p-6, 0x1.d50cfe0000000p-10, 0x1.5250740000000p-3, -0x1.37215e0000000p-3, 0x1.54cb1a0000000p-5, -0x1.b0f04a0000000p-4, -0x1.2686420000000p-3, -0x1.0808100000000p-8, 0x1.3f1e200000000p-3, 0x1.90984a0000000p-3, 0x1.9ab1c60000000p-5, -0x1.7238580000000p-4, -0x1.465e7c0000000p-2, -0x1.ffd7320000000p-3, -0x1.569b900000000p-3, -0x1.0745880000000p-4, 0x1.a895f80000000p-3, 0x1.42c4f60000000p-4, -0x1.e6f4e20000000p-3, -0x1.198aa00000000p-2, -0x1.236caa0000000p-4, -0x1.49b0120000000p-4, 0x1.15b5420000000p-3, -0x1.3435de0000000p-2, 0x1.07d6ee0000000p-3, 0x1.2cce940000000p-3, 0x1.36cefc0000000p-2, 0x1.01b4c00000000p-5, -0x1.2c875c0000000p-6, -0x1.4cc7100000000p-3, 0x1.85d9b60000000p-3, 0x1.2a7f6e0000000p-6, -0x1.85d99a0000000p-3, 0x1.9a98fe0000000p-3}
, {0x1.47ac1c0000000p-3, -0x1.061ef80000000p-2, 0x1.6030220000000p-2, 0x1.da66960000000p-6, -0x1.0f8c240000000p-9, -0x1.23686a0000000p-4, 0x1.00eef20000000p-3, 0x1.9440720000000p-3, -0x1.c8fe740000000p-4, 0x1.dd7eb60000000p-8, -0x1.ecad540000000p-3, 0x1.ea70de0000000p-6, -0x1.282c900000000p-2, 0x1.cfd5f00000000p-7, -0x1.60a5340000000p-4, 0x1.8fae260000000p-5, 0x1.45a9920000000p-7, 0x1.453f2a0000000p-5, -0x1.6fccfc0000000p-7, -0x1.72a9fc0000000p-3, -0x1.3e08060000000p-4, -0x1.af40140000000p-4, -0x1.c4a0800000000p-3, -0x1.ea55520000000p-4, -0x1.a40ae60000000p-4, -0x1.ce0f020000000p-4, -0x1.2340260000000p-4, -0x1.bbfe560000000p-3, -0x1.1e3b000000000p-4, 0x1.4c42360000000p-2, -0x1.3388680000000p-3, 0x1.2902020000000p-2, -0x1.ff0f9e0000000p-3, -0x1.9e889c0000000p-6, -0x1.529e080000000p-4, -0x1.9a5f820000000p-2, -0x1.53b9a40000000p-6, -0x1.60cd340000000p-3, -0x1.03d1120000000p-3, -0x1.286d920000000p-5, -0x1.86018e0000000p-5, -0x1.0f77540000000p-4, 0x1.ec11ae0000000p-6, 0x1.254ed40000000p-3, 0x1.4f19f80000000p-4, 0x1.526d520000000p-7, 0x1.aec3a40000000p-3, -0x1.81ef400000000p-4, -0x1.69c0ae0000000p-5, -0x1.0ccf9a0000000p-6, -0x1.2a86940000000p-3, -0x1.643a560000000p-1, -0x1.63a7d40000000p-3, -0x1.c3e84e0000000p-5, -0x1.24ebc00000000p-2, 0x1.18fe420000000p-2, 0x1.4a2ade0000000p-3, -0x1.863b700000000p-4, -0x1.314a140000000p-3, -0x1.d252d40000000p-4, 0x1.5e88d60000000p-4, 0x1.4621220000000p-4, -0x1.afb5a20000000p-2, -0x1.f477a00000000p-6}
, {0x1.05a6f40000000p-9, -0x1.3bd31e0000000p-3, -0x1.1b4f220000000p-6, -0x1.5e40c60000000p-3, -0x1.b8e0900000000p-5, -0x1.f852ce0000000p-3, 0x1.1f1f160000000p-2, -0x1.42a3300000000p-3, -0x1.717b760000000p-2, 0x1.1e88f60000000p-3, 0x1.a4ad5e0000000p-3, 0x1.486cf00000000p-6, -0x1.ae6bee0000000p-2, 0x1.7f97bc0000000p-3, 0x1.16bf200000000p-5, 0x1.1306860000000p-5, -0x1.1b74780000000p-3, 0x1.3ba7b40000000p-4, -0x1.f4cae80000000p-5, 0x1.81a5060000000p-4, -0x1.1518220000000p-2, 0x1.1b27d60000000p-2, 0x1.0836480000000p-4, -0x1.d0d02e0000000p-2, -0x1.79dfce0000000p-3, -0x1.214fb40000000p-4, -0x1.6cc4740000000p-3, 0x1.238df40000000p-2, -0x1.db5d560000000p-6, -0x1.a512080000000p-4, -0x1.1c11b80000000p-4, -0x1.42e3b00000000p-3, -0x1.696f040000000p-2, 0x1.8371800000000p-3, 0x1.b8fc580000000p-4, -0x1.3d07220000000p-2, -0x1.f241080000000p-3, -0x1.cfc4ba0000000p-2, -0x1.dc1f8c0000000p-5, 0x1.5460ee0000000p-3, -0x1.91065a0000000p-3, -0x1.3f44720000000p-5, -0x1.808e780000000p-4, -0x1.d758240000000p-3, 0x1.63cce80000000p-6, -0x1.45303c0000000p-9, -0x1.16fbd20000000p-2, -0x1.85ca680000000p-4, -0x1.8a023e0000000p-4, -0x1.00d98c0000000p-2, -0x1.eb0e680000000p-5, -0x1.e80d440000000p-2, 0x1.4f63000000000p-4, -0x1.ba613a0000000p-4, 0x1.1e54940000000p-6, 0x1.d6dd320000000p-2, 0x1.4af51a0000000p-3, -0x1.60e22c0000000p-2, 0x1.bdaf420000000p-3, -0x1.edbe420000000p-2, -0x1.9129bc0000000p-3, -0x1.177b6e0000000p-6, -0x1.fe915a0000000p-2, -0x1.4a39d60000000p-3}
, {0x1.b6e9e60000000p-6, 0x1.30647a0000000p-4, -0x1.5377140000000p-4, -0x1.042ea20000000p-2, 0x1.182dd00000000p-3, 0x1.08622e0000000p-3, -0x1.16beb40000000p-2, -0x1.a5be160000000p-3, 0x1.0726ac0000000p-4, 0x1.36b8ce0000000p-3, 0x1.0d50e60000000p-3, 0x1.411f880000000p-4, 0x1.5b0eb40000000p-3, -0x1.3c12ea0000000p-3, -0x1.c0f72a0000000p-3, -0x1.16a6080000000p-2, 0x1.3b70a80000000p-4, 0x1.95c47c0000000p-2, -0x1.90465a0000000p-4, -0x1.55501e0000000p-4, -0x1.dc9a6c0000000p-4, -0x1.469d4a0000000p-4, 0x1.8f68b60000000p-3, 0x1.23709e0000000p-3, 0x1.22ce7c0000000p-3, 0x1.1396340000000p-3, -0x1.80b5300000000p-2, 0x1.7b0a540000000p-3, -0x1.a484c00000000p-4, -0x1.2778a80000000p-3, 0x1.5cd59a0000000p-3, -0x1.7254ce0000000p-6, -0x1.c3682e0000000p-4, -0x1.43684e0000000p-2, 0x1.ad174c0000000p-7, 0x1.5bb5a20000000p-2, -0x1.0fe6bc0000000p-2, -0x1.26a1380000000p-3, 0x1.35fb9c0000000p-4, -0x1.ce65a40000000p-11, -0x1.fcc5f60000000p-5, -0x1.e70ece0000000p-3, -0x1.4bbdf00000000p-3, -0x1.e005f00000000p-3, 0x1.4d3f220000000p-4, 0x1.335c1c0000000p-2, 0x1.12f5000000000p-4, -0x1.4a32fe0000000p-4, -0x1.aa91fa0000000p-4, -0x1.d3e9340000000p-3, 0x1.1f09540000000p-2, -0x1.23d9c00000000p-3, -0x1.4b06040000000p-2, 0x1.05ff720000000p-2, 0x1.29956a0000000p-2, 0x1.2f35980000000p-7, -0x1.686fa60000000p-6, -0x1.1adaae0000000p-2, -0x1.6627a40000000p-5, -0x1.b00aca0000000p-3, 0x1.f3e3ce0000000p-3, 0x1.4d6c240000000p-5, -0x1.f552340000000p-5, -0x1.c2459c0000000p-3}
, {0x1.142a400000000p-3, -0x1.7747f40000000p-2, 0x1.22d52c0000000p-3, -0x1.2002620000000p-2, -0x1.2b31da0000000p-2, 0x1.2595ea0000000p-2, 0x1.4a05cc0000000p-2, -0x1.8264400000000p-4, -0x1.47b5be0000000p-3, 0x1.89acac0000000p-5, -0x1.4492f20000000p-2, 0x1.58479e0000000p-3, -0x1.b51c020000000p-2, 0x1.42cb980000000p-3, 0x1.17c1e20000000p-5, 0x1.8538e80000000p-4, -0x1.9f99b00000000p-3, 0x1.3ed2880000000p-3, -0x1.6a969a0000000p-3, 0x1.0d67dc0000000p-3, -0x1.d651bc0000000p-3, 0x1.45e7a80000000p-8, 0x1.2f20200000000p-2, -0x1.b936e80000000p-3, 0x1.2f23440000000p-3, 0x1.eb05940000000p-5, -0x1.1924280000000p-2, 0x1.38e46c0000000p-9, 0x1.1ddf7a0000000p-5, 0x1.7db4f80000000p-2, 0x1.ad51f20000000p-3, -0x1.2891560000000p-2, -0x1.9a5a140000000p-2, 0x1.844f660000000p-4, -0x1.18a76c0000000p-5, 0x1.47863a0000000p-3, -0x1.4d412e0000000p-2, -0x1.bb32920000000p-2, -0x1.5ffaa60000000p-4, 0x1.41f33c0000000p-4, -0x1.113eee0000000p-2, -0x1.2160500000000p-2, -0x1.172f040000000p-3, -0x1.e8c5ca0000000p-5, 0x1.cf37060000000p-4, -0x1.821ba00000000p-4, -0x1.81b0a80000000p-2, 0x1.af5ada0000000p-3, -0x1.f20b8a0000000p-3, 0x1.5f07800000000p-2, 0x1.5532fe0000000p-4, 0x1.e81cca0000000p-2, -0x1.b33ad40000000p-3, 0x1.41adc60000000p-4, -0x1.af449c0000000p-5, 0x1.5a80c20000000p-5, 0x1.32b8da0000000p-4, -0x1.95603e0000000p-4, -0x1.1b43e60000000p-3, 0x1.037b7e0000000p-4, 0x1.fccf7a0000000p-4, 0x1.247af80000000p-4, -0x1.be97bc0000000p-2, -0x1.9944d60000000p-3}
, {-0x1.1899500000000p-2, -0x1.0a0e420000000p-3, 0x1.3e70220000000p-2, 0x1.4cf15a0000000p-4, -0x1.368f7e0000000p-3, 0x1.13d8880000000p-4, 0x1.f521de0000000p-3, 0x1.3a281c0000000p-3, -0x1.4b3ace0000000p-3, 0x1.54b70a0000000p-2, -0x1.a2b2a00000000p-2, 0x1.cf9d260000000p-4, -0x1.4da49a0000000p-2, -0x1.cec21e0000000p-3, -0x1.68c8a20000000p-3, 0x1.88b2800000000p-6, -0x1.d9a3060000000p-8, -0x1.08ad6a0000000p-1, -0x1.9c7d6e0000000p-5, -0x1.8b5fa40000000p-2, 0x1.caeb6c0000000p-4, 0x1.b247f40000000p-4, -0x1.3be83e0000000p-5, -0x1.9e3b740000000p-4, -0x1.99869c0000000p-7, -0x1.0154840000000p-2, 0x1.e2e74e0000000p-3, -0x1.43f0620000000p-2, -0x1.29493c0000000p-4, 0x1.edece40000000p-6, 0x1.9cf7ac0000000p-9, 0x1.3d1f8a0000000p-2, -0x1.358f600000000p-6, -0x1.c12a6c0000000p-4, 0x1.561bd00000000p-4, -0x1.8b89300000000p-7, 0x1.3378ba0000000p-3, -0x1.32c8ae0000000p-2, -0x1.2e50760000000p-3, -0x1.5ad2da0000000p-2, 0x1.0b88d80000000p-4, -0x1.9be52a0000000p-2, -0x1.4e1b0c0000000p-2, 0x1.2094c60000000p-5, -0x1.74a8000000000p-5, 0x1.0766660000000p-5, 0x1.fca7220000000p-5, -0x1.df09f00000000p-4, -0x1.95bca80000000p-6, 0x1.25982e0000000p-2, -0x1.b532fe0000000p-2, -0x1.6343ac0000000p-5, -0x1.4005100000000p-2, 0x1.b9ac080000000p-5, -0x1.4d6a520000000p-8, 0x1.8301060000000p-4, 0x1.8bb70a0000000p-3, 0x1.4ed29a0000000p-2, -0x1.61cf7e0000000p-2, 0x1.ff08c20000000p-7, -0x1.6eaef60000000p-2, -0x1.5deb9e0000000p-4, 0x1.414fb20000000p-7, -0x1.abee000000000p-4}
, {0x1.338ba00000000p-3, 0x1.2ac0660000000p-3, -0x1.5a283c0000000p-3, -0x1.ca1f3e0000000p-3, 0x1.1e9b300000000p-4, 0x1.5653040000000p-7, -0x1.173a900000000p-2, -0x1.1c2b560000000p-2, 0x1.f5278c0000000p-3, 0x1.6e0bac0000000p-8, -0x1.9844800000000p-6, -0x1.af13a80000000p-5, 0x1.69ecd60000000p-5, -0x1.b124540000000p-1, -0x1.525ade0000000p-3, 0x1.533ed00000000p-2, -0x1.15b81c0000000p-4, 0x1.d160c60000000p-8, 0x1.ed49bc0000000p-6, -0x1.2286660000000p-4, -0x1.9cd7e80000000p-4, 0x1.61abc80000000p-5, 0x1.12a5b60000000p-6, 0x1.a516900000000p-3, 0x1.8c5d620000000p-3, -0x1.d2128a0000000p-2, -0x1.b30cae0000000p-3, 0x1.d8bd560000000p-3, 0x1.3d1e4e0000000p-3, 0x1.1749240000000p-8, 0x1.1c3ffa0000000p-2, -0x1.3aea260000000p-7, -0x1.6af1500000000p-4, -0x1.e81e2e0000000p-4, 0x1.8b07ca0000000p-2, 0x1.cf78c20000000p-6, -0x1.6238bc0000000p-3, 0x1.ee9eec0000000p-8, -0x1.38a87c0000000p-1, -0x1.cbc3b20000000p-3, 0x1.ebd4f80000000p-5, -0x1.a6d4640000000p-2, -0x1.ecb9e20000000p-4, 0x1.8535780000000p-4, 0x1.0cb8160000000p-3, 0x1.46a8dc0000000p-3, -0x1.9565c60000000p-4, -0x1.7a2e9e0000000p-4, -0x1.504b180000000p-7, 0x1.1cf49a0000000p-5, 0x1.6e83740000000p-3, -0x1.6d32120000000p-5, -0x1.c1b5780000000p-4, 0x1.fc6c8c0000000p-3, -0x1.69e5040000000p-2, 0x1.970de80000000p-4, 0x1.a3a6760000000p-5, -0x1.3801740000000p-2, -0x1.5365f80000000p-2, -0x1.2debc40000000p-3, 0x1.8aa28e0000000p-4, 0x1.193d760000000p-3, 0x1.caaf060000000p-9, 0x1.c7d7a40000000p-7}
, {0x1.43b4c60000000p-5, -0x1.1d07340000000p-2, -0x1.47a8e60000000p-5, 0x1.131c180000000p-3, -0x1.00f1020000000p-3, 0x1.bc14500000000p-9, -0x1.faeab60000000p-6, -0x1.1f4f140000000p-2, -0x1.ff112e0000000p-7, 0x1.3bb8c00000000p-4, 0x1.ce20fe0000000p-5, 0x1.873dfc0000000p-5, 0x1.b5e9360000000p-3, -0x1.f3831a0000000p-8, 0x1.4050800000000p-6, 0x1.8b6db20000000p-5, -0x1.72bee60000000p-5, 0x1.105c000000000p-5, -0x1.29dd340000000p-2, 0x1.7a9ad40000000p-6, 0x1.2edc5a0000000p-3, -0x1.5a4a840000000p-4, -0x1.66f0060000000p-5, -0x1.63687e0000000p-3, -0x1.55e9be0000000p-5, -0x1.f051000000000p-3, -0x1.09ddf00000000p-2, -0x1.4a48a60000000p-2, 0x1.7f00160000000p-7, -0x1.ec353e0000000p-3, 0x1.679dd00000000p-2, -0x1.5db3920000000p-3, 0x1.96385c0000000p-2, -0x1.b0ed5e0000000p-4, -0x1.a54ffa0000000p-3, 0x1.3468de0000000p-2, -0x1.e3e9340000000p-3, 0x1.46bcb60000000p-3, 0x1.86e45c0000000p-5, 0x1.af12760000000p-5, 0x1.872a300000000p-2, 0x1.96d4740000000p-3, 0x1.46e0620000000p-5, 0x1.961cc20000000p-2, -0x1.7e332e0000000p-3, -0x1.c8d90c0000000p-5, 0x1.170eea0000000p-3, 0x1.a51f600000000p-4, -0x1.8dadc80000000p-3, 0x1.52095e0000000p-2, 0x1.0bcd3c0000000p-4, -0x1.0fe0260000000p-4, -0x1.b38f380000000p-8, 0x1.515c560000000p-2, 0x1.0d5bd20000000p-2, -0x1.2d3a560000000p-1, 0x1.a559720000000p-6, 0x1.3462b60000000p-6, -0x1.a820640000000p-2, -0x1.f537a20000000p-6, 0x1.72737a0000000p-4, 0x1.0728ea0000000p-3, 0x1.0da1ea0000000p-5, 0x1.681d3c0000000p-4}
, {-0x1.adddd60000000p-10, -0x1.d966e00000000p-2, 0x1.e4ff9c0000000p-5, 0x1.3707b80000000p-4, 0x1.3a62ee0000000p-4, -0x1.bad1d00000000p-5, 0x1.00d3740000000p-2, 0x1.b01a0e0000000p-2, -0x1.ba522c0000000p-5, 0x1.d6dcae0000000p-4, -0x1.a602940000000p-3, -0x1.197e1c0000000p-2, -0x1.469f360000000p-3, -0x1.ae62860000000p-4, 0x1.ae3c500000000p-6, -0x1.2ee0180000000p-7, -0x1.88bf340000000p-4, 0x1.76b38a0000000p-4, -0x1.1c2c0c0000000p-2, 0x1.a6ae560000000p-7, 0x1.2e1e880000000p-5, -0x1.7b57dc0000000p-3, -0x1.ed0c2a0000000p-3, 0x1.18758c0000000p-6, -0x1.7673620000000p-3, 0x1.aa6f2a0000000p-3, 0x1.70557e0000000p-2, 0x1.50c7b80000000p-4, 0x1.606d2c0000000p-5, 0x1.44760c0000000p-2, -0x1.7931860000000p-2, 0x1.338f5c0000000p-4, 0x1.3947420000000p-5, 0x1.d7b3f80000000p-4, -0x1.92757e0000000p-2, -0x1.7dc6700000000p-2, 0x1.9d26f00000000p-13, -0x1.29f22c0000000p-2, 0x1.12c4220000000p-4, -0x1.3afb760000000p-5, -0x1.d215940000000p-4, 0x1.e19c140000000p-8, 0x1.c5186a0000000p-5, 0x1.1591be0000000p-2, -0x1.8015260000000p-5, -0x1.f2e59a0000000p-5, -0x1.03fa980000000p-2, 0x1.65b16c0000000p-5, -0x1.996c0e0000000p-3, -0x1.0647580000000p-4, -0x1.8f00aa0000000p-5, -0x1.fcd5360000000p-2, -0x1.56f8940000000p-3, -0x1.a926ba0000000p-3, -0x1.2197a80000000p-2, 0x1.fe0a280000000p-4, 0x1.b2eb4c0000000p-3, 0x1.9326340000000p-4, 0x1.d33bca0000000p-3, 0x1.9b858e0000000p-4, -0x1.5fb3de0000000p-3, -0x1.f00c8e0000000p-4, -0x1.233de40000000p-2, 0x1.0c4ad20000000p-3}
, {-0x1.403a320000000p-4, -0x1.e221680000000p-4, -0x1.6257340000000p-4, 0x1.e659060000000p-4, -0x1.a7f74e0000000p-4, -0x1.179a680000000p-3, 0x1.90aee00000000p-5, -0x1.18b2540000000p-6, -0x1.284ed80000000p-2, 0x1.0e60f60000000p-3, -0x1.a7a67e0000000p-4, -0x1.fea9a80000000p-5, 0x1.5418fa0000000p-3, 0x1.1070160000000p-4, -0x1.cb82ae0000000p-5, -0x1.cafaa40000000p-3, -0x1.7808b20000000p-4, 0x1.3588c80000000p-5, -0x1.b1953e0000000p-5, -0x1.fe499c0000000p-6, -0x1.00882c0000000p-1, 0x1.916ef60000000p-3, 0x1.b09d080000000p-5, -0x1.5eba7e0000000p-5, 0x1.2f88ee0000000p-3, -0x1.86d6c80000000p-3, -0x1.709e0e0000000p-3, -0x1.dcfe400000000p-4, -0x1.3251900000000p-3, -0x1.00fcd00000000p-7, 0x1.06f5ba0000000p-3, 0x1.4b92980000000p-3, 0x1.ccd7f80000000p-4, 0x1.7fb14e0000000p-3, -0x1.d3550a0000000p-4, -0x1.7191b80000000p-5, 0x1.9c37600000000p-4, 0x1.17a2440000000p-2, 0x1.a7ce4e0000000p-5, 0x1.af43800000000p-5, 0x1.dce4ea0000000p-4, 0x1.903fca0000000p-3, -0x1.776a880000000p-3, -0x1.7799e40000000p-2, -0x1.ab327c0000000p-8, 0x1.7dc9bc0000000p-3, 0x1.7e59ce0000000p-4, 0x1.00adca0000000p-8, 0x1.a60d260000000p-4, -0x1.387dfa0000000p-3, 0x1.7d5a7c0000000p-3, 0x1.c9a7f80000000p-10, -0x1.1090ea0000000p-3, 0x1.9cc2640000000p-6, 0x1.2c43da0000000p-4, -0x1.2c148c0000000p-2, 0x1.8c1c9c0000000p-3, 0x1.233be20000000p-3, 0x1.1d95e40000000p-3, 0x1.9ba3040000000p-4, 0x1.21df2e0000000p-3, 0x1.d2b87e0000000p-7, -0x1.042eec0000000p-5, 0x1.e787220000000p-3}
, {-0x1.96de160000000p-5, -0x1.51d6700000000p-3, -0x1.a219020000000p-2, 0x1.e435660000000p-4, -0x1.10a2b00000000p-6, -0x1.080ca80000000p-3, -0x1.867ffe0000000p-2, -0x1.77511a0000000p-5, 0x1.086cfe0000000p-3, -0x1.6384800000000p-4, 0x1.370fbc0000000p-3, 0x1.4ea6fe0000000p-5, 0x1.3db0380000000p-2, 0x1.c3977a0000000p-4, -0x1.838a520000000p-3, -0x1.2c51e00000000p-6, -0x1.b7a1940000000p-4, -0x1.3245cc0000000p-2, -0x1.f783b00000000p-3, -0x1.b9f1600000000p-4, -0x1.73f2680000000p-5, -0x1.118dda0000000p-7, -0x1.b99e7c0000000p-4, 0x1.10588e0000000p-2, -0x1.84b8f60000000p-2, -0x1.f0ea3a0000000p-4, 0x1.85d2900000000p-5, 0x1.abed180000000p-7, -0x1.3dd2320000000p-4, -0x1.05564c0000000p-1, 0x1.e2bc4a0000000p-2, 0x1.1bd5160000000p-4, -0x1.97bbd80000000p-5, -0x1.64b6fe0000000p-4, 0x1.6a2b140000000p-4, 0x1.f6c5bc0000000p-3, -0x1.724c6c0000000p-3, -0x1.c4e16a0000000p-6, 0x1.81b7580000000p-4, 0x1.e029960000000p-4, 0x1.b141300000000p-5, -0x1.b487400000000p-7, -0x1.44d8180000000p-4, 0x1.0161980000000p-4, -0x1.5933340000000p-2, 0x1.1c1b500000000p-9, 0x1.50cae80000000p-2, 0x1.7591b00000000p-7, -0x1.915ce80000000p-4, -0x1.9182ac0000000p-3, 0x1.6e2b6a0000000p-2, -0x1.3adc920000000p-4, 0x1.bf69240000000p-6, 0x1.0c8a2a0000000p-4, -0x1.5bff740000000p-2, -0x1.b1a0500000000p-4, -0x1.99fab40000000p-5, -0x1.7776ac0000000p-5, -0x1.2c13700000000p-2, -0x1.7324780000000p-2, 0x1.2b6bda0000000p-7, 0x1.b87ea80000000p-3, -0x1.834b100000000p-2, -0x1.d7a49e0000000p-5}
, {-0x1.5acf560000000p-6, -0x1.dd0eaa0000000p-4, 0x1.44d8dc0000000p-3, -0x1.79dd780000000p-3, -0x1.162be40000000p-5, 0x1.3056c20000000p-3, 0x1.a6acc20000000p-3, 0x1.096fda0000000p-3, -0x1.05a1220000000p-3, -0x1.9005e00000000p-6, -0x1.ce29ea0000000p-4, 0x1.909fbe0000000p-3, 0x1.97b94c0000000p-5, -0x1.13aba00000000p-7, -0x1.ebac040000000p-5, 0x1.26f2600000000p-4, -0x1.c2a59e0000000p-4, 0x1.48991c0000000p-3, -0x1.fe2e760000000p-4, 0x1.3282ee0000000p-5, -0x1.4f5bc00000000p-2, 0x1.09fa1e0000000p-2, 0x1.cd3fbc0000000p-4, -0x1.8013e60000000p-3, 0x1.16470c0000000p-2, 0x1.4831040000000p-5, 0x1.1076b60000000p-3, -0x1.3de70e0000000p-3, -0x1.c801340000000p-4, 0x1.d513c80000000p-3, -0x1.d7ee1e0000000p-4, 0x1.37259c0000000p-3, 0x1.63cc1a0000000p-4, -0x1.1d102c0000000p-3, 0x1.f290f40000000p-9, 0x1.4cbc2a0000000p-3, -0x1.0fcc320000000p-3, 0x1.0e17260000000p-3, 0x1.398e8c0000000p-4, -0x1.c542ec0000000p-4, 0x1.9730240000000p-5, 0x1.9861660000000p-8, -0x1.66b7ce0000000p-2, -0x1.f202d20000000p-3, -0x1.0754640000000p-6, 0x1.5b23240000000p-4, -0x1.07dcb80000000p-4, 0x1.d1361c0000000p-4, -0x1.90a8560000000p-3, 0x1.ef8ac00000000p-7, 0x1.532a460000000p-3, 0x1.a7313e0000000p-5, 0x1.34bd7c0000000p-4, -0x1.c13e800000000p-6, 0x1.9d6cc00000000p-4, -0x1.1b2b320000000p-2, 0x1.80bbb00000000p-4, 0x1.bd1ca00000000p-4, 0x1.705f680000000p-4, 0x1.f952140000000p-5, 0x1.49d1e80000000p-3, 0x1.9e19240000000p-3, -0x1.6662400000000p-5, -0x1.f0da3e0000000p-5}
, {-0x1.87b81c0000000p-3, -0x1.e2c5600000000p-2, -0x1.03ab920000000p-2, 0x1.2f8b8e0000000p-3, 0x1.c3358a0000000p-6, -0x1.5f8c500000000p-3, -0x1.f8bdaa0000000p-5, -0x1.088fa20000000p-3, -0x1.a441d40000000p-4, -0x1.3caaec0000000p-3, -0x1.7c87520000000p-3, 0x1.d1986a0000000p-3, -0x1.f6de8e0000000p-5, -0x1.a5ef620000000p-3, 0x1.d22e4c0000000p-4, -0x1.a16cf00000000p-5, -0x1.f1cce80000000p-3, 0x1.b159320000000p-3, -0x1.5a06ae0000000p-3, 0x1.0235160000000p-2, 0x1.d2a4b40000000p-7, -0x1.b0f2940000000p-2, -0x1.c3bccc0000000p-4, -0x1.2114580000000p-9, 0x1.fb605c0000000p-3, -0x1.c6b56a0000000p-3, 0x1.6e780a0000000p-3, -0x1.20e0b00000000p-2, -0x1.0596f20000000p-1, -0x1.65f22e0000000p-3, -0x1.f3530e0000000p-6, 0x1.d863ec0000000p-4, 0x1.75ec860000000p-2, 0x1.bdfe420000000p-4, 0x1.8ee3180000000p-6, 0x1.f3660c0000000p-3, -0x1.41ef7a0000000p-4, -0x1.df948e0000000p-3, 0x1.8587a60000000p-2, 0x1.1eacea0000000p-3, 0x1.fcd1e60000000p-3, -0x1.7d69700000000p-3, 0x1.766fba0000000p-4, -0x1.2b8e060000000p-3, -0x1.b6d16a0000000p-2, -0x1.9c92020000000p-3, -0x1.15218a0000000p-2, 0x1.a860ae0000000p-2, -0x1.c372860000000p-3, 0x1.6585100000000p-5, 0x1.278faa0000000p-3, -0x1.97168a0000000p-4, -0x1.da9fd80000000p-4, -0x1.7423080000000p-4, 0x1.b830c60000000p-4, -0x1.1aa6480000000p-2, 0x1.bced3e0000000p-3, -0x1.83b6fc0000000p-4, 0x1.3006900000000p-3, -0x1.a8a2fc0000000p-4, 0x1.72f6620000000p-3, 0x1.1125120000000p-3, -0x1.3e6b3a0000000p-2, 0x1.3906360000000p-2}
, {-0x1.4557de0000000p-1, -0x1.500ecc0000000p-6, -0x1.4b43c60000000p-3, 0x1.c8d47a0000000p-3, -0x1.392a0e0000000p-3, -0x1.0c135e0000000p-3, 0x1.5385ae0000000p-7, 0x1.a01b820000000p-3, -0x1.1323200000000p-2, 0x1.21788c0000000p-8, -0x1.cd99d60000000p-5, 0x1.bf53f40000000p-4, -0x1.253f060000000p-4, -0x1.8365be0000000p-3, -0x1.530b080000000p-3, -0x1.4fb16a0000000p-2, -0x1.00e6960000000p-2, -0x1.4371b60000000p-2, -0x1.caf5580000000p-6, -0x1.78cea40000000p-4, -0x1.3181d40000000p-2, -0x1.3d7d840000000p-4, 0x1.136b5c0000000p-4, -0x1.03d4480000000p-1, 0x1.dc5a720000000p-6, -0x1.2c04b00000000p-4, -0x1.47cc0e0000000p-5, 0x1.fa7b320000000p-8, -0x1.cb8bc40000000p-2, 0x1.2c41de0000000p-5, -0x1.936f140000000p-5, 0x1.87783e0000000p-3, 0x1.b516560000000p-3, -0x1.2e337e0000000p-2, 0x1.61273e0000000p-3, -0x1.03639c0000000p-6, -0x1.a4aa7e0000000p-2, -0x1.2c5a240000000p-3, -0x1.266cf60000000p-8, -0x1.4446ba0000000p-5, 0x1.2ec1f40000000p-4, -0x1.d569300000000p-3, -0x1.19db4a0000000p-3, -0x1.ce208c0000000p-3, -0x1.b43a800000000p-3, 0x1.6afbf60000000p-5, -0x1.a409ba0000000p-4, 0x1.19dd920000000p-5, 0x1.70fce60000000p-5, 0x1.39b9400000000p-9, 0x1.08dc220000000p-3, -0x1.6a79280000000p-3, -0x1.0a21d20000000p-2, 0x1.1c18760000000p-2, 0x1.96b47a0000000p-4, -0x1.b823060000000p-5, -0x1.89ca300000000p-6, 0x1.0a85520000000p-3, 0x1.e52cba0000000p-4, -0x1.38c7680000000p-4, -0x1.d11b760000000p-2, 0x1.5958f00000000p-4, 0x1.cd66c60000000p-5, -0x1.cce7c40000000p-5}
, {-0x1.fb33420000000p-2, 0x1.7b489a0000000p-5, -0x1.e423a60000000p-4, -0x1.9b61c00000000p-2, -0x1.a1c92c0000000p-3, -0x1.28e8be0000000p-3, -0x1.0af66c0000000p-2, 0x1.f93bd40000000p-4, -0x1.1ad9d40000000p-2, -0x1.5ffa720000000p-4, 0x1.e3359e0000000p-4, 0x1.f89b7e0000000p-3, 0x1.8d5a920000000p-4, -0x1.68e89e0000000p-1, 0x1.2b1eca0000000p-3, -0x1.f0003c0000000p-5, -0x1.28bf640000000p-3, -0x1.6be23c0000000p-4, -0x1.9473c60000000p-4, 0x1.3721860000000p-2, -0x1.5d4f6a0000000p-2, -0x1.8093a40000000p-5, -0x1.e0848e0000000p-4, -0x1.fd68520000000p-4, -0x1.3a7a9e0000000p-3, -0x1.c6158e0000000p-5, 0x1.be8aea0000000p-4, 0x1.791bc80000000p-3, -0x1.3725180000000p-2, -0x1.8d83140000000p-3, 0x1.5da5000000000p-3, -0x1.e54b100000000p-6, 0x1.85b3600000000p-5, -0x1.2aa7d60000000p-3, -0x1.5b64200000000p-3, -0x1.d182e20000000p-4, -0x1.4c401c0000000p-2, -0x1.f4e5d80000000p-3, -0x1.4826240000000p-2, -0x1.2455780000000p-2, -0x1.3644e80000000p-2, 0x1.86a0920000000p-4, -0x1.b8b2be0000000p-2, 0x1.2d94960000000p-4, -0x1.d563ce0000000p-4, -0x1.6f29e80000000p-3, 0x1.28ac760000000p-2, -0x1.d8e1560000000p-3, 0x1.4a4fd40000000p-7, 0x1.bf6ad60000000p-4, -0x1.41497c0000000p-3, 0x1.752b340000000p-4, -0x1.e11d780000000p-3, 0x1.9308c40000000p-4, -0x1.692c6c0000000p-9, -0x1.14f4b00000000p-5, -0x1.5338560000000p-4, -0x1.b43c600000000p-3, -0x1.6e80980000000p-4, -0x1.223c680000000p-6, -0x1.888fe00000000p-5, 0x1.e7c6c20000000p-3, -0x1.5692960000000p-9, -0x1.d0ac3a0000000p-5}
, {-0x1.6e3b320000000p-3, 0x1.4ccb040000000p-2, -0x1.01a6a00000000p-4, -0x1.597dd20000000p-4, 0x1.6f7bea0000000p-4, 0x1.fad9ac0000000p-5, -0x1.444ac80000000p-3, -0x1.9108e00000000p-4, -0x1.d27bd20000000p-4, -0x1.3d0cda0000000p-3, 0x1.e2feb00000000p-6, 0x1.76703a0000000p-4, 0x1.8ba6420000000p-7, 0x1.89e8340000000p-5, -0x1.6749100000000p-8, -0x1.1d65ea0000000p-2, 0x1.32f6020000000p-2, -0x1.188dd40000000p-1, -0x1.a4061a0000000p-4, -0x1.719e780000000p-3, -0x1.c4ce200000000p-3, 0x1.9c87100000000p-3, 0x1.1579060000000p-2, -0x1.af951c0000000p-6, -0x1.2696440000000p-2, -0x1.45cccc0000000p-4, -0x1.2e48ce0000000p-6, 0x1.06cde40000000p-2, -0x1.1e0ec80000000p-3, -0x1.1502dc0000000p-2, 0x1.1b83100000000p-2, 0x1.dd6e140000000p-5, -0x1.4389bc0000000p-2, -0x1.671b360000000p-3, -0x1.7a0e5a0000000p-4, 0x1.5796540000000p-4, 0x1.0afd360000000p-3, -0x1.00aa820000000p-2, 0x1.be31f60000000p-6, -0x1.4493160000000p-11, -0x1.0c74200000000p-2, 0x1.a2b3ae0000000p-5, 0x1.18d4340000000p-4, -0x1.bb00a40000000p-3, -0x1.012d2c0000000p-3, -0x1.12672a0000000p-3, 0x1.0f0b900000000p-6, -0x1.780e9c0000000p-2, -0x1.2093680000000p-3, -0x1.95eb080000000p-3, -0x1.fbc1e00000000p-4, -0x1.1c49a60000000p-2, -0x1.5bc70c0000000p-2, 0x1.bb3e540000000p-7, 0x1.4baf980000000p-4, 0x1.9da6460000000p-5, 0x1.80844e0000000p-5, -0x1.429af20000000p-2, 0x1.72e3060000000p-2, 0x1.401d340000000p-5, 0x1.7d50d60000000p-3, 0x1.ac3e500000000p-4, -0x1.c1d3140000000p-2, -0x1.2f4c240000000p-2}
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

typedef float dense_1_output_type[FC_UNITS];

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
#define WEIGHTS_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define NUMBER_T float
#define LONG_NUMBER_T float


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


#error "Data type unsupported by CMSIS-NN"

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


const float dense_1_bias[FC_UNITS] = {0x1.17a63c0000000p-7}
;

const float dense_1_kernel[FC_UNITS][INPUT_SAMPLES] = {{-0x1.540ef00000000p-5, -0x1.c629be0000000p-5, -0x1.8cf1be0000000p-4, 0x1.e9fe980000000p-5, 0x1.14cede0000000p-4, 0x1.08aa5e0000000p-4, 0x1.78ab2e0000000p-5, -0x1.f586d20000000p-6, -0x1.2a0a500000000p-5, 0x1.5660c00000000p-6, -0x1.6dbf9a0000000p-5, -0x1.51e3be0000000p-5, -0x1.21e4240000000p-4, 0x1.1f77920000000p-4, 0x1.0ec4f20000000p-4, -0x1.7229080000000p-4}
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

#define MODEL_INPUT_SCALE_FACTOR 0 // scale factor of InputLayer
#define MODEL_INPUT_NUMBER_T float
#define MODEL_INPUT_LONG_NUMBER_T float

// node 0 is InputLayer so use its output shape as input shape of the model
// typedef  input_t[48][1];
typedef float input_t[48][1];
typedef dense_1_output_type output_t;


void cnn(
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


void cnn(
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
