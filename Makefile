# ─────────────────────────────
# Platform detection
# ─────────────────────────────
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
HAS_SYSROOT := $(shell [ -d /opt/redpitaya-sysroot ] && echo 1 || echo 0)

ifeq ($(UNAME_S),Linux)
  ifeq ($(UNAME_M),x86_64)
    ifeq ($(HAS_SYSROOT),1)
      SYSROOT ?= /opt/redpitaya-sysroot
      CC := arm-linux-gnueabihf-gcc
      CXX := arm-linux-gnueabihf-g++
    else
      $(error Cannot cross-compile: /opt/redpitaya-sysroot not found on x86_64)
    endif
  else ifeq ($(UNAME_M),armv7l)
    SYSROOT :=
    CC := gcc
    CXX := g++
  else
    $(error Unsupported architecture: $(UNAME_M))
  endif
else
  $(error Unsupported system: $(UNAME_S))
endif

# ─────────────────────────────
# Model selection
# ─────────────────────────────
MODEL ?= Z10

# ─────────────────────────────
# Flags
# ─────────────────────────────
COMMON_FLAGS  = -Wall -Wextra -O3 -pedantic -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -mtune=cortex-a9 -D$(MODEL)

ifeq ($(UNAME_M),x86_64)
  COMMON_FLAGS += --sysroot=$(SYSROOT)
  COMMON_FLAGS += -I$(SYSROOT)/opt/redpitaya/include
  LDFLAGS += --sysroot=$(SYSROOT) -L$(SYSROOT)/opt/redpitaya/lib
else
  COMMON_FLAGS += -I/opt/redpitaya/include
  LDFLAGS += -L/opt/redpitaya/lib
endif

COMMON_FLAGS += \
  -I$(CURDIR)/CMSIS \
  -I$(CURDIR)/CMSIS/Core/Include \
  -I$(CURDIR)/CMSIS/DSP/Include \
  -I$(CURDIR)/CMSIS/NN/Include \
  -I$(CURDIR)/CMSIS/NN/Source/ActivationFunctions \
  -I$(CURDIR)/CMSIS/NN/Source/ConvolutionFunctions \
  -I$(CURDIR)/CMSIS/NN/Source/FullyConnectedFunctions \
  -I$(CURDIR)/model1/include \
  -I$(CURDIR)/model2/include \
  -I$(CURDIR)/include

CFLAGS   = -std=gnu11 $(COMMON_FLAGS)
CXXFLAGS = -std=c++20 $(COMMON_FLAGS)

LDLIBS = -lrp -lrp-i2c -lm -lpthread -lrt -lrp-hw -lrp-hw-calib -lrp-hw-profiles -lstdc++

ifeq ($(MODEL),Z20_250_12)
  COMMON_FLAGS += -I$(SYSROOT)/opt/redpitaya/include/api250-12
  LDLIBS += -lrp-gpio
endif

# ─────────────────────────────
# Files & Targets
# ─────────────────────────────
PRGS = can

MODEL1_C_FILES := $(wildcard model1/*.c)
MODEL2_C_FILES := $(filter-out model2/model.c, $(wildcard model2/*.c))
MODEL2_WEIGHT_C_FILES := $(wildcard model2/weights/*.c)

MODEL1_OBJS := $(MODEL1_C_FILES:.c=.o)
MODEL2_OBJS := $(MODEL2_C_FILES:.c=.o)
MODEL2_WEIGHT_OBJS := $(MODEL2_WEIGHT_C_FILES:.c=.o)

CMSIS_C_FILES := $(shell find CMSIS/NN -type f -name "*.c")
CMSIS_CPP_FILES := $(shell find CMSIS/NN -type f -name "*.cpp")
CMSIS_OBJS := $(CMSIS_C_FILES:.c=.o) $(CMSIS_CPP_FILES:.cpp=.o)

COMMON_CPP_FILES := $(shell find src/Common -name "*.cpp")
CH1_CPP_FILES := $(shell find src/CH1 -name "*.cpp")
CH2_CPP_FILES := $(shell find src/CH2 -name "*.cpp")

COMMON_OBJS := $(COMMON_CPP_FILES:.cpp=.o)
CH1_OBJS := $(CH1_CPP_FILES:.cpp=.o)
CH2_OBJS := $(CH2_CPP_FILES:.cpp=.o)

MAIN_CPP := src/main.cpp
MAIN_OBJ := $(MAIN_CPP:.cpp=.o)

ALL_OBJS := $(MODEL1_OBJS) $(MODEL2_OBJS) $(MODEL2_WEIGHT_OBJS) \
            $(CMSIS_OBJS) $(COMMON_OBJS) $(CH1_OBJS) $(CH2_OBJS) $(MAIN_OBJ)

all: $(PRGS)

$(PRGS): $(ALL_OBJS)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

# Compilation rules
%.o: %.c
	$(CC) -c $< $(CFLAGS) -o $@

%.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

clean:
	find . -name "*.o" -delete
	$(RM) $(PRGS)
	@if [ -d DataOutput ]; then find DataOutput -type f -delete; fi
	@if [ -d ModelOutput ]; then find ModelOutput -type f -delete; fi

.PHONY: all clean
