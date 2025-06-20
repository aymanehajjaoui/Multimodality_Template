# ───────────── Platform detection ─────────────
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

# ───────────── Model selection ─────────────
MODEL ?= Z10

# ───────────── Flags ─────────────
COMMON_FLAGS = -Wall -Wextra -O3 -pedantic -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -mtune=cortex-a9 -D$(MODEL)

ifeq ($(UNAME_M),x86_64)
  COMMON_FLAGS += --sysroot=$(SYSROOT) -I$(SYSROOT)/opt/redpitaya/include
  LDFLAGS += --sysroot=$(SYSROOT) -L$(SYSROOT)/opt/redpitaya/lib
else
  COMMON_FLAGS += -I/opt/redpitaya/include
  LDFLAGS += -L/opt/redpitaya/lib
endif

COMMON_FLAGS += -Imodel1 -Imodel2 -Iinclude -Iinclude/Common -Iinclude/CH1 -Iinclude/CH2 -Iinclude/Hardware -Iinclude/Loggers

CFLAGS   = -std=gnu11 $(COMMON_FLAGS)
CXXFLAGS = -std=c++20 $(COMMON_FLAGS)

LDLIBS  = -lrp -lrp-hw -lrp-i2c -lrp-hw-calib -lrp-hw-profiles -lpthread -lrt -lm -lstdc++

ifeq ($(MODEL),Z20_250_12)
  COMMON_FLAGS += -I$(SYSROOT)/opt/redpitaya/include/api250-12
  LDLIBS += -lrp-gpio
endif

# ───────────── Files & Targets ─────────────
TARGET = main

SRC_DIRS = src/Common src/Hardware src/Loggers src/CH1 src/CH2
SRCS = main.cpp $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)/*.cpp))
OBJS = $(SRCS:.cpp=.o)

# ───────────── Rules ─────────────
all: clean $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) $(LDLIBS) -o $@

%.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

clean:
	rm -f $(TARGET) $(OBJS)
	@if [ -d DataOutput ]; then find DataOutput -type f -delete; fi
	@if [ -d ModelOutput ]; then find ModelOutput -type f -delete; fi

.PHONY: all clean
