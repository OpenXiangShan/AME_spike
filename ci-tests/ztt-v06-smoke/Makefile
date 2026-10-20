SPIKE ?= ../../build/spike/spike
CROSS_COMPILE ?= riscv64-linux-gnu-
CC := $(CROSS_COMPILE)gcc

TARGET := ztt-v06-smoke.elf
SRC := ztt-v06-smoke.S
LDS := ztt-v06-smoke.ld

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC) $(LDS)
	$(CC) -march=rv64imac_zicsr -mabi=lp64 -nostdlib -nostartfiles -static \
	  -Wl,-T,$(LDS) -Wl,-e,_start -Wl,--build-id=none \
	  -o $@ $(SRC)

run: $(TARGET)
	$(SPIKE) --isa=rv64imac_ztt $(TARGET)

clean:
	rm -f $(TARGET)
