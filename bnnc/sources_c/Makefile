# Default Makefile

# Compiler definitions
CC := gcc
DUMP := objdump

CFLAGS = -O3 -I.

CSRCS := $(shell find -name '*.c')
OBJS := $(CSRCS:%.c=%.o) $(ASRCS:%.S=%.o)

main: main.elf
	@./$<

main.elf : $(OBJS)
	@$(CC) $(CFLAGS) $^ -o $@

%.o: %.c
	@$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) main.elf
