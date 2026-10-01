/*
 * Copyright (c) 2026 BOSC & ICT, CAS
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* Minimal Spike HTIF runtime shared by all standalone AME examples. */
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

__attribute__((section(".tohost"), aligned(8)))
volatile uint64_t tohost;
__attribute__((section(".tohost"), aligned(8)))
volatile uint64_t fromhost;

struct ztt_baremetal_file {
    int in_use;
};

static struct ztt_baremetal_file stderr_file = {1};
static struct ztt_baremetal_file files[4];
FILE *stderr = &stderr_file;

static void
spike_putchar(unsigned char value)
{
    while (tohost != 0)
        __asm__ volatile("fence rw, rw" ::: "memory");
    tohost = (UINT64_C(1) << 56) | (UINT64_C(1) << 48) | value;
    __asm__ volatile("fence rw, rw" ::: "memory");
    while (tohost != 0)
        __asm__ volatile("fence rw, rw" ::: "memory");
}

void *
memcpy(void *destination, const void *source, size_t size)
{
    unsigned char *out = destination;
    const unsigned char *in = source;
    for (size_t i = 0; i < size; ++i)
        out[i] = in[i];
    return destination;
}

void *
memset(void *destination, int value, size_t size)
{
    unsigned char *out = destination;
    for (size_t i = 0; i < size; ++i)
        out[i] = (unsigned char)value;
    return destination;
}

int
memcmp(const void *lhs, const void *rhs, size_t size)
{
    const unsigned char *a = lhs;
    const unsigned char *b = rhs;
    for (size_t i = 0; i < size; ++i) {
        if (a[i] != b[i])
            return a[i] < b[i] ? -1 : 1;
    }
    return 0;
}

size_t
strlen(const char *string)
{
    size_t size = 0;
    while (string[size])
        ++size;
    return size;
}

static int
write_stream(FILE *stream, const char *data, size_t size)
{
    if (!stream || !stream->in_use)
        return 0;
    for (size_t index = 0; index < size; ++index)
        spike_putchar((unsigned char)data[index]);
    return (int)size;
}

FILE *
fopen(const char *path, const char *mode)
{
    if (!path || !mode || mode[0] != 'w')
        return NULL;
    for (size_t index = 0; index < sizeof(files) / sizeof(files[0]); ++index) {
        if (!files[index].in_use) {
            files[index].in_use = 1;
            return &files[index];
        }
    }
    return NULL;
}

int
fclose(FILE *stream)
{
    if (!stream || !stream->in_use || stream == stderr)
        return EOF;
    stream->in_use = 0;
    return 0;
}

int
fputc(int character, FILE *stream)
{
    char value = (char)character;
    return write_stream(stream, &value, 1) == 1 ? (unsigned char)value : EOF;
}

struct format_buffer {
    char *data;
    size_t size;
    size_t used;
};

static void
put_char(struct format_buffer *buffer, char value)
{
    if (buffer->used + 1 < buffer->size)
        buffer->data[buffer->used++] = value;
}

static void
put_string(struct format_buffer *buffer, const char *string)
{
    if (!string)
        string = "(null)";
    while (*string)
        put_char(buffer, *string++);
}

static void
put_unsigned(struct format_buffer *buffer, uint64_t value, unsigned base,
             unsigned width, int zero_pad)
{
    char digits[32];
    unsigned count = 0;
    do {
        const unsigned digit = value % base;
        digits[count++] = digit < 10 ? (char)('0' + digit) :
                                      (char)('a' + digit - 10);
        value /= base;
    } while (value && count < sizeof(digits));
    while (count < width) {
        put_char(buffer, zero_pad ? '0' : ' ');
        --width;
    }
    while (count)
        put_char(buffer, digits[--count]);
}

static int
vfprintf_impl(FILE *stream, const char *format, va_list arguments)
{
    char output[512];
    struct format_buffer buffer = {output, sizeof(output), 0};

    while (*format) {
        if (*format != '%') {
            put_char(&buffer, *format++);
            continue;
        }
        ++format;
        if (*format == '%') {
            put_char(&buffer, *format++);
            continue;
        }

        int zero_pad = 0;
        unsigned width = 0;
        if (*format == '0') {
            zero_pad = 1;
            ++format;
        }
        while (*format >= '0' && *format <= '9') {
            width = width * 10 + (unsigned)(*format - '0');
            ++format;
        }

        unsigned length = 0;
        while (*format == 'l') {
            ++length;
            ++format;
        }

        switch (*format++) {
        case 's':
            put_string(&buffer, va_arg(arguments, const char *));
            break;
        case 'c':
            put_char(&buffer, (char)va_arg(arguments, int));
            break;
        case 'd':
        case 'i': {
            int64_t value = length > 1 ? va_arg(arguments, long long) :
                            length == 1 ? va_arg(arguments, long) :
                                          va_arg(arguments, int);
            uint64_t magnitude;
            if (value < 0) {
                put_char(&buffer, '-');
                magnitude = (uint64_t)(-(value + 1)) + 1;
            } else {
                magnitude = (uint64_t)value;
            }
            put_unsigned(&buffer, magnitude, 10, width, zero_pad);
            break;
        }
        case 'u': {
            uint64_t value = length > 1 ?
                va_arg(arguments, unsigned long long) : length == 1 ?
                va_arg(arguments, unsigned long) :
                va_arg(arguments, unsigned int);
            put_unsigned(&buffer, value, 10, width, zero_pad);
            break;
        }
        case 'x': {
            uint64_t value = length > 1 ?
                va_arg(arguments, unsigned long long) : length == 1 ?
                va_arg(arguments, unsigned long) :
                va_arg(arguments, unsigned int);
            put_unsigned(&buffer, value, 16, width, zero_pad);
            break;
        }
        case 'g':
        case 'f':
            (void)va_arg(arguments, double);
            put_string(&buffer, "<fp>");
            break;
        default:
            put_char(&buffer, '?');
            break;
        }
    }
    output[buffer.used] = '\0';
    return write_stream(stream, output, buffer.used);
}

int
fprintf(FILE *stream, const char *format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    int result = vfprintf_impl(stream, format, arguments);
    va_end(arguments);
    return result;
}

static __attribute__((noreturn)) void
baremetal_exit(int status)
{
    while (tohost != 0)
        __asm__ volatile("fence rw, rw" ::: "memory");
    tohost = ((uint64_t)(unsigned)status << 1) | 1;
    __asm__ volatile("fence rw, rw" ::: "memory");
    for (;;)
        __asm__ volatile("wfi");
}

extern int main(int argc, char **argv);

__attribute__((noreturn)) void
ztt_baremetal_start(void)
{
    static char program_name[] = "ame-example";
    static char report_path[] = "spike-report";
    static char *argv[] = {program_name, report_path, NULL};
    baremetal_exit(main(2, argv));
}
