/*
 * split_fixed.c
 *
 * 在 split.c 基础上修 3 个 bug：
 *   1. skip_count 语义：scan 返回的是"本字符占几个字节"，首字节本身已读，
 *      所以接下来要跳过的是 scan(bits) - 1 个 continuation byte，不是 scan(bits) 个。
 *      原代码对 ASCII（1 字节）设 skip_count=1，导致下一个字符被直接吞掉。
 *   2. UTF-8 BOM（EF BB BF）处理：开头若存在 BOM，跳过 3 字节。
 *   3. 最终输出：按题目要求"字符间用空格隔开"输出切分结果，调试信息走 stderr。
 *
 * byte_to_bits 和 scan 两个函数保留原实现不变（思路是对的）。
 */

#include <stdio.h>
#include <stdlib.h>

void byte_to_bits(int ch, int bits[8]) {
    if (ch < 0 || ch > 255) return;

    for (int i = 7; i >= 0; i--) {
        bits[7 - i] = (ch >> i) & 1;  // 或 (ch & (1 << i)) ? 1 : 0
    }
}

int scan(int bits[8]) {
    int count = 1;
    int is_continuously= 1;
    for(int i = 0; i < 7; i++)
    {
        // printf("bit %d,%bit %d: %d,%d\n", i, i+1, bits[i], bits[i+1]);
        if(is_continuously == 0)
        {
            break;
        }
        if(bits[i] == 1 && bits[i+1] == 1)
        {
            // printf("bit %d,%bit %d: %d,%d\n", i, i+1, bits[i], bits[i+1]);
            count++;
        }
        else{
            is_continuously = 0;
        }
    }
    return count;
}

int main()
{
    FILE *fp;
    fp = fopen("./data/Sentence.txt", "rb");
    if(fp == NULL)
    {
        fprintf(stderr, "Error: File not found.\n");
        return 1;
    }

    // 处理 UTF-8 BOM（EF BB BF）：存在则跳过 3 字节，不存在则退回这 3 个字节
    int b1 = fgetc(fp), b2 = fgetc(fp), b3 = fgetc(fp);
    if (!(b1 == 0xEF && b2 == 0xBB && b3 == 0xBF)) {
        ungetc(b3, fp);
        ungetc(b2, fp);
        ungetc(b1, fp);
    }

    int ch = fgetc(fp);
    int skip_count = 0;          // 接下来还要跳过几个 continuation byte
    int char_bytes[4];           // 缓存当前字符的所有字节（UTF-8 最多 4 字节）
    int char_len = 0;

    while(ch != EOF)
    {
        if(skip_count > 0)
        {
            // 这是 continuation byte（10xxxxxx），属于当前字符
            char_bytes[char_len++] = ch;
            skip_count--;
            if (skip_count == 0) {
                // 一个完整字符读完了，输出
                for (int i = 0; i < char_len; i++) {
                    putchar(char_bytes[i]);
                }
                putchar(' ');  // 字符间用空格隔开
            }
            ch = fgetc(fp);
            continue;
        }

        // 新字符的首字节
        char_len = 0;
        char_bytes[char_len++] = ch;

        int bits[8];
        byte_to_bits(ch, bits);

        skip_count = scan(bits) - 1;  // ★ 修复 1：首字节已读，还要跳过 total-1 个
        fprintf(stderr, "skip_count: %d\n", skip_count);

        if (skip_count == 0) {
            // ASCII 单字节字符，直接输出
            putchar(ch);
            putchar(' ');
        }

        ch = fgetc(fp);
        fprintf(stderr, "====================================\n");
    }

    putchar('\n');
    fclose(fp);
    return 0;
}
