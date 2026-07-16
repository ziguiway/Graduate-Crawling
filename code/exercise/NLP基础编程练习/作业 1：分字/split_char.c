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
    printf("开始读取文件\n");

    FILE *fp;
    fp = fopen("./data/Sentence.txt", "rb");
    if(fp == NULL)
    {
        printf("Error: File not found.\n");
        return 1;
    }

    // 处理 UTF-8 BOM（EF BB BF）：存在则跳过 3 字节，不存在则退回
    int b1 = fgetc(fp), b2 = fgetc(fp), b3 = fgetc(fp);
    if (!(b1 == 0xEF && b2 == 0xBB && b3 == 0xBF)) {
        ungetc(b3, fp);
        ungetc(b2, fp);
        ungetc(b1, fp);
    }

    int ch;
    unsigned char char_bytes[4];   // 缓存当前字符的所有字节（UTF-8 最多 4 字节）
    int char_len;

    // 输出文件：循环外打开一次，"wb" 覆盖写（每次跑都是干净结果）
    FILE *fp_out = fopen("./data/Sentence_split.txt", "wb");
    if (fp_out == NULL) {
        fprintf(stderr, "Error: cannot open output file.\n");
        fclose(fp);
        return 1;
    }

    while((ch = fgetc(fp)) != EOF)
    {
        // 1. 把首字节转成 2 进制
        int bits[8];
        byte_to_bits(ch, bits);

        // 2. 计算当前字符占几个字节
        char_len = scan(bits);

        // 3. 首字节先存起来
        char_bytes[0] = (unsigned char)ch;

        // 4. 主动读完剩下的 char_len - 1 个 continuation byte
        for (int i = 1; i < char_len; i++) {
            int cb = fgetc(fp);
            if (cb == EOF) break;       // 文件中途结束，保护一下
            char_bytes[i] = (unsigned char)cb;
        }

        // 5. 输出到 stdout（屏幕上看结果）
        for (int i = 0; i < char_len; i++) {
            putchar(char_bytes[i]);
        }
        putchar('*');

        // 6. 写入到文件：字符字节 + 分隔符
        fwrite(char_bytes, sizeof(unsigned char), char_len, fp_out);
        fputc('*', fp_out);
    }

    putchar('\n');
    fclose(fp);
    fclose(fp_out);

    return 0;
}
