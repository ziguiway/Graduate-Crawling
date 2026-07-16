#include <stdio.h>
#include <stdlib.h>

void print_binary(int ch) {
    // 只处理 0~255 的有效字符
    if (ch < 0 || ch > 255) return;
    
    for (int i = 7; i >= 0; i--) {
        putchar((ch & (1 << i)) ? '1' : '0');
    }
}

int main()
{
    FILE *fp;
    fp = fopen("./data/Sentence.txt", "rb");
    if(fp == NULL)
    {
        printf("Error: File not found.\n");
        return 1;
    }

    int ch = fgetc(fp);
    
    while(ch != EOF)
    {
        print_binary(ch);
        printf("\n");
        ch = fgetc(fp);
    }


    fclose(fp);
    return 0;
}
