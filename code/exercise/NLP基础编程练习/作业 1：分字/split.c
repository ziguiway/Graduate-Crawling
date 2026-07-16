#include <stdio.h>
#include <stdlib.h>

void byte_to_bits(int ch, int bits[8]) {
    if (ch < 0 || ch > 255) return;
    
    for (int i = 7; i >= 0; i--) {
        bits[7 - i] = (ch >> i) & 1;  // 或 (ch & (1 << i)) ? 1 : 0
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
        int bits[8];
        byte_to_bits(ch, bits);
        for(int i = 0; i < 8; i++)
        {
            printf("%d", bits[i]);
        }
        printf("\n");
        ch = fgetc(fp);
    }


    fclose(fp);
    return 0;
}
