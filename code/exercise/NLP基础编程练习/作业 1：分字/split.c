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
    for(int i = 0; i < 7; i++)
    {
        if(bits[i] == 1 && bits[i+1] == 1)
        {
            count++;
        }
    }
    return count;
}

int main()
{
    // FILE *fp;
    // fp = fopen("./data/Sentence.txt", "rb");
    // if(fp == NULL)
    // {
    //     printf("Error: File not found.\n");
    //     return 1;
    // }

    // int ch = fgetc(fp);
    
    // while(ch != EOF)
    // {
    //     int bits[8];
    //     byte_to_bits(ch, bits);
    //     for(int i = 0; i < 8; i++)
    //     {
    //         printf("%d", bits[i]);
    //     }
    //     printf("\n");
    //     ch = fgetc(fp);
    // }


    // fclose(fp);

    int a[8] = {1, 0, 1, 1, 0, 1, 1, 0};
    printf("%d", scan(a));
    return 0;
}
