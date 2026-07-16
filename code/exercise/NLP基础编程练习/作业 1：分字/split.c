#include <stdio.h>
#include <stdlib.h>

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
    printf("%c", ch);


    fclose(fp);
    return 0;
}
