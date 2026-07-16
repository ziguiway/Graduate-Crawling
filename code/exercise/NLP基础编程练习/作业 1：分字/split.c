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
    
    while(ch != EOF)
    {
        printf("%c", ch);
        ch = fgetc(fp);
    }


    fclose(fp);
    return 0;
}
