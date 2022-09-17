
#include <math.h> 
#include <stdlib.h>
#include <stdio.h>


struct timespec b, e;
int label_4[24754];
int label_4_pos[24754];

double small_X[24754][784];
double small_y[24754][4];

#define N0  784
#define N1  1000
#define N2  500
#define N3 4
#define A  1.7159
#define B  0.6666

double IN[N0];
double W0[N0][N1];
double B1[N1];
double H1S[N1];
double H1O[N1];

double W1[N1][N2];
double B2[N2];
double H2S[N2];
double H2O[N2];


double W2[N2][N3];
double B3[N3];
double OS[N3];
double OO[N3];













double activation(double x)
{

    double xx = A*tanh(B*x);
	return xx;
}


void forward(double *input)
{

        for (int i = 3; i<N0; i+=4) 
		
		{
			IN[i-3] = input[i-3];
			IN[i-2] = input[i-2];
			IN[i-1] = input[i-1];
			IN[i]   = input[i];
		}


        // compute the weighted sum HS in the hidden layer
        for (int i=3; i<N1; i+=4) 
        {
			H1S[i-3] = B1[i-3];
			H1S[i-2] = B1[i-2];
			H1S[i-1] = B1[i-1];
			H1S[i] = B1[i];
		}



        for (int j=0; j<N0; j++) 
        {
        	double temp = IN[j];

			for (int i=0; i<N1; i++)
			{
				H1S[i] += temp*W0[j][i];

          	}
		}



        // Comput the output of the hidden layer, HO[N1];

        for (int i=3; i<N1; i+=4) 
        {
			H1O[i-3] = activation(H1S[i-3]);
			H1O[i-2] = activation(H1S[i-2]);
			H1O[i-1] = activation(H1S[i-1]);
			H1O[i] = activation(H1S[i]);

		}


	//compute weighted sum of H2
	for (int i=0; i<N2; i++) {
		H2S[i] = B2[i];
        
	}

	for (int j=0; j<N1; j++) 
	{
		double temp  = H1O[j];
		for (int i=0; i<N2; i++)
		{
			H2S[i] += temp*W1[j][i];

        }
	}

	//compute the output of H2

	for (int i=0; i<N2; i++) {
		H2O[i] = activation(H2S[i]);
        //cout<<HS[i]<<"   "<<HO[i]<<endl;
	}

        // compute the weighted sum  in the output layer
        for (int i=0; i<N3; i++) {
		OS[i] = B3[i];
	}
        for (int i=0; i<N3; i++) {
		for (int j=0; j<N2; j++)
			OS[i] += H2O[j]*W2[j][i];
	}

        // Comput the output of the output layer, OO[N2];

        for (int i=0; i<N3; i++) {
		OO[i] = activation(OS[i]);
        
	}
}



	
void train(int iter)
{
	printf("-----strating training-----\n");
	for (int i = 0; i< iter; i++) 
	{
		int ii = i % 24754;
		printf("now running sample number: %d\n", ii);
		forward(&(small_X[ii][0]));
	}
}



int main(int argc, char *argv[]) 
{

	int pos;
 

//Create Dummy Dataset


	for(int i = 0; i<24754; i++)
	{
		for (int j = 0; j<784; j++)
		{
			small_X[i][j] = random()*1.0/RAND_MAX;
		}

		pos = rand()%4;
		printf("%d\n", pos);
		for (int k = 0; k<4; k++)
		{
			if (k == pos){
				small_y[i][k] = 1;
			}
			else
				small_y[i][k] = 0;

		}
	}



// randomize weights
    for (int i = 0; i<N1; i++)
		B1[i] = random()*1.0/RAND_MAX/100;
    for (int i = 0; i<N0; i++)
		for (int j = 0; j<N1; j++)
			W0[i][j] = random()*1.0/RAND_MAX/100;
    
    for (int i = 0; i<N2; i++)
		B2[i] = random()*1.0/RAND_MAX/100;
    
    for (int i = 0; i<N1; i++)
		for (int j = 0; j<N2; j++)
			W1[i][j] = random()*1.0/RAND_MAX/100;

	for (int i = 0; i<N3; i++)
		B3[i] = random()*1.0/RAND_MAX/100;
    
    for (int i = 0; i<N2; i++)
		for (int j = 0; j<N3; j++)
			W2[i][j] = random()*1.0/RAND_MAX/100;



	if (argc == 2) train(atoi(argv[1]));
        else train(100000000);


    

}
