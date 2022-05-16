#include <iostream>
#include <math.h> 
#include <stdlib.h>
#include <bits/stdc++.h>
#include <random>
#include <chrono>
#include <ctime>
#include <xmmintrin.h>
#include<sys/time.h>
#include<time.h>
#include <emmintrin.h>

using namespace std;

#define N0  784
#define N1  1000
#define N2  500
#define N3 4
#define A  1.7159
#define B  0.6666


int label_4[24754];
int label_4_pos[24754];

double *IN, *W0, *B1, *H1S, *H1O, *W1, *B2, *H2S, *H2O, *W2, *B3, *OS, *OO;
double *dE_OO, *dOO_OS, *dE_OS, *dE_B3, *dE_W2, *dE_H2O, *dH2O_H2S, *dE_H2S, *dE_B2, *dE_W1, *dE_H1O, *dH1O_H1S, *dE_H1S, *dE_B1, *dE_W0;
double *B1_dev;



double small_X[24754][784];
double small_y[24754][4];
double err;
double rate = 0.005;

int counts[N3];
int n_rows=28;
int n_cols=28;
int  image[60000][28][28];
int  label[60000];
int  predicted_label[60000];

double X_data[60000][784];
double  Y_label[60000][10];

int cnt_Z=0;

int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

double sigmoid(double x)
{
    //cout<<x<<endl;
    double xx = A*tanh(B*x);
	return xx;
}



void training_image(){
    ifstream file ("train-images.idx3-ubyte");
    if (file.is_open()){
        int magic_number=0;
        int number_of_images=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    image[i][r][c] = temp;
                }
            }

        }  
        //display_image_by_id(7000);
    }
    else{
        cout<<"Unable to openfile \n";
        exit(0);
    }
}

void training_label(){
    int number_of_images=0;
    ifstream file ("train-labels.idx1-ubyte");
    if (file.is_open())
    {
        int magic_number=0;        
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        for(int i=0;i<number_of_images;++i)
        {
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            label[i]= temp;
            counts[temp] += 1;  
        }  
        //display_label_by_id(7000);
    }
}


void create_small_dataset(){
	int c = 0;
	for (int i = 0; i<60000; i++){
		if (label[i] < 4){
			label_4[c] = label[i];
			label_4_pos[c] = i;
			c+=1;
		}
	}
}

int counter = 0;

void forward(double *input)
{

        for (int i = 0; i<N0; i++) 
		
		{

			IN[i]   = input[i];
		}


		

        // compute the weighted sum HS in the hidden layer
        for (int i=0; i<N1; i++) 
        {

			H1S[i] = B1[i];
		}


        for (int i=0; i<N1; i++) {
                for (int j=0; j<N0; j++){
                        H1S[i] += IN[j]*W0[N1*j+i];
            //cout<<W0[j][i]<<endl;
            }
        }

//matched so far

 

 //        // Comput the output of the hidden layer, HO[N1];

        for (int i=0; i<N1; i++) 
        {

			H1O[i] = sigmoid(H1S[i]);

		}


	//compute weighted sum of H2
	for (int i=0; i<N2; i++) {
		H2S[i] = B2[i];
        
	}

        for (int i=0; i<N2; i++) {
                for (int j=0; j<N1; j++){
                        H2S[i] += H1O[j]*W1[N2*j+i];
            //cout<<W0[j][i]<<endl;
            }
        }


	//compute the output of H2

	for (int i=0; i<N2; i++) {
		H2O[i] = sigmoid(H2S[i]);
        //cout<<HS[i]<<"   "<<HO[i]<<endl;
	}

        // compute the weighted sum  in the output layer
    for (int i=0; i<N3; i++) {
		OS[i] = B3[i];
	}


    for (int i=0; i<N3; i++) {
		for (int j=0; j<N2; j++)
			OS[i] += H2O[j]*W2[N3*j+i];
	}

//matched********

        // Comput the output of the output layer, OO[N2];

        for (int i=0; i<N3; i++) {
		OO[i] = sigmoid(OS[i]);
        
	}

		cout << "OO: ";
	for(int i = 0; i<4; i++)
		cout << OO[i] << " ";
	cout << endl;

	counter++;
	if (counter>10)
		exit(0);
	// cout << "forward complete" << endl;
	// exit(0);
}


void backward(double *O, double *Y)
{
        // compute error
	err = 0.0;
        for (int i=0; i<N3; i++) 
		err += (O[i] - Y[i])*(O[i]-Y[i]);
	err = err / N3;

        // compute dE_OO
        for (int i=0; i<N3; i++) 
		dE_OO[i] = (O[i] - Y[i])*2.0/N3;

        // compute dOO_OS = OO dot (1-OO)
        for (int i=0; i<N3; i++)
		dOO_OS[i] = A*B*(1- ((OO[i]/A) * (OO[i]/A)));

        // compute dE_OS = dE_OO dot dOO_OS
        for (int i=0; i<N3; i++)
		dE_OS[i] = dE_OO[i] * dOO_OS[i];

        // compute dE_B3 = dE_OS
        for (int i=0; i<N3; i++)
		dE_B3[i] = dE_OS[i];



        // compute dE_W2
        for (int i=0; i<N2; i++)
		for (int j = 0; j<N3; j++) 
			dE_W2[i*N3+j] = dE_OS[j]*H2O[i];


//matched till here


	// 	//last layer done******************

	// compute dE_H2O
	 for (int i=0; i<N2; i++) {
	 	dE_H2O[i] = 0;
	 	for (int j = 0; j<N3; j++)
	 		dE_H2O[i] += dE_OS[j]*W2[i*N3+j];
	}

 //        // compute dH2O_H2S = H2O dot (1-H2O)
        for (int i=0; i<N2; i++)
         {
         	dH2O_H2S[i] = A*B*(1- ((H2O[i]/A) * (H2O[i]/A)));
         }
 
  // compute dE_H2S = dE_H2O dot dH2O_H2S
         for (int i=0; i<N2; i++)
		
 	{
 		 dE_H2S[i] = dE_H2O[i] * dH2O_H2S[i];

	}


 //        // compute dE_B2 = dE_H2S
       for (int i=0; i<N2; i++)
	 	dE_B2[i] = dE_H2S[i];

      // compute dE_W1
	  for (int i=0; i<N1; i++)
	 	for (int j = 0; j<N2; j++) 
	 		dE_W1[i*N2+j] = dE_H2S[j]*H1O[i];



//matched till here

	for (int  i=0; i<N1; i++) {
		dE_H1O[i] = 0;
		for (int j = 0; j<N2; j++)
		{
			dE_H1O[i] += dE_H2S[j]*W1[i*N2+j];
		}

	}

        // compute dH1O_H1S = H1O dot (1-H1O)
        for (int i=0; i<N1; i++)
		dH1O_H1S[i] = A*B*(1- ((H1O[i]/A) * (H1O[i]/A)));


        // compute dE_H1S = dE_H1O dot dH1O_H1S
        for (int i=0; i<N1; i++)
		
		{
			 dE_H1S[i] = dE_H1O[i] * dH1O_H1S[i];
			//_mm_storeu_pd(&dE_H1S[i],_mm_mul_pd(_mm_load_pd(&dE_H1O[i]),_mm_load_pd(&dH1O_H1S[i])));
		}
        // compute dE_B1 = dE_H1S
        for (int i=0; i<N1; i++)
		dE_B1[i] = dE_H1S[i];



        // compute dE_W0
        for (int i=0; i<N0; i++)
		for (int j = 0; j<N1; j++) 
			dE_W0[i*N1+j] = dE_H1S[j]*IN[i];




	// 	//******************************
	// /*
	// cout << "err = " << err << "\n";
	// print_1d(IN, N0, "IN");
	// print_1d(dE_OO, N2, "dE_OO");
	// print_1d(dOO_OS, N2, "dOO_OS");
	// print_1d(OO, N2, "OO");
	// print_1d(dE_OS, N2, "dE_OS");
 //        print_1d(dE_B2, N2, "dE_B2");
 //        print_12(dE_W1, "dE_W1");
 //        print_1d(dE_B1, N1, "dE_B1");
 //        print_01(dE_W0, "dE_W0");
	// */

 //        // update W0, W1, B1, B2;

	// for (int i=0; i<N0; i++)
	// 	for (int j=0; j<N1; j++)
	// 		W0[i][j] = W0[i][j] - rate * dE_W0[i][j];
	for (int i = 0; i<N0*N1; i++)
		W0[i] = W0[i] - rate*dE_W0[i];

	for (int i=0; i<N1; i++)
		B1[i] = B1[i] - rate * dE_B1[i];


	// for (int i=0; i<N1; i++)
	// 	for (int j=0; j<N2; j++)
	// 		W1[i][j] = W1[i][j] - rate * dE_W1[i][j];
	for (int i = 0; i<N1*N2; i++)
		W1[i] = W1[i] - rate*dE_W1[i];

	for (int i=0; i<N2; i++)
		B2[i] = B2[i] - rate * dE_B2[i];

	// for (int i=0; i<N2; i++)
	// 	for (int j=0; j<N3; j++)
	// 		W2[i][j] = W2[i][j] - rate * dE_W2[i][j];
	for (int i = 0; i<N2*N3; i++)
		W0[i] = W0[i] - rate*dE_W0[i];

	for (int i=0; i<N3; i++)
		B3[i] = B3[i] - rate * dE_B3[i];

}


// 	for(int i =0; i<10; i++)
// 	{
// 		cout << B3[i] << " ";
// 	}
// 	cout << " B3" << endl;
// }  

int ii;

double cal_acc()
{
	int c = 0;
	int pred_label[5000];

	for(int i = 0; i<5000; i++)
	{
		double temp_in[N0];
		forward(&(small_X[i][0]));
		double max_temp = OO[0];
		int k = 0;
		for (int j = 1; j<N3; j++){
			if (max_temp<OO[j]){
				max_temp = OO[j];
				k = j;
			}
		}
		// cout << "original " << label[i] << " , predicted: " << k << endl;
		if (label_4[i] == k)
			c+=1;
	}
	cout << "counter: " << c << endl;
	double acc = (c/5000.0) * 100.0;

	return acc;
}

void train(int iter)
{
	for (int i = 0; i< iter; i++) {
		//int ii = random () % 4;
		ii = i % 24754;
                //int ii= 3;
		forward(&(small_X[ii][0]));
		backward(OO, (&small_y[ii][0]));
				if (i % 10000 == 0) 
			{
                double acc;
                acc = cal_acc();
 //                MyFile.open("output.txt", std::ios_base::app);
                cout << "Iteration " << i << ": err =" << err << " acc: " << acc << "\n";
 //                string str;

 //                str = "At iteration "+ to_string(i)+": err = "+to_string(err)+", Y= "+to_string(label_4[ii])+"\n";
 //                MyFile << str;
 //                print_val(ii);
 //                MyFile.close();
                

 //            }

	// 	// break;
	// }


			}

}
}
		



int main()
{
	training_image();
    training_label();
    create_small_dataset();


    IN = (double*)malloc(N0*sizeof(double));
	W0 = (double*)malloc(N0*N1*sizeof(double));
	B1 = (double*)malloc(N1*sizeof(double));
	H1S = (double*)malloc(N1*sizeof(double));
	H1O = (double*)malloc(N1*sizeof(double));

	W1 = (double*)malloc(N1*N2*sizeof(double));
	B2 = (double*)malloc(N2*sizeof(double));
	H2S = (double*)malloc(N2*sizeof(double));
	H2O = (double*)malloc(N2*sizeof(double));

	W2 = (double*)malloc(N2*N3*sizeof(double));
	B3 = (double*)malloc(N3*sizeof(double));
	OS = (double*)malloc(N3*sizeof(double));
	OO = (double*)malloc(N3*sizeof(double));


	dE_OO = (double*)malloc(N3*sizeof(double));
	dOO_OS = (double*)malloc(N3*sizeof(double));
	dE_OS = (double*)malloc(N3*sizeof(double));
	dE_B3 = (double*)malloc(N3*sizeof(double));
	dE_W2 = (double*)malloc(N2*N3*sizeof(double));

	dE_H2O = (double*)malloc(N2*sizeof(double));
	dH2O_H2S = (double*)malloc(N2*sizeof(double));
	dE_H2S = (double*)malloc(N2*sizeof(double));
	dE_B2 = (double*)malloc(N2*sizeof(double));
	dE_W1 = (double*)malloc(N1*N2*sizeof(double));

	dE_H1O = (double*)malloc(N1*sizeof(double));
	dH1O_H1S = (double*)malloc(N1*sizeof(double));
	dE_H1S = (double*)malloc(N1*sizeof(double));
	dE_B1 = (double*)malloc(N1*sizeof(double));
	dE_W0 = (double*)malloc(N0*N1*sizeof(double));





	for (int i=0; i<60000;i++){
	    for (int j=0; j<28;j++){
	        for(int k=0;k<28;k++){
	            X_data[i][k+j*28]=(image[i][j][k]/127.5)-1;
	        }
	    }        
	} 

	for(int i=0; i<60000; i++){
	    for (int j =0; j<10; j++){
	        if (j==label[i]){
	            Y_label[i][j]=1.0;
	        }
	        else{
	            Y_label[i][j]=-1.0;
	        }
	    }             
	}

		for(int i = 0; i<24754; i++){
			int temp = label_4_pos[i];

			for (int j = 0; j<784; j++){
				small_X[i][j] = X_data[temp][j];
			}
		}

		for (int i = 0; i<24754; i++){
			int temp = label_4_pos[i];
			for (int j = 0; j<N3; j++){
				small_y[i][j] = Y_label[temp][j];

			}
		}


	// randomize weights
    for (int i = 0; i<N1; i++)
		B1[i] = random()*1.0/RAND_MAX/100;

  //   for (int i = 0; i<N0; i++)
		// for (int j = 0; j<N1; j++)
		// 	W0[i][j] = random()*1.0/RAND_MAX/100;
	for (int i =0; i<N0*N1; i++)
		W0[i] = random()*1.0/RAND_MAX/100;
    
    for (int i = 0; i<N2; i++)
		B2[i] = random()*1.0/RAND_MAX/100;
    
    for (int i = 0; i<N1*N2; i++)
		W1[i] = random()*1.0/RAND_MAX/100;

	for (int i = 0; i<N3; i++)
		B3[i] = random()*1.0/RAND_MAX/100;
    
    for (int i = 0; i<N2*N3; i++)
		W2[i] = random()*1.0/RAND_MAX/100;


	cudaMalloc( &B1_dev, N1*sizeof(double) );
	cudaMemcpy( B1_dev, B1, N1*sizeof(double), cudaMemcpyHostToDevice );

	// for (int i =(N0-1)*1000; i<N0*1000+10; i++)
	// 	cout << W0[i] << " ";
	// cout << endl;

    train(100000000);





return 0;

}