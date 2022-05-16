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

ofstream MyFile;
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



long int print_duration(struct timespec *b, struct timespec *c)
{
	long long r = c->tv_nsec - b->tv_nsec;
        r += ((long long)(c->tv_sec - b->tv_sec) ) * 1000000000;
	// printf("duration = %lld nanoseconds\n", r);
	return r;
}

int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
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


double sigmoid(double x)
{
    //cout<<x<<endl;
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
			H1O[i-3] = sigmoid(H1S[i-3]);
			H1O[i-2] = sigmoid(H1S[i-2]);
			H1O[i-1] = sigmoid(H1S[i-1]);
			H1O[i] = sigmoid(H1S[i]);

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
		H2O[i] = sigmoid(H2S[i]);
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
		OO[i] = sigmoid(OS[i]);
        
	}
}

void print_val(int i){
    forward(&(small_X[i][0]));
    //cout<<"Val_out label= "<<label[0]<<endl;
    for (int i=0;i<N3;i++){
        string str;
        str = "OO[" + to_string(i) + "] = " + to_string(OO[i]) + " Corresponding OS = " + to_string(OS[i])+"\n";
        // str = "OO[" + to_string(i) + "] = " + to_string(OO[i]) + "\n";
        //cout<<OO[i]<<endl;
        MyFile << str;
    }
    string str;
    time_t my_time = time(NULL);
    str = ctime(&my_time);
    MyFile << str;
}





double dE_OO[N3];
double dOO_OS[N3];
double dE_OS[N3];
double dE_B3[N3];
double dE_W2[N2][N3];

double dE_H2O[N2];
double dH2O_H2S[N2];
double dE_H2S[N2];
double dE_B2[N2];
double dE_W1[N1][N2];

double dE_H1O[N1];
double dH1O_H1S[N1];
double dE_H1S[N1];
double dE_B1[N1];
double dE_W0[N0][N1];


double backward(double *O, double *Y)
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
			dE_W2[i][j] = dE_OS[j]*H2O[i];

		//last layer done******************

	// compute dE_H2O
	for (int i=0; i<N2; i++) {
		dE_H2O[i] = 0;
		for (int j = 0; j<N3; j++)
			dE_H2O[i] += dE_OS[j]*W2[i][j];
	}

        // compute dH2O_H2S = H2O dot (1-H2O)
        for (int i=0; i<N2; i++)
        {
        	dH2O_H2S[i] = A*B*(1- ((H2O[i]/A) * (H2O[i]/A)));
        }
		

        // compute dE_H2S = dE_H2O dot dH2O_H2S
        for (int i=0; i<N2; i+=2)
		
		{
			dE_H2S[i] = dE_H2O[i] * dH2O_H2S[i];
			// _mm_storeu_pd(&dE_H2S[i],_mm_mul_pd(_mm_load_pd(&dE_H2O[i]),_mm_load_pd(&dH2O_H2S[i])));	
		}

        // compute dE_B2 = dE_H2S
        for (int i=0; i<N2; i++)
		dE_B2[i] = dE_H2S[i];

        // compute dE_W1
        for (int i=0; i<N1; i++)
		for (int j = 0; j<N2; j++) 
			dE_W1[i][j] = dE_H2S[j]*H1O[i];

		//***************

		//compute dE_H10

			for (int i=0; i<N1; i++) {
		dE_H1O[i] = 0;
		for (int j = 0; j<N2; j++)
			dE_H1O[i] += dE_H2S[j]*W1[i][j];
	}

        // compute dH1O_H1S = H1O dot (1-H1O)
        for (int i=0; i<N1; i++)
		dH1O_H1S[i] = A*B*(1- ((H1O[i]/A) * (H1O[i]/A)));

        // compute dE_H1S = dE_H1O dot dH1O_H1S
        for (int i=0; i<N1; i+=2)
		
		{
			dE_H1S[i] = dE_H1O[i] * dH1O_H1S[i];
			// _mm_storeu_pd(&dE_H1S[i],_mm_mul_pd(_mm_load_pd(&dE_H1O[i]),_mm_load_pd(&dH1O_H1S[i])));
		}

        // compute dE_B1 = dE_H1S
        for (int i=0; i<N1; i++)
		dE_B1[i] = dE_H1S[i];

        // compute dE_W0
        for (int i=0; i<N0; i++)
		for (int j = 0; j<N1; j++) 
			dE_W0[i][j] = dE_H1S[j]*IN[i];
		//******************************
	/*
	cout << "err = " << err << "\n";
	print_1d(IN, N0, "IN");
	print_1d(dE_OO, N2, "dE_OO");
	print_1d(dOO_OS, N2, "dOO_OS");
	print_1d(OO, N2, "OO");
	print_1d(dE_OS, N2, "dE_OS");
        print_1d(dE_B2, N2, "dE_B2");
        print_12(dE_W1, "dE_W1");
        print_1d(dE_B1, N1, "dE_B1");
        print_01(dE_W0, "dE_W0");
	*/

        // update W0, W1, B1, B2;

	for (int i=0; i<N0; i++)
		for (int j=0; j<N1; j++)
			W0[i][j] = W0[i][j] - rate * dE_W0[i][j];

	for (int i=0; i<N1; i++)
		B1[i] = B1[i] - rate * dE_B1[i];


	for (int i=0; i<N1; i++)
		for (int j=0; j<N2; j++)
			W1[i][j] = W1[i][j] - rate * dE_W1[i][j];

	for (int i=0; i<N2; i++)
		B2[i] = B2[i] - rate * dE_B2[i];

	for (int i=0; i<N2; i++)
		for (int j=0; j<N3; j++)
			W2[i][j] = W2[i][j] - rate * dE_W2[i][j];

	for (int i=0; i<N3; i++)
		B3[i] = B3[i] - rate * dE_B3[i];

}  

double X[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
double Y[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}};
//double Y[4][2] = {{0.0, 0.0}, {1.0, 1.0}, {1.0, 1.0}, {0.0, 0.0}};
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
		int ii = i % 24754;
                //int ii= 3;
		forward(&(small_X[ii][0]));
		backward(OO, &(small_y[ii][0]));

		if (i % 10000 == 0) 
			{
                double acc;
                acc = cal_acc();
                MyFile.open("output.txt", std::ios_base::app);
                cout << "Iteration " << i << ": err =" << err << " acc: " << acc << "\n";
                string str;

                str = "At iteration "+ to_string(i)+": err = "+to_string(err)+", Y= "+to_string(label_4[ii])+"\n";
                MyFile << str;
                print_val(ii);
                MyFile.close();
                

            }

		// break;
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


void calc_forward_time()
{
 	
 	long int temp = 0;

	for (int i = 0; i<100; i++)
	{	
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &b);
		forward(&(small_X[0][0]));
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &e);
		// cout << "time for one forward prop: ";
		temp = temp + print_duration(&b, &e);
	}

	temp = temp/100.0;
	cout << "forward duration avg opt: " << temp/1000.0 << endl;

}

void calc_backward_time()
{
 	
 	long int temp = 0;
 	
	for (int i = 0; i<100; i++)
	{	
		forward(&(small_X[i][0]));
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &b);
		backward(OO, &(small_y[i][0]));
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &e);
		temp = temp + print_duration(&b, &e);
	}

	temp = temp/100.0;
	cout << "backward duration avg opt: " << temp/1000.0 << endl;

}

int main(int argc, char *argv[]) 
{
    MyFile.open("output.txt");
    MyFile.close();
    training_image();
    training_label();
    create_small_dataset();
 

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

/*
//show an image and corresponding label

int index = 300;

 	for(int i = 0; i<784; i++){
 		if (i%28== 0 )
 			cout << " " << endl;
 		if (small_X[index][i] != -1)
 			cout << "@";
 		else
 			cout << ".";
 		// cout << small_X[0][i] << " ";
 	}

 	cout << " " << endl;
 	for(int i = 0; i<N3; i++)
 		cout << small_y[index][i];
 	cout << " " << endl;

*/



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



			
    //cout<<sigmoid(-273613);
	// if (argc == 2) train(atoi(argv[1]));
 //        else train(100000000);

		calc_forward_time();
	 // calc_backward_time();
    

}
