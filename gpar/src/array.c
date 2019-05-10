/*
Filename array.c
Purpose: Array Analysis modules include sliding slowness beamforming, vespetrum (Seimology)
Author: Guanning Pang
Email: yuupgn@gmail.com
*/
#include <stdio.h>
#include <stdlib.h>
//#define _USE_MATH_DEFINES
#include <math.h>
//#include "platform.h"

#define TSHIFT_TABLE(I,J,K) tshift_table[(I) * grdpts_x*grdpts_y + (J)*grdpts_y + (K)]
#define TIME_TABLE(I, J) time_table[(I) * grdpts + (J)]
#define DATA(I,J) data[(I)*npts + (J)]
#define HIBERT(I,J) hibert[(I)*npts + (J)]
#define SLOW(I,J) slow[(I)*2 + (J)]
//#define ABSPOW(I) abspow[(I)]
#define COHERE(I) cohere[(I)]
#define REAL(I) real[(I)]
#define IMAG(I) imag[(I)]
#define WEIGHT(I) weight[(I)]


#define ABSPOW(I, J) abspow[(I)*grdpts_y+(J)]
#define ENERGY(I, J) energy[(I)*winpts + (J)]
//#define COHERE(I, J, K) cohere[(I)*grdpts_x*grdpts_y+(J)*grdpts_y+(K)]

typedef struct cplxS{
	double re;
	double im;
}cplx;

int do_beam(const cplx *const, const double[],
	int, double,int, 
	double, int, double*);
int do_beam_psw(const cplx *const, const double[],
	int, double,int, 
	double, int, int, double*);
int do_beam_root(const cplx *const, const double[],
    int, double,int, 
    double, int, int, double*);
double get_mag(double*, int, int);
double get_coherence(const cplx *const, const double *const, 
					const double *const,
					int, int, int, int, int, 
					int, int, double, int);

double sign(double);

int slidebeam(const int nstat, const int grdpts_x, const int grdpts_y,
				const int sflag, const int bflag, const int npts, const double delta,
				const double sbeg, const int winpts, const int order,
				const double * const tshift_table, const cplx * const data,
				double *abspow){
	/*
	nstat: number of stations in the array
	grdpts_x: number of kx grid points
	grdpts_y: number of ky grid points
	sflag: flag for amplitude function: 0 for maximum value, 1 for average absolute value
			or 2 for root-mean-square value
	iflag: flag for stacking type: 1 -- linear
								   2 -- 2nd root
	npts: number of points in data
	delta: sample rate
	tshift_table: time shift table for slowness at grid points
	data: 
	*/

	int kx,ky, i; 
	double val;
    double time[nstat];
	double *bdat;
	if((bdat=(double*)malloc(winpts*sizeof(double)))==NULL){
            fprintf(stderr,"Memory Allocation Error\n");
            exit(1);
    }
	for(kx=0;kx<grdpts_x;kx++){
		for(ky=0;ky<grdpts_y;ky++){
			for(i=0;i<nstat;i++){
                time[i] = TSHIFT_TABLE(i, kx, ky);
            }
			switch(bflag){
				case 1:
					if(do_beam(data,time, nstat,sbeg,winpts,delta,npts,bdat)==1){
						fprintf(stderr, "Problem making beam\n");
						exit(1);
					}
					val = get_mag(bdat,winpts,sflag);
					ABSPOW(kx, ky) = val;
					//fprintf(stderr, "power is %f\n", val);
					break;
					case 2:
						if(do_beam_psw(data,time,nstat,sbeg,winpts,delta,npts,order,bdat)==1){
                            fprintf(stderr, "Problem making psw beams\n");
                            exit(1);
                        }
                        val = get_mag(bdat,winpts,sflag);
                        ABSPOW(kx, ky) = val;
                        break;
                    case 3:
                        if(do_beam_root(data,time,nstat,sbeg,winpts,delta,npts,order,bdat)==1){
                            fprintf(stderr, "Problem making psw beams\n");
                            exit(1);
                        }
                        val = get_mag(bdat,winpts,sflag);
                        ABSPOW(kx, ky) = val;
                        break;
				default:
					printf("Bad beam type selected\n");
					exit(1);
			}

		}
	}
    /*free(bdat);*/
    //fprintf(stderr, "done beam\n");
	return(0);
}

int slant(const int nstat, const int grdpts,
          const int bflag, const int npts, const double delta,
          const double sbeg, const int winpts, const int order,
          const double * const time_table, const cplx * const data,
          double *energy){

    int i, j, k;
    double time[nstat];
    double *bdat;
    if((bdat=(double*)malloc(winpts*sizeof(double)))==NULL){
        fprintf(stderr,"Memory Allocation Error\n");
        exit(1);
    }

    for(k=0;k<grdpts;k++){
        for(i=0; i<nstat;i++){
            time[i] = TIME_TABLE(i,k);
        }
        switch(bflag){
            case 1:
                if(do_beam(data,time, nstat,sbeg,winpts,delta,npts,bdat)==1){
                    fprintf(stderr, "Problem making beam\n");
                    exit(1);
                }
                for(j=0;j<winpts;j++){
                    ENERGY(k,j) = bdat[j];
                }
                break;
                case 2:
                    if(do_beam_psw(data,time,nstat,sbeg,winpts,delta,npts,order,bdat)==1){
                        fprintf(stderr, "Problem making psw beams\n");
                        exit(1);
                    }
                    for(j=0;j<winpts;j++){
                        ENERGY(k,j) = bdat[j];
                    }
                    break;
                case 3:
                    if(do_beam_root(data,time,nstat,sbeg,winpts,delta,npts,order,bdat)==1){
                        fprintf(stderr, "Problem making psw beams\n");
                        exit(1);
                    }
                    for(j=0;j<winpts;j++){
                        ENERGY(k,j) = bdat[j];
                    }
                    break;
            default:
                printf("Bad beam type selected\n");
                exit(1);
        }
    }
    return(0);

}

int do_beam(const cplx * const data, const double time[],
			const int nstat, double sbeg, const int winpts,
			double delta, int npts,
			double *stack_amp)
{
	/*
	data: Hibert trasnform of seismic data, a complex designed array total length is nstat*npts
	tshift_table: a table for time shifts for grid search
	nstat: number of stations
	npts: number of data points in each seismic trace
	delta: sample rate of the array
	snpt: start index point for cutting data
	winpts: total points of data that cutting
	stack_amp: stacking data
	*/
        int i,l,sind;
        double tshift,dx;
        double v1;

        for(i=0;i<winpts;i++){
            stack_amp[i]=0.0;
        }

        for(i=0;i<nstat;i++){
        	//tshift = TSHIFT_TABLE(i,kx,ky);
            tshift = time[i];
        	sind = (int)((sbeg + tshift)/delta)+1;
        	dx = tshift - delta*(double)((int)(tshift/delta));
            if( (sind<0) && ((sind+winpts)<npts) ){
                fprintf(stderr,"Shift time is before start of record\n");
                fprintf(stderr,"Padding %d points ...\n",-sind);
                for(l=0;l<-sind;l++){
                    stack_amp[l]+=DATA(i,0).re;
                }
                for(l=-sind;l<winpts;l++){
                    v1 = DATA(i,sind+l).re +
                        (dx/delta)*(DATA(i,sind+l+1).re - DATA(i,sind+l).re);
                    stack_amp[l]+=v1;
                }
            }else if( (sind>=0) && ((sind+winpts)>=npts)){
                fprintf(stderr,"Shift time puts you past end of record\n");
                fprintf(stderr,"Padding %d points ...\n",sind+winpts-npts);
            	for(l=0;l<(npts-sind-1);l++){
            		v1 = DATA(i,sind+l).re +
                        (dx/delta)*(DATA(i,sind+l+1).re - DATA(i,sind+l).re);
            		stack_amp[l]+=v1;
            	}
            	for(l=(npts-sind-1);l<winpts;l++){
            		stack_amp[l]+=DATA(i,npts-1).re;
            		}
            }else if((sind>=0) && ((sind+winpts)<npts)){
            	for(l=0;l<winpts;l++){
            		v1 = DATA(i,sind+l).re + (dx/delta)*(DATA(i,sind+l+1).re-DATA(i,sind+l).re);
            		stack_amp[l]+=v1;
            	}
            }else if((sind<0) && ((sind+winpts)>=npts)){
            	for(l=0;l<-sind;l++){
            		stack_amp[l]+=DATA(i,0).re;
            	}
            	for(l=-sind;l<(npts-sind-1);l++){
            		v1 = DATA(i,sind+l).re + (dx/delta)*(DATA(i,sind+l+1).re-DATA(i,sind+l).re);
            		stack_amp[l]+=v1;
            	}
            	for(l=(npts-sind-1);l<winpts;l++){
            		stack_amp[l]+=DATA(i,npts-1).re;
	        	}
            }
        	
        }
        //fprintf(stderr, "finished sliding\n");
        for(i=0;i<winpts;i++){
        	stack_amp[i]/=(double)nstat;
        }
        return(0);        
}

int do_beam_psw(const cplx * const data, const double time[],
			const int nstat, double sbeg, const int winpts,
			double delta, int npts, const int order,
			double *stack_amp){
    /*
    data: Hibert trasnform of seismic data, a complex designed array total length is nstat*npts
    tshift_table: a table for time shifts for grid search
    nstat: number of stations
    npts: number of data points in each seismic trace
    delta: sample rate of the array
    snpt: start index point for cutting data
    winpts: total points of data that cutting
    stack_amp: stacking data
    */

    int i,l,sind;
    double tshift, *real, *imag, *weight, tmp, v1, v2, dx;

    if((real=(double*)malloc(winpts*sizeof(double)))==NULL){
        fprintf(stderr,"Can't allocate memory\n");
        return(1);
    }
    if((imag=(double*)malloc(winpts*sizeof(double)))==NULL){
        fprintf(stderr,"Can't allocate memory\n");
        return(1);
    }
    if((weight=(double*)malloc(winpts*sizeof(double)))==NULL){
        fprintf(stderr,"Can't allocate memory\n");
        return(1);
    }

    for(i=0;i<winpts;i++){
        stack_amp[i]=0.0;
        real[i]=0.0;
        imag[i]=0.0;
        weight[i]=0.0;
    }
    for(i=0;i<nstat;i++){
        //tshift = TSHIFT_TABLE(i,kx,ky);
        tshift = time[i];
        sind = (int)((sbeg + tshift)/delta)+1;
        dx = tshift - delta*(double)((int)(tshift/delta));
        if( (sind<0) && ((sind+winpts)<npts) ){
            fprintf(stderr,"Shift time is before start of record\n");
            fprintf(stderr,"Padding %d points ...\n",-sind);
            for(l=0;l<-sind;l++){
                stack_amp[l]+=DATA(i,0).re;
                tmp=sqrt(DATA(i,0).re*DATA(i,0).re + DATA(i,0).im*DATA(i,0).im);
                real[l]+=DATA(i,0).re/tmp;
                imag[l]+=DATA(i,0).im/tmp;
            }
            for(l=-sind;l<winpts;l++){
                v1 = DATA(i,sind+l).re +
                    (dx/delta)*(DATA(i,sind+l+1).re - DATA(i,sind+l).re);
                v2 = DATA(i,sind+l).im +
                    (dx/delta)*(DATA(i,sind+l+1).im - DATA(i,sind+l).im);
                stack_amp[l]+=v1;
                tmp = sqrt(v1*v1+v2*v2);
                real[l]+=v1/tmp;
                imag[l]+=v2/tmp;
            }
        }else if( (sind>=0) && ((sind+winpts)>=npts)){
            fprintf(stderr,"Shift time puts you past end of record\n");
            fprintf(stderr,"Padding %d points ...\n",sind+winpts-npts);
            for(l=0;l<(npts-sind-1);l++){
                v1 = DATA(i,sind+l).re +
                        (dx/delta)*(DATA(i,sind+l+1).re - DATA(i,sind+l).re);
                v2 = DATA(i,sind+l).im +
                        (dx/delta)*(DATA(i,sind+l+1).im - DATA(i,sind+l).im);
                stack_amp[l]+=v1;
                tmp = sqrt(v1*v1+v2*v2);
                real[l]+=v1/tmp;
                imag[l]+=v2/tmp;
            }
            for(l=(npts-sind-1);l<winpts;l++){
                stack_amp[l]+=DATA(i,npts-1).re;
                tmp=sqrt(DATA(i,npts-1).re*DATA(i,npts-1).re + DATA(i,npts-1).im*DATA(i,npts-1).im);
                real[l]+=DATA(i,npts-1).re/tmp;
                imag[l]+=DATA(i,npts-1).im/tmp;
            }
        }else if((sind>=0) && ((sind+winpts)<npts)){
            for(l=0;l<winpts;l++){
                v1 = DATA(i,sind+l).re + (dx/delta)*(DATA(i,sind+l+1).re - DATA(i,sind+l).re);
                v2 = DATA(i,sind+l).im + (dx/delta)*(DATA(i,sind+l+1).im - DATA(i,sind+l).im);
                stack_amp[l]+=v1;
                tmp = sqrt(v1*v1+v2*v2);
                real[l]+=v1/tmp;
                imag[l]+=v2/tmp;
            }
        }else if((sind<0) && ((sind+winpts)>=npts)){
            for(l=0;l<-sind;l++){
                stack_amp[l]+=DATA(i,0).re;
                tmp=sqrt(DATA(i,0).re*DATA(i,0).re + DATA(i,0).im*DATA(i,0).im);
                real[l]+=DATA(i,0).re/tmp;
                imag[l]+=DATA(i,0).im/tmp;
            }
            for(l=-sind;l<(npts-sind-1);l++){
                v1 = DATA(i,sind+l).re + (dx/delta)*(DATA(i,sind+l+1).re-DATA(i,sind+l).re);
                v2 = DATA(i,sind+l).im + (dx/delta)*(DATA(i,sind+l+1).im - DATA(i,sind+l).im);
                stack_amp[l]+=v1;
                tmp = sqrt(v1*v1+v2*v2);
                real[l]+=v1/tmp;
                imag[l]+=v2/tmp;
            }
            for(l=(npts-sind-1);l<winpts;l++){
                stack_amp[l]+=DATA(i,npts-1).re;
                tmp=sqrt(DATA(i,npts-1).re*DATA(i,npts-1).re + DATA(i,npts-1).im*DATA(i,npts-1).im);
                real[l]+=DATA(i,npts-1).re/tmp;
                imag[l]+=DATA(i,npts-1).im/tmp;
            }
        }
            
    }

    for(i=0;i<winpts;i++){
        weight[i] = sqrt(real[i]*real[i] + imag[i]*imag[i])/(double)nstat;
        weight[i] = pow(weight[i],(double)order);
        stack_amp[i]*=(weight[i]/(double)nstat);
    }
    free(imag);
    free(real);
    free(weight);
    return(0);        

}

int do_beam_root(const cplx * const data, const double time[],
            const int nstat, double sbeg, const int winpts,
            double delta, int npts, const int order,
            double *stack_amp){
    /*
    data: Hibert trasnform of seismic data, a complex designed array total length is nstat*npts
    tshift_table: a table for time shifts for grid search
    nstat: number of stations
    npts: number of data points in each seismic trace
    delta: sample rate of the array
    snpt: start index point for cutting data
    winpts: total points of data that cutting
    stack_amp: stacking data
    */

    int i,l,sind;
    double tshift, v1, dx;

    for(i=0;i<winpts;i++){
        stack_amp[i]=0.0;
    }
    for(i=0;i<nstat;i++){
        //tshift = TSHIFT_TABLE(i,kx,ky);
        tshift = time[i];
        sind = (int)((sbeg + tshift)/delta)+1;
        dx = tshift - delta*(double)((int)(tshift/delta));
        if( (sind<0) && ((sind+winpts)<npts) ){
            fprintf(stderr,"Shift time is before start of record\n");
            fprintf(stderr,"Padding %d points ...\n",-sind);
            for(l=0;l<-sind;l++){
                stack_amp[l]+=sign(DATA(i,0).re)*pow(fabs(DATA(i,0).re), 1.0/(double)order);
            }
            for(l=-sind;l<winpts;l++){
                v1 = DATA(i,sind+l).re +
                    (dx/delta)*(DATA(i,sind+l+1).re - DATA(i,sind+l).re);
                stack_amp[l]+=sign(v1)*pow(fabs(v1), 1.0/(double)order);
            }
        }else if( (sind>=0) && ((sind+winpts)>=npts)){
            fprintf(stderr,"Shift time puts you past end of record\n");
            fprintf(stderr,"Padding %d points ...\n",sind+winpts-npts);
            for(l=0;l<(npts-sind-1);l++){
                v1 = DATA(i,sind+l).re +
                        (dx/delta)*(DATA(i,sind+l+1).re - DATA(i,sind+l).re);
                stack_amp[l]+=sign(v1)*pow(fabs(v1), 1.0/(double)order);
            }
            for(l=(npts-sind-1);l<winpts;l++){
                stack_amp[l]+=sign(DATA(i,npts-1).re)*pow(fabs(DATA(i,npts-1).re), 1.0/(double)order);
            }
        }else if((sind>=0) && ((sind+winpts)<npts)){
            for(l=0;l<winpts;l++){
                v1 = DATA(i,sind+l).re + (dx/delta)*(DATA(i,sind+l+1).re - DATA(i,sind+l).re);
                stack_amp[l]+=sign(v1)*pow(fabs(v1), 1.0/(double)order);
            }
        }else if((sind<0) && ((sind+winpts)>=npts)){
            for(l=0;l<-sind;l++){
                stack_amp[l]+=sign(DATA(i,0).re)*pow(fabs(DATA(i,0).re), 1.0/(double)order);
            }
            for(l=-sind;l<(npts-sind-1);l++){
                v1 = DATA(i,sind+l).re + (dx/delta)*(DATA(i,sind+l+1).re-DATA(i,sind+l).re);
                stack_amp[l]+=sign(v1)*pow(fabs(v1), 1.0/(double)order);
            }
            for(l=(npts-sind-1);l<winpts;l++){
                stack_amp[l]+=sign(DATA(i,npts-1).re)*pow(fabs(DATA(i,npts-1).re), 1.0/(double)order);
            }
        }
            
    }

    for(i=0;i<winpts;i++){
        stack_amp[i]/=(double)nstat;
        stack_amp[i] = sign(stack_amp[i]) * pow(fabs(stack_amp[i]),(double)order);
    }
    return(0);        

}	

double sign(double x){
    if (x>=0){
        return(1.0);
    } else {
        return(-1.0);
    }
}


double get_mag(double *dat, int npts, int sflag){
	int i;
	double val;

	if(sflag==0){
		val=0.0;
		for(i=0;i<npts;i++){
			if(fabs(dat[i])>val){
				val=fabs(dat[i]);
			}
		}
	}else if(sflag==1){
		val=0.0;
		for(i=0;i<npts;i++){
			val+=fabs(dat[i]);
		}
		val/=(double)npts;
	}else if(sflag==2){
		val=0.0;
		for(i=0;i<npts;i++){
			val+=dat[i]*dat[i];
		}
		val=sqrt(val/(double)npts);
	}else{
		fprintf(stderr, "Bad choice for amplitude metric\n");
		exit(1);
	}
	return(val);
}

double get_coherence(const cplx * const data, const double * const tshift_table,
					const double * const hibert,
				  const int grdpts_x, const int grdpts_y,
				  const int xind, const int yind, const int nstat,
				  const int snpt, const int winpts, const double delta, 
				  const int npts){
	/*
	data: Hibert trasnform of seismic data
	tshift_table: a table for time shifts for grid search
	nstat: number of stations
	npts: number of data points in each seismic trace
	delta: sample rate of the array
	snpt: start index point for cutting data
	winpts: total points of data that cutting  
	*/
	int i,l,sind;
	double tshift,dx, ar, ai;
	double cohere;
    double *real, *imag, *weight, v1, v2, tmp;
    fprintf(stderr, "calculating coherence\n");

    if((real=(double*) calloc(winpts, sizeof(double)))==NULL){
            fprintf(stderr,"Can't allocate memory\n");
            return(1);
        }
    if((imag=(double*) calloc(winpts, sizeof(double)))==NULL){
        fprintf(stderr,"Can't allocate memory\n");
        return(1);
    }
    if((weight=(double*) calloc(winpts, sizeof(double)))==NULL){
        fprintf(stderr,"Can't allocate memory\n");
        return(1);
    }
    for(i=0;i<winpts;i++){
        real[i] = 0.0;
        imag[i] = 0.0;
        weight[i] = 0.0;
    }

    for(i=0;i<nstat;i++){
    	tshift = TSHIFT_TABLE(i,xind,yind);
        sind = snpt + (int)(tshift/delta)+1;
        dx = tshift - delta*(double)((int)(tshift/delta));

        if( (sind<0) && ((sind+winpts)<npts) ){
        	/*fprintf(stderr,"Shift time for station %d is before start of record\n",i);
        	fprintf(stderr,"Padding %d points ...\n",-sind);*/
            for(l=0;l<-sind;l++){
                tmp = sqrt(DATA(i,0).re*DATA(i,0).re+
                    		((DATA(i,0).im))*((DATA(i,0).im)));
                ar = DATA(i,0).re/tmp;
                ai = HIBERT(i,0)/tmp;
                REAL(l)+=ar;
                IMAG(l)+=ai;
            }
            for(l=-sind;l<winpts;l++){
                v1 = DATA(i,sind+l).re +
                        (dx/delta)*(DATA(i,sind+l+1).re - DATA(i,sind+l).re);
                v2 = DATA(i,sind+l).im +
                        (dx/delta)*(DATA(i,sind+l+1).im - DATA(i,sind+l).im);
                tmp = sqrt(v1*v1+v2*v2);
                REAL(l)+=v1/tmp;
                IMAG(l)+=v2/tmp;
            }
        }else if( (sind>=0) && ((sind+winpts)>=npts)){
        	fprintf(stderr,"Shift time for station %d puts you past end of record\n",i);
            fprintf(stderr,"Padding %d points ...\n",sind+winpts-npts);
            for(l=0;l<(npts-sind-1);l++){
            	v1 = DATA(i,sind+l).re +
                        (dx/delta)*(DATA(i,sind+l+1).re - DATA(i,sind+l).re);
                v2 = DATA(i,sind+l).im +
                        (dx/delta)*(DATA(i,sind+l+1).im - DATA(i,sind+l).im);
            	tmp = sqrt(v1*v1+v2*v2);
                REAL(l)+=v1/tmp;
                IMAG(l)+=v2/tmp;
            }
            for(l=(npts-sind-1);l<winpts;l++){
            	tmp = sqrt(DATA(i,npts-1).re*DATA(i,npts-1).re+
            			DATA(i,npts-1).im*DATA(i,npts-1).im);
            	ar = DATA(i,npts-1).re/tmp;
            	ai = HIBERT(i,npts-1)/tmp;
            	REAL(l)+=ar;
            	IMAG(l)+=ai;
            }
        }else if((sind>=0) && ((sind+winpts)<npts)){
            for(l=0;l<winpts;l++){
            	v1 = DATA(i,sind+l).re + (dx/delta)*(DATA(i,sind+l+1).re-DATA(i,sind+l).re);
            	v2 = DATA(i,sind+l).im + (dx/delta)*(DATA(i,sind+l+1).im-DATA(i,sind+l).im);
            	tmp = sqrt(v1*v1+v2*v2);
            	REAL(l)+=v1/tmp;
            	IMAG(l)+=v2/tmp;
            }
        }else if((sind<0) && ((sind+winpts)>=npts)){
        	fprintf(stderr,"Shift time  for station %d is before start of record and puts you past end of record\n",i);
            fprintf(stderr,"Padding %d points ...\n",-sind);
            fprintf(stderr,"Padding %d points ...\n",sind+winpts-npts);
            for(l=0;l<-sind;l++){
            	tmp = sqrt(DATA(i,0).re*DATA(i,0).re+
                    		((DATA(i,0).im))*((DATA(i,0).im)));
                REAL(l)+=DATA(i,0).re/tmp;
                IMAG(l)+=DATA(i,0).im/tmp;
            }
            for(l=-sind;l<(npts-sind-1);l++){
            	v1 = DATA(i,sind+l).re + (dx/delta)*(DATA(i,sind+l+1).re-DATA(i,sind+l).re);
            	v2 = DATA(i,sind+l).im + (dx/delta)*(DATA(i,sind+l+1).im-DATA(i,sind+l).im);
            	tmp = sqrt(v1*v1+v2*v2);
            	REAL(l)+=v1/tmp;
            	IMAG(l)+=v2/tmp;
            }
            for(l=(npts-sind-1);l<winpts;l++){
            	tmp = sqrt(DATA(i,npts-1).re*DATA(i,npts-1).re+
            			DATA(i,npts-1).im*DATA(i,npts-1).im);
            	REAL(l)+=DATA(i,npts-1).re/tmp;
            	IMAG(l)+=DATA(i,npts-1).im/tmp;
            }
        }
        	
    }
    for(i=0;i<winpts;i++){
        WEIGHT(i)=sqrt(REAL(i)*REAL(i)+IMAG(i)*IMAG(i))/(double)nstat;
    }
    cohere=0.0;
    for(i=0;i<winpts;i++){
        cohere+=WEIGHT(i);
    }
    cohere/=(double)winpts;
    fprintf(stderr, "coherence is %f\n", cohere);
    fprintf(stderr, "free dum array\n");
    free(real);
    real=NULL;
    fprintf(stderr, "free image\n");
    free(imag);
    imag=NULL;
    fprintf(stderr, "free weight\n");
    free(weight);
    weight=NULL;
    fprintf(stderr, "done free\n");
    return(cohere);        
}



