from PIL import Image
from pylab import *
import numpy as np
from scipy import misc
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import fmin_cg
import scipy.optimize
import random
import matplotlib.cm as cm
#define constants, imsize is the size of one size of the desired image (i.e. imsize=100 gives images of 100x100 pixels)
imsize=100
#layers sizes are the number of neurons in each layer of the network, num_labels it the output layer of the network
input_layer_size  = imsize**2
hidden_layer_size = 1
num_labels = 2
#how many sample images to use for each class of the training set
numimages=10
#lamb is learning rate
lamb=1

#variables to hold images of each class, JA=jennifer aniston, CD=cameron diaz
JA=np.zeros((numimages,imsize*imsize))
CD=np.zeros((numimages,imsize*imsize))

#gadient descent algorithm
def graddesc(weights,insize,hidsize,outsize,x,y,lamb,numiter):
	weights=np.ndarray.flatten(weights)
	grad=NNgrad(weights,insize,hidsize,outsize,x,y,lamb)
	for i in range(0,numiter):
		grad=NNgrad(weights,insize,hidsize,outsize,x,y,lamb)
		weights=weights-grad
		NNcost(weights,insize,hidsize,outsize,x,y,lamb)
		
	
	return weights


#calculate cost function of parameters
def NNcost(weights,insize,hidsize,outsize,x,y,lamb):
	#theta1=weights[0:hidsize*(insize+1),0]
	theta1=weights[0:hidsize*(insize+1)]
	theta1.shape=((hidsize,(insize+1)))
	#theta2=weights[hidsize*(insize+1):len(weights),0]
	theta2=weights[hidsize*(insize+1):]
	theta2.shape=((outsize,(hidsize+1)))
	Theta1_grad=np.zeros((len(theta1),len(theta1[0])))
	Theta2_grad=np.zeros((len(theta2),len(theta2[0])))
	m=float(len(x))
	yy=np.zeros((len(y),outsize))
	#print('size of yy is:'+str(shape(yy)))
	y=y.astype(int)

	for i in range(0,len(y)):
		yy[i,y[i]-1]=1
		
	x=np.column_stack((np.ones((m,1)),x))
	z2=dot(x,theta1.T)
	a2=sigmoid(z2)
	n=len(a2)
	a2=np.column_stack((np.ones((n,1)),a2))
	z3=dot(a2,theta2.T)
	a3=sigmoid(z3)
	tempvar=-yy*log(a3)-((1-yy)*log(1-a3))
	#regularizing variable for cost
	costreg=(lamb/(2*m))*(sum(sum(theta1[:,2:]**2))+sum(sum(theta2[:,2:]**2)))
	J=(1/m)*sum(sum(tempvar.T))+costreg

	print('current J:'+str(J))
	
	return J
#calculate gradient of parameters	
def NNgrad(weights,insize,hidsize,outsize,x,y,lamb):
	theta1=weights[0:hidsize*(insize+1)]
	theta1.shape=((hidsize,(insize+1)))
	theta2=weights[hidsize*(insize+1):]
	theta2.shape=((outsize,(hidsize+1)))
	Theta1_grad=np.zeros((len(Theta1),len(theta1[0])))
	Theta2_grad=np.zeros((len(Theta2),len(theta2[0])))
	m=float(len(x))
	yy=np.zeros((len(y),outsize))
	#print('size of yy is:'+str(shape(yy)))
	y=y.astype(int)

	for i in range(0,len(y)):
		yy[i,y[i]-1]=1
		

		
	x=np.column_stack((np.ones((m,1)),x))
	z2=dot(x,theta1.T)
	a2=sigmoid(z2)
	n=len(a2)
	a2=np.column_stack((np.ones((n,1)),a2))
	z3=dot(a2,theta2.T)
	a3=sigmoid(z3)
	costreg=(lamb/(2*m))*(sum(sum(theta1[:,2:len(Theta1[0])]**2))+sum(sum(theta2[:,2:len(theta2[0])]**2)))
	#print('J:'+str(J))
	#print('m:'+str(type(m)))
	for t in range(0,int(m)):
		d3=a3[t,]-yy[t,]
		d2=dot(d3,theta2[:,1:])*sigmoidGradient(z2[t,])
		d3=np.matrix(d3)
		d2=np.matrix(d2)
		#print('size of d3:'+str(shape(d3))+'size of a2:'+str(shape(np.matrix(a2[t,]))))
		Theta2_grad=Theta2_grad+dot(d3.T,np.matrix(a2[t,:]))
		Theta1_grad=Theta1_grad+dot(d2.T,np.matrix(x[t,:]))

		

	regtheta1=np.zeros(shape(theta1))
	regtheta2=np.zeros(shape(theta2))
	regtheta1[:,1:]=theta1[:,1:]
	regtheta2[:,1:]=theta2[:,1:]	
	Theta1_grad=(1/m)*Theta1_grad+(lamb/m)*regtheta1
	Theta2_grad=(1/m)*Theta2_grad+(lamb/m)*regtheta2
	grad=np.zeros((size(theta1)+size(theta2),))
	grad[0:(size(theta1)),]=np.ravel(Theta1_grad)
	grad[size(theta1):,]=np.ravel(Theta2_grad)
	return grad
	
	
		
#generate matrix of random connection weights between neurons
def randweights(Lin,Lout):
	einit=0.12
	W=(np.random.random(size=(Lout,(1+Lin)))*2*einit)-einit
	return W
	
#hypothesis is a sigmoid function	
def sigmoid(z):
	g=1/(1+np.exp(-z))
	return g
#gradient of the sigmoid function
def sigmoidGradient(z):
	g=sigmoid(z)*(1-sigmoid(z))
	return g
	
def rgb2gray(rgb):

    
    rgb=double(rgb)/255
    r=rgb[:,0]
    g=rgb[:,1]
    b=rgb[:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
		
		
#load training data
for i in range(0,numimages):
	


	im=Image.open('/home/ben/Machine_Learning_coursera/facialrecognition/camerondiaz_images/'+str(i+1)+'.jpg')
	im=im.resize((imsize,imsize),Image.NEAREST)
	im=np.asarray(im.getdata())
	imgray=rgb2gray(im)
	CD[i,]=imgray
	im=Image.open('/home/ben/Machine_Learning_coursera/facialrecognition/jenniferaniston_images/'+str(i+1)+'.jpg')
	im=im.resize((imsize,imsize),Image.NEAREST)
	im=np.asarray(im.getdata())
	imgray=rgb2gray(im)
	JA[i,]=imgray
#put training data into a single variable, x
x=np.zeros((len(CD)+len(JA),imsize*imsize))
x[0:len(CD),:]=CD
x[len(CD):,:]=JA
y=np.zeros((len(CD)+len(JA),1))
#correct answers for training set...1 is for cameron diaz, 2 for jennifer aniston
y[0:len(CD),]=int(1)
y[len(CD):,]=int(2)

#generate random connection matrix
Theta1=randweights(input_layer_size,hidden_layer_size)
Theta2=randweights(hidden_layer_size,num_labels)

#nnparams is a vector that unpacks all of the weight matrix values into one variable
nnparams=np.zeros((Theta1.size+Theta2.size,1))
nnparams[0:size(Theta1),0]=np.reshape(Theta1,size(Theta1),1)
nnparams[size(Theta1):,0]=np.reshape(Theta2,size(Theta2),1)

#these handles are created so it is easy to pass it to the optimization function leaving nnparams to be optimized while holding all other
#values constant
handlecost= lambda p: NNcost(p,input_layer_size,hidden_layer_size,num_labels,x,y,lamb)
handlegrad= lambda q: NNgrad(q,input_layer_size,hidden_layer_size,num_labels,x,y,lamb)


#optimize the parameters using the conjugate gradient algorithm
nnparamsnew=optimize.fmin_cg(handlecost,nnparams,fprime=handlegrad,maxiter=int(100),disp=True)

#optimize the parameters using the gradient descent algorithm
#nnparamsnew=graddesc(nnparams,input_layer_size,hidden_layer_size,num_labels,x,y,lamb,numiter=100)
#theta1=weights[0:hidsize*(insize+1),0]
theta1=nnparamsnew[0:hidden_layer_size*(input_layer_size+1)]
theta1.shape=((hidden_layer_size,(input_layer_size+1)))
#theta2=weights[hidsize*(insize+1):len(weights),0]
theta2=nnparamsnew[hidden_layer_size*(input_layer_size+1):]
theta2.shape=((num_labels,(hidden_layer_size+1)))
m=int(len(x))
x=np.column_stack((np.ones((m,1)),x))
z2=dot(x,theta1.T)
a2=sigmoid(z2)
n=int(len(a2))
a2=np.column_stack((np.ones((n,1)),a2))
z3=dot(a2,theta2.T)
a3=sigmoid(z3)
print('a3:'+str(a3))

for i in range(0,numimages*2):
	img=x[i,1:]
	img.shape=((imsize,imsize))
	plt.imshow(img,cmap = cm.Greys_r)
	index=np.argmax(a3[i,:])
	if(index==0):
		ttl='cameron diaz'
	elif(index==1):
		ttl='jennifer aniston'
		
	title(ttl)
	show()


