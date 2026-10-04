#Simulate a single Artificial Neuron
import math
x1 = 2
x2 = 3

# Weights
w1 = 0.4
w2 = 0.6

bias = 0.5

weighted_sum = (x1*w1)+(x2*w2)+ bias

output = 1/(1+math.exp(-weighted_sum))

print("Weighted Sum:",weighted_sum)
print("Final OutPut:",output)

if output >= 0.5:
  print("Out Put is Colse to 1")
else:
  print("OutPut is close to 0")


print("...........................................................")

#Q2 
import numpy as np
import matplotlib.pylab as plt

x = np.linspace(-10,10,100)

def sigmoid(x):
  return 1 /(1+np.exp(-x))
def relu(x):
  return np.maximum(0,x)
def tanh(x):
  return np.tanh(x)

y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)

#sigmoid
plt.figure()
plt.plot(x,y_sigmoid)
plt.title("Sigmoid Activation Function")

plt.xlabel("Input")
plt.ylabel("Output")

plt.grid()
plt.show()

# ReLU
plt.figure()
plt.plot(x,y_relu)
plt.title("Relu Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()

# Tanh
plt.figure()
plt.plot(x,y_tanh)
plt.title("Tanh Activation Function")
plt.xlabel("Input")
plt.ylabel("OutPut")
plt.grid()
plt.show()

#Q3

# MSE
# BCE
# Actual values
# Predicated values

print("-"*40)
import math
actual = [1,0,1,1]
predicted = [0.9,0.2,0.8,0.7]

mse = 0

for i in range(len(actual)):
  error = actual[i]=predicted[i]
  mse = mse + error ** 2
mse = mse /len(actual)
print("Mean Squared Error:",mse)

# Binary Cross Entropy

bce = 0
for i in range(len(actual)):
  y = actual[i]
  p = predicted[i]
  bce = bce + (y * math.log(p)+(1-y)*math.log(1-p))
  bce = -bce/len(actual)

print("Binary Cross Entropy:",bce)




#Q3

# MSE
# BCE
# Actual values
# Predicated values

print("-"*40)
import math
actual = [1,0,1,1]
predicted = [0.9,0.2,0.8,0.7]

mse = 0

for i in range(len(actual)):
  error = actual[i]=predicted[i]
  mse = mse + error ** 2
mse = mse /len(actual)
print("Mean Squared Error:",mse)

# Binary Cross Entropy

bce = 0
for i in range(len(actual)):
  y = actual[i]
  p = predicted[i]
  bce = bce + (y * math.log(p)+(1-y)*math.log(1-p))
  bce = -bce/len(actual)

print("Binary Cross Entropy:",bce)


# Show How Weights Are Updated In ANN

# Input 
# Weights
# Bias
# Target
# Learing rate
# Predication
# Error 
# Updated Weight

print("-"*50)

x = 2
weight = 0.5
bias = 0.1
target = 1
learining_rate = 0.1

prediction = (x*weight)+bias
print("Old Weight:",weight)
print("Predication:",prediction)

error = target - prediction
print("Error :",error)

weight_update = learining_rate *error*x
new_weight = weight+weight_update

# Updated  bias
new_bias = bias + (learining_rate*error)
print("Updated Weight:",new_weight)
print("Updated Bias:",new_bias)
