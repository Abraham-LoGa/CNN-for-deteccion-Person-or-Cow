"""
Nombre: López García Abraham Grado_7 Grupo:5

"""
  # Importamos librerías
import cv2
import os
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt

  # Función para el redimensionamiento y la obtención de valores de imágenes 
def make_data(path,h,w):
	data_set = []  # Matriz donde se guardaran los valores de imágenes
	label = ["Faces","Vacas"]  # Nombre de las carpetas
	  # Ciclo para la extracción de información de cada carpeta
	for form in label:
		c_num = label.index(form)  # Nombre de cada archivo
		d_path = os.path.join(path,form)  # Obtención de cada imagen de la carpeta
		  # Ciclo para cada una de las imágenes dentro de la carpeta
		for file in os.listdir(d_path):
			img = cv2.imread(os.path.join(d_path,file), cv2.IMREAD_GRAYSCALE)  # Lectura y conversión a grises de cada imagen
			img = cv2.resize(img,(h,w))  # Redimensionamiento de imágenes
			data_set.append([img,c_num]) # Se añade cada imagen e índice a la matriz vacía
	  # Declaración de nuevas matrices
	X=[] 
	Y=[]
	  # Ciclo para guardar los valores obtenidos anteriormente
	for info, label in data_set:
		  # Se añaden los valores
		X.append(info)
		Y.append(label)
	  # Se guardan los valores
	X = np.array(X).reshape(-1,h,w)
	Y = np.array(Y)
	return X,Y

  # To training
A,B = make_data("Trainig",100,100)
  # To test
C,D = make_data("Test",100,100)

  # Primera capa de convolución 
class CNN:
	
	def __init__(self, num_filters,filter_size):
		  # Se inicializan los parámetros para la convulución y creación de métodos
		self.num_filters = num_filters
		self.filter_size = filter_size
		self.conv_filter=np.random.randn(num_filters,filter_size,filter_size)/(filter_size*filter_size)

	  # Función para la región de imagen a convolucionar
	def image_region(self, image):
		h = image.shape[0]
		w = image.shape[1]

		self.image = image
		for i in range(h-self.filter_size + 1):
			for j in range(w - self.filter_size + 1):
				image_patch = image[i : (i + self.filter_size), j : (j + self.filter_size)]
				yield image_patch, i, j 
	  # Propagación hacia adelante 
	def forward_prop(self, image):
		h,w = image.shape
		a = h - self.filter_size + 1
		b = w - self.filter_size + 1
		conv_out = np.zeros((a,b,self.num_filters))
		for image_patch,i,j in self.image_region(image):
			conv_out[i,j]=np.sum(image_patch*self.conv_filter,axis= (1,2))
		return conv_out 

	  # Propagación en retroceso dL_out es el gradiente de pérdida para las salidadeas de la capa
	def back_prop(self, dL_dout, learning_rate):
		dL_dF_params = np.zeros(self.conv_filter.shape)
		for image_patch, i, j in self.image_region(self.image):
			for k in range(self.num_filters):
				dL_dF_params[k] += image_patch*dL_dout[i,j,k]

		  # Actualización de filtros
		self.conv_filter -= learning_rate*dL_dF_params
		return dL_dF_params

  # Función para la agrupación máxima con un tammaño definido
class Max_Pool:

	def __init__(self, filter_size):
		self.filter_size = filter_size 
	
	  # Función para generar regiones de imágenes no superpuestas para la agrupación
	def image_region(self, image):
		new_h = image.shape[0]//self.filter_size
		new_w = image.shape[1]//self.filter_size
		self.image = image
		for i in range(new_h):
			for j in range(new_w):
				a = i*self.filter_size
				b = i*self.filter_size + self.filter_size
				c = j*self.filter_size
				d = j*self.filter_size + self.filter_size
				image_patch = image[a:b,c:d]
				yield image_patch, i, j
	
	  # Función que realiza una pasada hacia delante utilizando las entradas dadas
	  # Devuelve una matriz de 3D
	def forward_prop(self, image):
		height, widht, num_filters = image.shape
		output = np.zeros((height//self.filter_size, widht//self.filter_size, num_filters))
		
		for image_patch, i, j in self.image_region(image):
			output[i,j] = np.amax(image_patch, axis = (0,1))

		return output 

	  # Realiza el pasao hacia atras devolviendo el degradado de pérdida para las entrafas de esta capa
	def back_prop(self,dL_dout):
		dL_dmax_pool = np.zeros(self.image.shape)
		for image_patch, i, j in self.image_region(self.image):
			h,w,num_filters = image_patch.shape
			maximun_val = np.amax(image_patch, axis = (0,1))

			for x in range(h):
				for y in range(w):
					for z in range(num_filters):
						if image_patch[x,y,z] == maximun_val[z]:
							dL_dmax_pool[i*self.filter_size + x, j*self.filter_size +y,z]=dL_dout[i,j,z]
			return dL_dmax_pool

  # Capa para la activación de softmax
class Softmax:
	def __init__(self, input_node, sofmax_node):
		 # Se reudce la viarianza de los valores iniciales
		self.weight = np.random.randn(input_node,sofmax_node)/input_node
		self.bias = np.zeros(sofmax_node)
    
      # Funciones para hacer una pasada hacia delante 
	def forward_prop(self, image):
		self.orig_im_shape = image.shape
		image_modified = image.flatten()
		self.modified_input = image_modified
		output_val = np.dot(image_modified, self.weight) + self.bias
		self.out = output_val
		exp_out = np.exp(output_val)
		val = exp_out/np.sum(exp_out, axis=0)
		return val
	   # Función para dar una pasda en retroceso
	def back_prop(self, dL_dout, learning_rate):
		for i, grad in enumerate(dL_dout):
			if grad == 0:
				continue

			transformation_eq = np.exp(self.out)
			S_total = np.sum(transformation_eq)

			dy_dz = -transformation_eq[i]*transformation_eq/(S_total**2)
			dy_dz[i] = transformation_eq[i]*(S_total - transformation_eq[i])/(S_total**2)

			dz_dw = self.modified_input
			dz_db = 1
			dz_d_input = self.weight

			dL_dz = grad*dy_dz

			dL_dw = dz_dw[np.newaxis].T@dL_dz[np.newaxis]
			dL_db = dL_dz*dz_db
			dL_d_input = dz_d_input@dL_dz

			self.weight-=learning_rate*dL_dw
			self.bias -= learning_rate*dL_db

			return dL_d_input.reshape(self.orig_im_shape)
  # Extracción de valores de imágenes obtenidos anteriormente
train_images = A[:20] # Imágenes de entramiento
train_labels = B[:20] # Nombre de imágenes de entrenamiento
test_images = C[:20]  # Imágenes para la prueba
test_labels = D[:20]  # Nombre de imágenes para las pruebas

  # Se llaman a las clases creadas para las imágenes
conv = CNN(8,3)                  # 8 filtros con tamaño de 3
pool = Max_Pool(2)               # Tamaño de filtro: 2
softmax = Softmax(49*49*8, 10)   # Obtención del vector de cada imagen 

  # Un pase hacia delante de la neurona y calcula la precisión y perdidad de entropía cruzada
def forward(image, label):
 
   # Transformación de la imagen para su fácil manejo
  out = conv.forward_prop((image / 255) - 0.5)
  out = pool.forward_prop(out)
  out = softmax.forward_prop(out)

  # Calcula la pérdida de entropía cruzada y precisión
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

  # Función para el entrenamiento completo de la imagen y etiqueta dada
def train(im, label, lr=.005):
  
  out, loss, acc = forward(im, label)

  # Calcula el gradoemte inicial
  gradient = np.zeros(20)
  gradient[label] = -1 / out[label]

  # Paso hacia atrás del gradiente
  gradient = softmax.back_prop(gradient, lr)
  gradient = pool.back_prop(gradient)
  gradient = conv.back_prop(gradient, lr)

  return loss, acc

print('Red Neuronal Convolucional Iniciada :v')

  # Entrnamiento con una época
for epoca in range(5):
  print('Epoca %d ' % (epoca + 1))

  # Se permuta los datos de entrenamiento
  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

  loss = 0
  num_correct = 0
  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i % 100 == 99:
      loss = 0
      num_correct = 0

    l, acc = train(im, label)
    loss += l
    num_correct += acc
    

# Test the CNN
print('\nInicio de prueba')
loss = 0
correct = 0
n=0
  # Ciclo para prueba con cada imágen y label dado
for im, label in zip(test_images, test_labels):
  _, l, acc = forward(im, label)
  loss += l
  correct += acc
  
  if acc == 1:
    if label==0:
        plt.imshow(test_images[n],cmap='gray')
        plt.title("Es un rostro")
        plt.show()
    else:
        plt.imshow(test_images[n],cmap='gray')
        plt.title("Vaquiña")
        plt.show()
  else:
    plt.imshow(test_images[n],cmap='gray')
    plt.title("Sin dato")
    plt.show()
  n=n+1
num_tests = len(test_images)
print('Porcentaje de aprendizaje:', correct / num_tests*100)