import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, BatchNormalization, ReLU, Add, GlobalAveragePooling2D
from tensorflow import Tensor
from tensorflow import keras
from keras.models import load_model

model_link = 'https://drive.google.com/file/d/1L7bJkPGEKSeWYdvJcI09FG16eAiLXej4/view?usp=sharing'


def residual_block(x: Tensor, downsample_1: bool, downsample_2:bool, filters: int, kernel_size: int = 3) -> Tensor:
  print("x shape : {}".format(x.shape))
  y = Conv2D(kernel_size=kernel_size,
              strides= (1 if not downsample_1 else 2),
              filters=filters,
              padding="same")(x)
  print("y shape : {}".format(y.shape))
  z = BatchNormalization()(y)
  p = ReLU()(z)
  print("p shape : {}".format(p.shape))
  q = Conv2D(kernel_size=kernel_size,
              strides= (1 if not downsample_2 else 2),
              filters=filters,
              padding="same")(p)
  print("q shape : {}".format(q.shape))
  r = BatchNormalization()(q)

  if downsample_1 or downsample_2:
    x = Conv2D(kernel_size=1,
                strides=2,
                filters=filters,
                padding="same")(x)
  s = Add()([x,r])

  output = ReLU()(s)

  return output

def section_a(input):
  sec_a_1 = residual_block(input,False,False,32,3)
  sec_a_2 = residual_block(sec_a_1,False,False,32,3)
  sec_a_3 = residual_block(sec_a_2,False,False,32,3)

  return sec_a_3

def section_b(input):
  # Setting downsample_1 to True changes the stride of conv1 in res block to stride =2
  sec_b_1 = residual_block(input,True,False,64,3)
  sec_b_2 = residual_block(sec_b_1,False,False,64,3)
  sec_b_3 = residual_block(sec_b_2,False,False,64,3)

  return sec_b_3

def section_c(input):
  # Setting downsample_1 to True changes the stride of conv1 in res block to stride =2
  sec_c_1 = residual_block(input,True,False,128,3)
  sec_c_2 = residual_block(sec_c_1,False,False,128,3)
  sec_c_3 = residual_block(sec_c_2,False,False,128,3)

  return sec_c_3


def create_model():
  img_shape = (32,32,3)
  img_inputs = keras.Input(shape=img_shape)
  conv_1 = keras.layers.Conv2D(kernel_size=3,filters=32)(img_inputs)
  a = BatchNormalization()(conv_1)
  input_resnet = ReLU()(a)

  output_section_a = section_a(input_resnet)
  output_section_b = section_b(output_section_a)
  output_section_c = section_c(output_section_b)

  gap_output = GlobalAveragePooling2D()(output_section_c)
  flatten_output = Flatten()(gap_output)
  output = Dense(10, activation='softmax')(flatten_output)

  model = keras.Model(inputs=img_inputs, outputs=output)


  return model

def get_trained_model():
  
  """Please download the saved model from above mentioned link from google drive as mentioned in variable model_link and keep it in
  same directory as ResModel.py"""

  model = load_model('my_model.h5')
  return model

if __name__ == "__main__":

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      print(e)

  cifar10 = tf.keras.datasets.cifar10
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0


  model = create_model()
  model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.fit(x_train,y_train,epochs=5, batch_size=128)
  model.save('my_model.h5')
  model.evaluate(x_test,y_test)

  