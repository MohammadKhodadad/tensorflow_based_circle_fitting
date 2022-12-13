import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px


sampled_number=32
train_val_split=0.7
#sampled=np.array() This should be your input data of shape (n,sampled_number,3)
train=sampled_data[:int(sampled_data.shape[0]*train_val_split)]
val=sampled_data[int(sampled_data.shape[0]*train_val_split):]



class circle_transform(tf.keras.layers.Layer):

    def __init__(self):
        '''Initializes the instance attributes'''
        super(circle_transform, self).__init__()

    def build(self,input_shape):
        '''Create the state of the layer (weights)'''
        # initialize the weights
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()
        self.tetha = tf.Variable(name="tetha",   initial_value=w_init(shape=(1,),
                 dtype='float32'),trainable=True)
        self.r = tf.Variable(name="r",   initial_value=b_init(shape=(1,),
                 dtype='float32'),trainable=True)

    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        x=inputs[:,:,0:1]
        y=inputs[:,:,1:2]
        head=inputs[:,:,2:]
        head_new=head+self.tetha
        x_center= x - self.r*tf.math.sin(head_new)
        y_center= y - self.r*tf.math.cos(head_new)
        return tf.concat([x_center,y_center],axis=-1)

class circle_loss(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
  def call(self, y,y2):
    estimated_center=tf.reduce_mean(y,axis=1,keepdims=True)
    return tf.reduce_mean(tf.math.reduce_std(tf.reduce_mean((estimated_center-y2)**2,axis=-1),axis=1))


class circle_loss2(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
  def call(self, y,y2):
    estimated_center=tf.reduce_mean(y,axis=1,keepdims=True)
    return tf.reduce_mean(tf.math.reduce_std(tf.reduce_mean((estimated_center-y2)**2,axis=-1),axis=1))
  
  
  
  
inp=tf.keras.layers.Input((sampled_number,3))
out=circle_transform()(inp)
model=tf.keras.Model(inputs=inp,outputs=out)
model.compile(loss=circle_loss(),run_eagerly=True)
model.summary()



checkpoint_filepath = 'CHECKPOINT.ckpt'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model.fit(train,train[:,:,:2],batch_size=16,epochs=5000,validation_data=(val,val[:,:,:2]),callbacks=[model_checkpoint_callback])


model.load_weights(checkpoint_filepath)



tetha=np.array(model.weights[0])[0]
r=np.array(model.weights[1])[0]
print(f"r: {r}\ntetha: {tetha} or {tetha*180/np.pi} degree")






