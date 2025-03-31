import tensorflow as tf, tf_keras
try:
    from .flownet import CustomFlow
    from .resnet_tf import Resnet
except:
    from flownet import CustomFlow
    from resnet_tf import Resnet


def std_conv(filter_size, out_channel, stride, use_bias=True, pad='same', name='conv'):
    conv_layer = tf_keras.layers.Conv2D(out_channel,
                                        (filter_size, filter_size),
                                         strides=(stride, stride), 
                                         use_bias=use_bias,
                                         padding=pad,
                                         name=name+'_'+'conv')
    return conv_layer

class PoseNetAB(tf_keras.Model):
    def __init__(self,
                 image_shape: tuple,
                 batch_size: int,
                 prefix='pose_resnet',
                 **kwargs):
        super(PoseNetAB, self).__init__(**kwargs)

        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.batch_size = batch_size

        # self.encoder = CustomFlow(image_shape=(self.image_height, self.image_width, 6), batch_size=batch_size, pretrained=True).build_model()
        # self.encoder.build((self.batch_size, self.image_height, self.image_width, 6))
        # self.encoder.trainable = True

        self.encoder = Resnet(image_shape=(self.image_height, self.image_width, 6), batch_size=batch_size, pretrained=True, prefix='resnet18_pose').build_model()
        self.encoder.build((self.batch_size, self.image_height, self.image_width, 6))
        self.encoder.trainable = True
        
        # 공통 특징 추출층
        self.shared_features_1 = tf_keras.Sequential([
            std_conv(1, 256, 1, use_bias=True, name='shared_conv1'),
            tf_keras.layers.ReLU(),
            std_conv(3, 256, 1, use_bias=True, name='shared_conv1_2'),
            tf_keras.layers.ReLU(),
        ])

        self.shared_features_2 = tf_keras.Sequential([
            std_conv(3, 256, 1, use_bias=True, name='shared_conv2'),
            tf_keras.layers.ReLU(),
        ])

        self.shared_features_3 = tf_keras.Sequential([
            std_conv(3, 6, 1, use_bias=True, name='shared_conv3'),
        ]) 

        # 밝기 조정 파라미터 브랜치 (a와 b)
        self.a_conv = tf_keras.layers.Conv2D(
            filters=1, kernel_size=(1, 1), strides=(1, 1),
            use_bias=True,
            padding='same', name='a_conv'
        )
        
        self.b_conv = tf_keras.layers.Conv2D(
            filters=1, kernel_size=(1, 1), strides=(1, 1),
            use_bias=True,
            padding='same', name='b_conv'
        )

    def call(self, inputs, training=False):
        x, _ = self.encoder(inputs, training=training)
        shared_1 = self.shared_features_1(x)
        shared_2 = self.shared_features_2(shared_1)
        shared_3 = self.shared_features_3(shared_2)

        out_pose = tf.reduce_mean(shared_3, axis=[1, 2], keepdims=False)

        out_a = self.a_conv(shared_2)
        out_a = tf.math.softplus(out_a) # softplus activation
        out_a = tf.reduce_mean(out_a, axis=[1, 2], keepdims=False)

        out_b = self.b_conv(shared_2)
        out_b = tf.math.tanh(out_b) # tanh activation
        out_b = tf.reduce_mean(out_b, axis=[1, 2], keepdims=False)

        out_pose *= 0.01
        out_a *= 0.01
        out_b *= 0.01

        return out_pose, out_a, out_b

    
if __name__ == '__main__':
    # Test PoseNet
    image_shape = (480, 640)
    batch_size = 2
    # posenet = PoseNetExtra(image_shape=image_shape, batch_size=batch_size)
    posenet = PoseNetAB(image_shape=image_shape, batch_size=batch_size)
    posenet.build((batch_size, image_shape[0], image_shape[1], 6))
    posenet.summary()
    
    # Test forward
    inputs = tf.random.normal((batch_size, image_shape[0], image_shape[1], 6))
    outputs = posenet(inputs)
    print(outputs)