import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow as tf, tf_keras
import keras
from dataset.data_loader import DataLoader
from utils.plot_utils import PlotTool
from eval import EvalTrajectory
from model.pose_net import PoseNetAB
from model.depth_net import DispNetSigma
from d3vo_learner import Learner
# from d3vo_learner_full import Learner
from tqdm import tqdm
import numpy as np
from datetime import datetime
import yaml

np.set_printoptions(suppress=True)

class Trainer(object):
    def __init__(self, config, ) -> None:
        self.config = config
        original_name = self.config['Directory']['exp_name']
        self.config['Directory']['exp_name'] = 'mode={0}_res={1}_ep={2}_bs={3}_initLR={4}_endLR={5}_prefix={6}'.format(self.config['Train']['mode'],
                                                                    (self.config['Train']['img_h'], self.config['Train']['img_w']),
                                                                    self.config['Train']['epoch'],
                                                                    self.config['Train']['batch_size'],
                                                                    self.config['Train']['init_lr'],
                                                                    self.config['Train']['final_lr'],
                                                                    original_name
                                                                    )

        self.configure_train_ops()
        print('initialize')
    
    def lr_schedule(self, epoch, total_epochs):
        initial_lr = self.config['Train']['init_lr']  # 10^-4
        final_lr = self.config['Train']['final_lr']   # 10^-5
        
        if epoch >= (total_epochs - 5):
            # For the last 5 epochs, decrease from initial_lr to final_lr linearly
            decay_epochs = 5
            remaining = total_epochs - epoch
            return final_lr + (initial_lr - final_lr) * (remaining / decay_epochs)
        else:
            # For all other epochs, return the initial learning rate
            return initial_lr

    def configure_train_ops(self) -> None:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)

        # 1. Model
        self.batch_size = self.config['Train']['batch_size']

        image_shape = (self.config['Train']['img_h'], self.config['Train']['img_w'])
        self.depth_net = DispNetSigma(image_shape=image_shape, batch_size=self.batch_size, prefix='disp_resnet')
        dispnet_input_shape = (self.config['Train']['batch_size'],
                               self.config['Train']['img_h'],
                               self.config['Train']['img_w'],
                               3)
        self.depth_net.build(dispnet_input_shape)
        _ = self.depth_net(tf.random.normal(dispnet_input_shape))

        # self.depth_net.load_weights('./assets/weights/vo/depth_net_epoch_24_model.weights.h5')

        self.pose_net = PoseNetAB(image_shape=image_shape, batch_size=self.batch_size, prefix='mono_posenet')
        posenet_input_shape = [(self.batch_size, *image_shape, 6)]
        self.pose_net.build(posenet_input_shape)

        # self.pose_net.load_weights('./assets/weights/vo/pose_net_epoch_24_model.weights.h5')
        
        # 2. Dataset
        self.data_loader = DataLoader(config=self.config)
        
        # 3. Optimizer
        self.warmup_scheduler = keras.optimizers.schedules.PolynomialDecay(self.config['Train']['init_lr'],
                                                                              self.config['Train']['epoch'],
                                                                              self.config['Train']['final_lr'],
                                                                              power=0.9)
        
        self.optimizer = keras.optimizers.Adam(learning_rate=self.config['Train']['init_lr'],
                                               beta_1=self.config['Train']['beta1'],
                                               weight_decay=self.config['Train']['weight_decay'] if self.config[
                                                   'Train']['weight_decay'] > 0 else None,
                                               )
        self.optimizer = keras.mixed_precision.LossScaleOptimizer(self.optimizer)

        # 4. Train Method
        self.learner = Learner(depth_model=self.depth_net,
                               pose_model=self.pose_net,
                               config=self.config)

        self.eval_tool = EvalTrajectory(depth_model=self.depth_net,
                                        pose_model=self.pose_net, config=self.config)

        self.plot_tool = PlotTool(config=self.config)

        # 5. Metrics
        self.train_total_loss = tf_keras.metrics.Mean(name='train_total_loss')
        self.train_pixel_loss = tf_keras.metrics.Mean(name='train_pixel_loss')
        self.train_reg_loss = tf_keras.metrics.Mean(name='train_reg_loss')
        self.train_uncertainty_loss = tf_keras.metrics.Mean(name='train_uncertainty_loss')

        self.valid_total_loss = tf_keras.metrics.Mean(name='valid_total_loss')
        self.valid_pixel_loss = tf_keras.metrics.Mean(name='valid_pixel_loss')
        self.valid_reg_loss = tf_keras.metrics.Mean(name='valid_reg_loss')
        self.valid_uncertainty_loss = tf_keras.metrics.Mean(name='valid_uncertainty_loss')

        # 6. Logger
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_path = os.path.join('vo', self.config['Directory']['log_dir'] + \
            '/' + current_time + '_')
        self.train_summary_writer = tf.summary.create_file_writer(
            tensorboard_path + self.config['Directory']['exp_name'] + '/train')
        self.valid_summary_writer = tf.summary.create_file_writer(
            tensorboard_path + self.config['Directory']['exp_name'] + '/valid')
        self.test_summary_writer = tf.summary.create_file_writer(
            tensorboard_path + self.config['Directory']['exp_name'] + '/test')

        os.makedirs(self.config['Directory']['weights'], exist_ok=True)
        self.save_path = '{0}/vo/{1}'.format(self.config['Directory']['weights'],
                                     self.config['Directory']['exp_name'])
        os.makedirs(self.save_path, exist_ok=True)
    
    @tf.function(jit_compile=True)
    def train_step(self, ref_images, target_image, intrinsic):
        with tf.GradientTape() as tape:
            total_loss, pixel_loss, reg_loss, uncertainty_losses, predictions = self.learner.forward_step(
                ref_images, target_image, intrinsic, training=True)
            scaled_loss = self.optimizer.scale_loss(total_loss)
        
        all_vars = self.depth_net.trainable_variables + self.pose_net.trainable_variables
        grads = tape.gradient(scaled_loss, all_vars)
        self.optimizer.apply_gradients(zip(grads, all_vars))
        return total_loss, pixel_loss, reg_loss, uncertainty_losses, predictions

    @tf.function(jit_compile=True)
    def validation_step(self, ref_images, target_image, intrinsic) -> tf.Tensor:
        total_loss, pixel_loss, reg_loss, uncertainty_losses, predictions = self.learner.forward_step(ref_images, target_image, intrinsic, training=False)
        return total_loss, pixel_loss, reg_loss, uncertainty_losses, predictions

    def train(self) -> None:        
        for epoch in range(self.config['Train']['epoch']):    
            lr = self.lr_schedule(epoch, self.config['Train']['epoch'])

            # Set learning rate
            self.optimizer.learning_rate = lr
            
            train_tqdm = tqdm(self.data_loader.train_dataset, total=self.data_loader.num_train_samples)
            print(' LR : {0}'.format(self.optimizer.learning_rate))
            train_tqdm.set_description('Training   || Epoch : {0} ||'.format(epoch,
                                                                             round(float(self.optimizer.learning_rate.numpy()), 8)))
            for idx, (ref_images, target_image, intrinsic) in enumerate(train_tqdm):
                train_t_loss, train_p_loss, train_r_loss, train_u_loss, predictions = self.train_step(ref_images, target_image, intrinsic)

                # Update train metrics
                self.train_total_loss(train_t_loss)
                self.train_pixel_loss(train_p_loss)
                self.train_reg_loss(train_r_loss)
                self.train_uncertainty_loss(train_u_loss)

                if idx % self.config['Train']['train_plot_interval'] == 0:
                    current_step = self.data_loader.num_train_samples * epoch + idx

                    # Draw depth plot
                    train_depth_plot = self.plot_tool.plot_images(images=target_image, # target_image
                                                                  predictions=predictions,
                                                                  denorm_func=self.data_loader.denormalize_image)

                    with self.train_summary_writer.as_default():
                        # Logging train images
                        tf.summary.image('Train/Depth Result', train_depth_plot, step=current_step)
                        
                train_tqdm.set_postfix(
                    total_loss=self.train_total_loss.result().numpy(),
                    pixel_loss=self.train_pixel_loss.result().numpy(),
                    smooth_loss=self.train_reg_loss.result().numpy(),
                    uncertainty_loss=self.train_uncertainty_loss.result().numpy()
                )
            
            # Logging train metrics
            with self.train_summary_writer.as_default():
                # Logging train total, pixel, smooth loss
                tf.summary.scalar(f'Train/{self.train_total_loss.name}',
                                    self.train_total_loss.result(), step=epoch)
                tf.summary.scalar(f'Train/{self.train_pixel_loss.name}',
                                    self.train_pixel_loss.result(), step=epoch)
                tf.summary.scalar(f'Train/{self.train_reg_loss.name}',
                                    self.train_reg_loss.result(), step=epoch)
                tf.summary.scalar(f'Train/{self.train_uncertainty_loss.name}',
                                    self.train_uncertainty_loss.result(), step=epoch)
            
            # Validation
            valid_tqdm = tqdm(self.data_loader.valid_dataset, total=self.data_loader.num_valid_samples)
            valid_tqdm.set_description('Validation || ')
            for idx, (ref_images, target_image, intrinsic) in enumerate(valid_tqdm):
                valid_t_loss, valid_p_loss, valid_s_loss, valid_u_loss, pred_valid_depths = self.validation_step(ref_images, target_image, intrinsic)

                # Update valid metrics
                self.valid_total_loss(valid_t_loss)
                self.valid_pixel_loss(valid_p_loss)
                self.valid_reg_loss(valid_s_loss)
                self.valid_uncertainty_loss(valid_u_loss)

                if idx % self.config['Train']['valid_plot_interval'] == 0:
                    current_step = self.data_loader.num_valid_samples * epoch + idx
                    # Draw target image - target depth plot
                    valid_depth_plot = self.plot_tool.plot_images(images=target_image,
                                                                  predictions=pred_valid_depths,
                                                                  denorm_func=self.data_loader.denormalize_image)

                    with self.valid_summary_writer.as_default():
                        # Logging valid images
                        tf.summary.image('Valid/Depth Result', valid_depth_plot, step=current_step)

                valid_tqdm.set_postfix(
                    total_loss=self.valid_total_loss.result().numpy(),
                    pixel_loss=self.valid_pixel_loss.result().numpy(),
                    smooth_loss=self.valid_reg_loss.result().numpy(),
                    uncertainty_loss=self.valid_uncertainty_loss.result().numpy()
                )

            # Logging valid metrics
            with self.valid_summary_writer.as_default():
                # Logging valid total, pixel, smooth loss
                tf.summary.scalar(f'Valid/{self.valid_total_loss.name}',
                                    self.valid_total_loss.result(), step=epoch)
                tf.summary.scalar(f'Valid/{self.valid_pixel_loss.name}',
                                    self.valid_pixel_loss.result(), step=epoch)
                tf.summary.scalar(f'Valid/{self.valid_reg_loss.name}',
                                    self.valid_reg_loss.result(), step=epoch)
                tf.summary.scalar(f'Valid/{self.valid_uncertainty_loss.name}',
                                    self.valid_uncertainty_loss.result(), step=epoch)

            # Eval
            print('Evaluate trajectory ... Current Epoch : {0}'.format(epoch))
            test_tqdm = tqdm(self.data_loader.test_dataset, total=self.data_loader.num_test_samples)
            test_tqdm.set_description('Test || ')
            for idx, (ref_images, target_image, intrinsic) in enumerate(test_tqdm):
                self.eval_tool.update_state(ref_images, target_image, intrinsic)

            eval_plot = self.eval_tool.eval_plot()
            with self.test_summary_writer.as_default():
                # Logging eval images
                tf.summary.image('Eval/Trajectory', eval_plot, step=epoch)
            
            # Save weights
            if epoch % self.config['Train']['save_freq'] == 0:
                self.depth_net.save_weights(self.save_path + '/depth_net_epoch_{0}_model.weights.h5'.format(epoch))
                self.pose_net.save_weights(self.save_path + '/pose_net_epoch_{0}_model.weights.h5'.format(epoch))
            
            # Reset metrics        
            self.train_total_loss.reset_states()
            self.train_pixel_loss.reset_states()
            self.train_reg_loss.reset_states()
            self.train_uncertainty_loss.reset_states()
            self.valid_total_loss.reset_states()
            self.valid_pixel_loss.reset_states()
            self.valid_reg_loss.reset_states()
            self.valid_uncertainty_loss.reset_states()

if __name__ == '__main__':
    with open('./vo/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Get GPU configuration and set visible GPUs
    gpu_config = config.get('Experiment', {})
    visible_gpus = gpu_config.get('gpus', [])
    gpu_vram = gpu_config.get('gpu_vram', None)
    gpu_vram_factor = gpu_config.get('gpu_vram_factor', None)

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            if visible_gpus:
                selected_gpus = [gpus[i] for i in visible_gpus]
                tf.config.set_visible_devices(selected_gpus, 'GPU')
            else:
                print("No GPUs specified in config. Using all available GPUs.")
                selected_gpus = gpus
            
            if gpu_vram and gpu_vram_factor:
                for gpu in selected_gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_vram * gpu_vram_factor)]
                    )
            
            print(f"Using GPUs: {selected_gpus}")
        except RuntimeError as e:
            print(f"Error during GPU configuration: {e}")
    else:
        print('No GPU devices found')
        raise SystemExit

    # with strategy.scope():
    trainer = Trainer(config=config)
    trainer.train()