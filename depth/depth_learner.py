from typing import Dict, Any, List, Tuple
import tensorflow as tf, tf_keras

class DepthLearner:
    def __init__(self, model: tf_keras.Model, config: Dict[str, Any]) -> None:
        """
        Initializes the DepthLearner class.

        Args:
            model (tf.keras.Model): The Keras model for training and inference.
            config (Dict[str, Any]): Configuration dictionary containing training parameters.
        """
        self.model = model
        self.train_mode: str = config['Train']['mode']  # 'relative' or 'metric'
        self.min_depth: float = config['Train']['min_depth']  # Minimum depth (e.g., 0.1)
        self.max_depth: float = config['Train']['max_depth']  # Maximum depth (e.g., 10.0)
        self.num_scales: int = 4  # Number of scales used

    @tf.function(jit_compile=True)
    def disp_to_depth(self, disp: tf.Tensor) -> tf.Tensor:
        """
        Converts disparity to depth.

        Args:
            disp (tf.Tensor): Input disparity map.

        Returns:
            tf.Tensor: Converted depth map.
        """
        min_disp = 1.0 / self.max_depth
        max_disp = 1.0 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1.0 / scaled_disp
        return tf.cast(depth, tf.float32)

    @tf.function(jit_compile=True)
    def scaled_depth_to_disp(self, depth: tf.Tensor) -> tf.Tensor:
        """
        Converts scaled depth to disparity.

        Args:
            depth (tf.Tensor): Input depth map.

        Returns:
            tf.Tensor: Converted disparity map.
        """
        min_disp = 1.0 / self.max_depth
        max_disp = 1.0 / self.min_depth
        scaled_disp = 1.0 / depth
        disp = (scaled_disp - min_disp) / (max_disp - min_disp)
        return tf.cast(disp, tf.float32)

    @tf.function(jit_compile=True)
    def get_smooth_loss(self, disp: tf.Tensor, img: tf.Tensor) -> tf.Tensor:
        """
        Computes the edge-aware smoothness loss.

        Args:
            disp (tf.Tensor): Disparity map.
            img (tf.Tensor): Reference image.

        Returns:
            tf.Tensor: Smoothness loss value.
        """
        disp_mean = tf.reduce_mean(disp, axis=[1, 2], keepdims=True) + 1e-7
        norm_disp = disp / disp_mean

        disp_dx, disp_dy = self.compute_gradients(norm_disp)
        img_dx, img_dy = self.compute_gradients(img)

        weight_x = tf.exp(-tf.reduce_mean(img_dx, axis=3, keepdims=True))
        weight_y = tf.exp(-tf.reduce_mean(img_dy, axis=3, keepdims=True))

        smoothness_x = disp_dx * weight_x
        smoothness_y = disp_dy * weight_y

        return tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)

    @tf.function(jit_compile=True)
    def compute_gradients(self, tensor: tf.Tensor) -> tf.Tensor:
        """
        Computes gradients in the x and y directions for a tensor.

        Args:
            tensor (tf.Tensor): Input tensor.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Gradients in the x and y directions.
        """
        tensor_dx = tf.abs(tensor[:, 1:, :, :] - tensor[:, :-1, :, :])
        tensor_dy = tf.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
        return tensor_dx, tensor_dy

    @tf.function(jit_compile=True)
    def l1_loss(self, pred: tf.Tensor, gt: tf.Tensor, valid_mask: tf.Tensor) -> tf.Tensor:
        """
        Computes the L1 loss for valid pixels only.

        Args:
            pred (tf.Tensor): Predicted depth map.
            gt (tf.Tensor): Ground truth depth map.
            valid_mask (tf.Tensor): Boolean mask indicating valid pixels.

        Returns:
            tf.Tensor: Scalar L1 loss value.
        """
        abs_diff = tf.abs(pred - gt)
        masked_abs_diff = tf.boolean_mask(abs_diff, valid_mask)
        return tf.reduce_mean(masked_abs_diff)

    def confidence_silog_loss(self,
                            pred_depth: tf.Tensor,
                            confidence: tf.Tensor,  # 신뢰도 (1=높은 신뢰도)
                            true_depth: tf.Tensor,
                            valid_mask: tf.Tensor,
                            variance_focus: float = 0.5,
                            eps: float = 1e-6) -> tf.Tensor:
        """
        신뢰도 기반 SILog 손실함수 (D3VO 자기지도학습과 호환)
        
        Args:
            pred_depth: 예측 깊이 맵
            confidence: 예측 신뢰도 맵 (1=높은 신뢰도, 0=낮은 신뢰도)
            true_depth: 실제 깊이 맵
            valid_mask: 유효 픽셀 마스크
            variance_focus: SILog 보정 계수
            eps: 수치 안정성을 위한 작은 값
            
        Returns:
            tf.Tensor: 최종 손실 값
        """
        # 예측 깊이가 eps 이상이 되도록 제한
        pred_depth = tf.maximum(pred_depth, eps)
        
        # 유효 마스크 적용
        pred_depth_valid = tf.boolean_mask(pred_depth, valid_mask)
        true_depth_valid = tf.boolean_mask(true_depth, valid_mask)
        confidence_valid = tf.boolean_mask(confidence, valid_mask)
        
        # 로그 깊이 차이
        d = tf.math.log(pred_depth_valid) - tf.math.log(true_depth_valid)
        d_squared = tf.square(d)
        
        # 신뢰도 기반 가중치 적용 (자기지도학습과 일관되게)
        weighted_loss = confidence_valid * d_squared
        
        # 신뢰도에 대한 정규화 항
        # 모든 픽셀에 대해 높은 신뢰도를 예측하는 것 방지
        reg_term = 0.1 * tf.math.log(tf.maximum(confidence_valid, eps))
        
        # 픽셀별 손실
        per_pixel_loss = weighted_loss - reg_term
        
        # 평균 계산
        mean_loss = tf.reduce_mean(per_pixel_loss)
        
        # 신뢰도 가중치를 적용한 평균 로그 오차 (SI 효과용)
        weighted_d = confidence_valid * d
        d_mean = tf.reduce_mean(weighted_d)
        
        # scale-invariant 항
        si_term = mean_loss - variance_focus * tf.square(d_mean)
        
        # 최종 손실
        total_loss = tf.sqrt(tf.maximum(si_term, 0.0))
        
        return total_loss

    # @tf.function(jit_compile=True)
    def multi_scale_loss(self, pred_depths: List[tf.Tensor], pred_sigmas, gt_depth: tf.Tensor,
                         rgb: tf.Tensor, valid_mask: tf.Tensor) -> Dict[str, tf.Tensor]:
        alpha = [1 / 2, 1 / 4, 1 / 8, 1 / 16]
        smooth_losses, log_losses, l1_losses = 0.0, 0.0, 0.0

        smooth_loss_factor = 1.0
        log_loss_factor = 1.0
        l1_loss_factor = 0.1

        original_shape = gt_depth.shape[1:3]

        for i in range(self.num_scales):
            pred_depth = pred_depths[i]                  
            pred_sigma = pred_sigmas[i]

            pred_depth_resized = tf.image.resize(
                pred_depth, original_shape, method=tf.image.ResizeMethod.BILINEAR
            )

            pred_sigma_resized = tf.image.resize(
                pred_sigma, original_shape, method=tf.image.ResizeMethod.BILINEAR
            )


            resized_disp = self.scaled_depth_to_disp(pred_depth_resized)

            # Calculate losses with uncertainty
            smooth_losses += self.get_smooth_loss(resized_disp, rgb) * alpha[i]

            log_losses += self.confidence_silog_loss(pred_depth_resized,
                                                     pred_sigma_resized,
                                                     gt_depth,
                                                     valid_mask) * alpha[i]

        return {
            'smooth_loss': smooth_losses * smooth_loss_factor,
            'log_loss': log_losses * log_loss_factor,
            'l1_loss': l1_losses * l1_loss_factor
        }
    
    def forward_step(self, rgb: tf.Tensor, depth: tf.Tensor, intrinsic, training: bool = True
                    ) -> Tuple[Dict[str, tf.Tensor], List[tf.Tensor]]:
        """
        Performs a forward step, predicting depth and calculating losses.

        Args:
            rgb (tf.Tensor): Input RGB image tensor of shape [B, H, W, 3].
            depth (tf.Tensor): Ground truth depth tensor of shape [B, H, W] or [B, H, W, 1].
            training (bool): Flag indicating whether the model is in training mode.

        Returns:
            Tuple[Dict[str, tf.Tensor], List[tf.Tensor]]:
                - Loss dictionary containing smoothness, SILog, and L1 losses.
                - List of predicted depth maps at different scales.
        """

        pred_disps, pred_sigmas = self.model(rgb, training=training)

        valid_mask = (depth > 0.) & (depth < (1. if self.train_mode == 'relative' else self.max_depth))

        pred_depths = [self.disp_to_depth(disp) for disp in pred_disps]

        loss_dict = self.multi_scale_loss(
            pred_depths=pred_depths,
            pred_sigmas=pred_sigmas,
            gt_depth=depth,
            rgb=rgb,
            valid_mask=valid_mask
        )

        return loss_dict, pred_depths, pred_sigmas