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

    def silog_loss_with_uncertainty(pred_depth: tf.Tensor,
                                pred_log_var: tf.Tensor,
                                true_depth: tf.Tensor,
                                valid_mask: tf.Tensor,
                                variance_focus: float = 0.5,
                                eps: float = 1e-6) -> tf.Tensor:
        """
        불확실성을 고려한 scale-invariant (SILog) 손실 함수입니다.
        
        Args:
            pred_depth (tf.Tensor): 모델이 예측한 깊이 (예: 단안 깊이).
            pred_log_var (tf.Tensor): 모델이 예측한 로그 분산 (log(sigma^2)).
            true_depth (tf.Tensor): ground truth 깊이.
            valid_mask (tf.Tensor): 손실 계산에 사용할 유효 픽셀에 대한 boolean 마스크.
            variance_focus (float, optional): SILog의 variance focus 파라미터 (기본값 0.5).
            eps (float, optional): 수치 안정성을 위한 작은 값 (기본값 1e-6).
        
        Returns:
            tf.Tensor: 계산된 불확실성 인식 SILog 손실 값.
        """
        # 예측 깊이가 eps 이상이 되도록 제한
        pred = tf.maximum(pred_depth, eps)
        
        # 유효 마스크 적용 (유효한 픽셀만 남김)
        pred = tf.boolean_mask(pred, valid_mask)
        true = tf.boolean_mask(true_depth, valid_mask)
        log_var = tf.boolean_mask(pred_log_var, valid_mask)
        
        # 로그 깊이 잔차 계산: d = log(pred) - log(true)
        d = tf.math.log(pred) - tf.math.log(true)
        # 각 픽셀에 대해 1/sigma^2 역할 (불확실성이 클수록 오차의 기여도를 줄임)
        inv_var = tf.exp(-log_var)
        
        # 가중치가 적용된 평균 제곱 잔차 및 가중치가 적용된 평균 잔차 계산
        weighted_mse = tf.reduce_mean(inv_var * tf.square(d))
        weighted_mean = tf.reduce_mean(inv_var * d)
        
        # SILog 손실 계산 (variance focus를 적용)
        silog_unc = weighted_mse - variance_focus * tf.square(weighted_mean)
        
        # 예측된 로그 분산에 대한 패널티 항 (과도한 불확실성 예측 방지)
        uncertainty_penalty = 0.5 * tf.reduce_mean(log_var)
        
        # 최종 손실은 SILog 항의 제곱근에 불확실성 패널티를 더한 형태입니다.
        loss = tf.sqrt(silog_unc) + uncertainty_penalty
        return loss

    def depth_uncertainty_loss(self, depth_pred: tf.Tensor,
                            log_var: tf.Tensor,
                            depth_gt: tf.Tensor,
                            valid_mask: tf.Tensor
                            ) -> tf.Tensor:
        """
        예측된 깊이와 불확실성(로그 분산)을 이용해 손실 함수를 계산합니다.
        
        Args:
            depth_pred (tf.Tensor): 예측된 깊이 맵
            log_var (tf.Tensor): 예측된 로그 분산 (log(σ²))
            depth_gt (tf.Tensor): 실제 깊이 맵
            valid_mask (tf.Tensor): 유효한 픽셀 마스크
            
        Returns:
            tf.Tensor: 불확실성을 고려한 평균 손실 값
        """
        # 유효한 픽셀만 선택
        depth_pred = tf.boolean_mask(depth_pred, valid_mask)
        depth_gt = tf.boolean_mask(depth_gt, valid_mask)
        log_var = tf.boolean_mask(log_var, valid_mask)
        
        # 예측 깊이와 실제 깊이 사이의 제곱 오차 계산
        sq_error = tf.square(depth_pred - depth_gt)
        
        # 불확실성을 고려한 손실: exp(-log_var) * (오차) + log_var
        # exp(-log_var)는 정밀도(precision)를 의미하며, 불확실할수록 작은 값
        loss = tf.exp(-log_var) * sq_error + log_var
        
        return tf.reduce_mean(loss)

    def silog_loss(self, prediction: tf.Tensor, target: tf.Tensor, valid_mask: tf.Tensor,
                   variance_focus: float = 0.5) -> tf.Tensor:
        eps = 1e-6
        prediction = tf.maximum(prediction, eps)

        valid_prediction = tf.boolean_mask(prediction, valid_mask)
        valid_target = tf.boolean_mask(target, valid_mask)

        d = tf.math.log(valid_prediction) - tf.math.log(valid_target)
        d2_mean = tf.reduce_mean(tf.square(d))
        d_mean = tf.reduce_mean(d)
        silog_expr = d2_mean - variance_focus * tf.square(d_mean)

        return tf.sqrt(silog_expr)
    
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
            log_losses += self.depth_uncertainty_loss(
                pred_depth_resized,
                pred_sigma_resized,
                gt_depth,
                valid_mask
            ) * alpha[i]


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