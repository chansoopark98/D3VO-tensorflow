import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_images(image: tf.Tensor,
                pred_depths: tf.Tensor,
                pred_sigmas: tf.Tensor,
                gt_depth: tf.Tensor,
                mode: str,
                depth_max: float) -> tf.Tensor:
    if mode not in ['relative', 'metric']:
        raise ValueError("Mode must be either 'relative' or 'metric'.")

    if mode == 'relative':
        prefix = 'Relative GT Depth'
        depth_max = 1.0
    elif mode == 'metric':
        prefix = 'Metric GT Depth'

    # Extract the first image and depth maps for visualization
    image = image[0]
    gt_depth = tf.clip_by_value(gt_depth[0], 0.0, depth_max)

    # Plot settings
    depth_len = len(pred_depths)
    # 2행, 1+depth_len 열 구조로 변경
    fig, axes = plt.subplots(2, 1 + depth_len, figsize=(20, 10))

    # 첫 번째 행: RGB 이미지와 예측 깊이
    # Input image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Image')
    axes[0, 0].axis('off')

    # Predicted depth maps (첫 번째 행)
    for idx, pred_depth in enumerate(pred_depths):
        pred_depth = tf.clip_by_value(pred_depth[0], 0.0, depth_max)
        axes[0, idx + 1].imshow(pred_depth.numpy(), vmin=0.0, vmax=depth_max, cmap='plasma')
        axes[0, idx + 1].set_title(f'Pred Depth Scale {idx}')
        axes[0, idx + 1].axis('off')
    
    # 두 번째 행: GT 깊이와 예측 시그마 맵
    # Ground truth depth
    axes[1, 0].imshow(gt_depth.numpy(), vmin=0.0, vmax=depth_max, cmap='plasma')
    axes[1, 0].set_title(f'{prefix} ({depth_max})')
    axes[1, 0].axis('off')
    
    # Predicted sigma maps (두 번째 행)
    for idx, pred_sigma in enumerate(pred_sigmas):
        # Normalize sigma values for better visualization
        pred_sigma_vis = pred_sigma[0]
        # # log_var에서 sigma(표준 편차)로 변환 후 시각화
        # sigma = tf.exp(0.5 * pred_sigma_vis)  # sigma = sqrt(exp(log_var))
        
        # # 시그마 값을 0-1 범위로 정규화 (시각화용)
        # sigma_min = tf.reduce_min(sigma)
        # sigma_max = tf.reduce_max(sigma)
        # normalized_sigma = (sigma - sigma_min) / (tf.maximum(sigma_max - sigma_min, 1e-5))
        
        axes[1, idx + 1].imshow(pred_sigma_vis.numpy(), cmap='inferno', vmin=0., vmax=1.)  # 불확실성은 다른 컬러맵 사용
        axes[1, idx + 1].set_title(f'Uncertainty Scale {idx}')
        axes[1, idx + 1].axis('off')

    fig.tight_layout()

    # Save the plot to a buffer and convert it to a TensorFlow tensor
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    # Decode the PNG buffer into a TensorFlow tensor
    image_tensor = tf.image.decode_png(buf.getvalue(), channels=4)
    return tf.expand_dims(image_tensor, 0)