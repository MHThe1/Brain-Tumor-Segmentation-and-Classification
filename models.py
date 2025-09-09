"""
Model architectures for Brain Tumor Segmentation and Classification
BRACU CSE428 Academic Project

Includes:
- U-Net architecture
- Attention U-Net architecture
- Various classifier heads
- EfficientDet integration
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer
import logging

class SelfAttentionLayer(Layer):
    """
    Custom Self-Attention layer for Keras
    """
    def __init__(self, filters, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.filters = filters
        
    def build(self, input_shape):
        self.query_conv = layers.Conv2D(self.filters // 8, 1)
        self.key_conv = layers.Conv2D(self.filters // 8, 1)
        self.value_conv = layers.Conv2D(self.filters, 1)
        super(SelfAttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Query, Key, Value projections
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        
        # Get spatial dimensions
        input_shape = tf.shape(query)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        
        # Reshape for attention computation
        query_flat = tf.reshape(query, (batch_size, height * width, self.filters // 8))
        key_flat = tf.reshape(key, (batch_size, height * width, self.filters // 8))
        value_flat = tf.reshape(value, (batch_size, height * width, self.filters))
        
        # Attention scores
        attention_scores = tf.matmul(query_flat, key_flat, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.filters // 8, tf.float32))
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply attention to values
        attended_flat = tf.matmul(attention_scores, value_flat)
        
        # Reshape back to spatial dimensions
        attended = tf.reshape(attended_flat, (batch_size, height, width, self.filters))
        
        # Residual connection
        attended = attended + x
        
        return attended
        
    def get_config(self):
        config = super(SelfAttentionLayer, self).get_config()
        config.update({'filters': self.filters})
        return config

class UNet:
    """
    Standard U-Net architecture for medical image segmentation
    """
    
    def __init__(self, input_shape=(256, 256, 3), num_classes=2, filters=64):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = filters
        self.logger = logging.getLogger(__name__)
    
    def conv_block(self, x, filters, kernel_size=3, padding='same', activation='relu'):
        """
        Convolutional block with batch normalization
        """
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x
    
    def build_model(self):
        """
        Build U-Net model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder
        c1 = self.conv_block(inputs, self.filters)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = self.conv_block(p1, self.filters * 2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = self.conv_block(p2, self.filters * 4)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = self.conv_block(p3, self.filters * 8)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        # Bottleneck
        c5 = self.conv_block(p4, self.filters * 16)
        
        # Decoder
        u6 = layers.Conv2DTranspose(self.filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = self.conv_block(u6, self.filters * 8)
        
        u7 = layers.Conv2DTranspose(self.filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = self.conv_block(u7, self.filters * 4)
        
        u8 = layers.Conv2DTranspose(self.filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = self.conv_block(u8, self.filters * 2)
        
        u9 = layers.Conv2DTranspose(self.filters, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = self.conv_block(u9, self.filters)
        
        # Output layer
        outputs = layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(c9)
        
        model = Model(inputs, outputs, name='UNet')
        return model

class AttentionUNet:
    """
    Enhanced Attention U-Net architecture with advanced attention mechanisms
    """
    
    def __init__(self, input_shape=(256, 256, 3), num_classes=2, filters=64, 
                 use_self_attention=True, use_channel_attention=True, 
                 use_deep_supervision=True, attention_type='spatial'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = filters
        self.use_self_attention = use_self_attention
        self.use_channel_attention = use_channel_attention
        self.use_deep_supervision = use_deep_supervision
        self.attention_type = attention_type
        self.logger = logging.getLogger(__name__)
    
    def conv_block(self, x, filters, kernel_size=3, padding='same', activation='relu', 
                   use_dropout=False, dropout_rate=0.1):
        """
        Enhanced convolutional block with optional dropout
        """
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        
        if use_dropout:
            x = layers.Dropout(dropout_rate)(x)
            
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x
    
    def channel_attention(self, x, filters, reduction_ratio=16):
        """
        Channel Attention Module (Squeeze-and-Excitation)
        """
        # Global Average Pooling
        gap = layers.GlobalAveragePooling2D()(x)
        gap = layers.Reshape((1, 1, -1))(gap)
        
        # Global Max Pooling
        gmp = layers.GlobalMaxPooling2D()(x)
        gmp = layers.Reshape((1, 1, -1))(gmp)
        
        # Shared MLP
        mlp = layers.Dense(filters // reduction_ratio, activation='relu')
        mlp2 = layers.Dense(filters, activation='sigmoid')
        
        # Apply MLP to both GAP and GMP
        gap_mlp = mlp2(mlp(gap))
        gmp_mlp = mlp2(mlp(gmp))
        
        # Combine and apply
        channel_attention = layers.Add()([gap_mlp, gmp_mlp])
        channel_attention = layers.Activation('sigmoid')(channel_attention)
        
        return layers.Multiply()([x, channel_attention])
    
    def spatial_attention(self, x):
        """
        Spatial Attention Module
        """
        # Channel-wise statistics using Keras operations
        avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(x)
        max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(x)
        
        # Concatenate and apply convolution
        concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
        spatial_attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(concat)
        
        return layers.Multiply()([x, spatial_attention])
    
    def self_attention(self, x, filters):
        """
        Self-Attention mechanism for long-range dependencies
        """
        return SelfAttentionLayer(filters)(x)
    
    def enhanced_attention_gate(self, g, x, filters, attention_type='spatial'):
        """
        Enhanced attention gate with multiple attention mechanisms
        """
        # g: gating signal from lower resolution
        # x: feature map from skip connection
        
        # Linear transformations
        g_conv = layers.Conv2D(filters, 1, padding='same')(g)
        g_conv = layers.BatchNormalization()(g_conv)
        
        x_conv = layers.Conv2D(filters, 1, padding='same')(x)
        x_conv = layers.BatchNormalization()(x_conv)
        
        # Add and apply ReLU
        add = layers.Add()([g_conv, x_conv])
        add = layers.Activation('relu')(add)
        
        # Apply different attention mechanisms
        if attention_type == 'spatial':
            # Spatial attention
            attention = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(add)
            attended = layers.Multiply()([x, attention])
        elif attention_type == 'channel':
            # Channel attention
            attended = self.channel_attention(x, filters)
        elif attention_type == 'both':
            # Both spatial and channel attention
            spatial_att = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(add)
            spatial_attended = layers.Multiply()([x, spatial_att])
            attended = self.channel_attention(spatial_attended, filters)
        else:
            # Default spatial attention
            attention = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(add)
            attended = layers.Multiply()([x, attention])
        
        return attended
    
    def decoder_block(self, x, skip_connection, filters, block_name):
        """
        Enhanced decoder block with attention and self-attention
        """
        # Upsampling
        x_up = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        
        # Apply attention gate
        if skip_connection is not None:
            attended_skip = self.enhanced_attention_gate(x_up, skip_connection, filters, self.attention_type)
            x_up = layers.concatenate([x_up, attended_skip])
        
        # Convolutional block
        x_up = self.conv_block(x_up, filters, use_dropout=True)
        
        # Apply self-attention if enabled
        if self.use_self_attention:
            x_up = self.self_attention(x_up, filters)
        
        # Apply channel attention if enabled
        if self.use_channel_attention:
            x_up = self.channel_attention(x_up, filters)
        
        return x_up
    
    def build_model(self):
        """
        Build Enhanced Attention U-Net model with deep supervision
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder with enhanced blocks
        c1 = self.conv_block(inputs, self.filters, use_dropout=True)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = self.conv_block(p1, self.filters * 2, use_dropout=True)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = self.conv_block(p2, self.filters * 4, use_dropout=True)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = self.conv_block(p3, self.filters * 8, use_dropout=True)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        # Bottleneck with self-attention
        c5 = self.conv_block(p4, self.filters * 16, use_dropout=True)
        if self.use_self_attention:
            c5 = self.self_attention(c5, self.filters * 16)
        if self.use_channel_attention:
            c5 = self.channel_attention(c5, self.filters * 16)
        
        # Enhanced decoder with attention gates
        d4 = self.decoder_block(c5, c4, self.filters * 8, 'd4')
        d3 = self.decoder_block(d4, c3, self.filters * 4, 'd3')
        d2 = self.decoder_block(d3, c2, self.filters * 2, 'd2')
        d1 = self.decoder_block(d2, c1, self.filters, 'd1')
        
        # Main output
        main_output = layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid', name='main_output')(d1)
        
        # Deep supervision outputs (optional)
        if self.use_deep_supervision:
            # Auxiliary outputs at different scales
            aux4 = layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid', name='aux_output_4')(d4)
            aux3 = layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid', name='aux_output_3')(d3)
            aux2 = layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid', name='aux_output_2')(d2)
            
            # Upsample auxiliary outputs to match input size
            aux4_up = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(aux4)
            aux3_up = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(aux3)
            aux2_up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(aux2)
            
            model = Model(inputs, [main_output, aux4_up, aux3_up, aux2_up], name='EnhancedAttentionUNet')
        else:
            model = Model(inputs, main_output, name='EnhancedAttentionUNet')
        
        return model
    
    def get_attention_maps(self, model, input_data, layer_names=None):
        """
        Extract attention maps from the model for visualization
        """
        if layer_names is None:
            # Look for layers that might contain attention information
            layer_names = []
            for layer in model.layers:
                if any(keyword in layer.name.lower() for keyword in ['attention', 'multiply', 'sigmoid']):
                    layer_names.append(layer.name)
        
        # Filter out layers that don't exist
        valid_layer_names = [name for name in layer_names if name in [l.name for l in model.layers]]
        
        if not valid_layer_names:
            # If no attention layers found, return empty dict
            return {}
        
        try:
            attention_model = Model(
                inputs=model.input,
                outputs=[model.get_layer(name).output for name in valid_layer_names]
            )
            
            attention_maps = attention_model.predict(input_data, verbose=0)
            return dict(zip(valid_layer_names, attention_maps))
        except Exception as e:
            print(f"Warning: Could not extract attention maps: {e}")
            return {}
    
    def visualize_attention(self, model, input_data, save_path=None):
        """
        Visualize attention maps
        """
        import matplotlib.pyplot as plt
        
        attention_maps = self.get_attention_maps(model, input_data)
        
        fig, axes = plt.subplots(2, len(attention_maps), figsize=(15, 8))
        if len(attention_maps) == 1:
            axes = axes.reshape(2, 1)
        
        for i, (layer_name, attention_map) in enumerate(attention_maps.items()):
            # Original image
            axes[0, i].imshow(input_data[0, :, :, 0], cmap='gray')
            axes[0, i].set_title(f'Input - {layer_name}')
            axes[0, i].axis('off')
            
            # Attention map
            if len(attention_map.shape) == 4:
                attention_map = attention_map[0, :, :, 0]
            axes[1, i].imshow(attention_map, cmap='hot')
            axes[1, i].set_title(f'Attention - {layer_name}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class AttentionLoss:
    """
    Custom loss functions for Attention U-Net training
    """
    
    @staticmethod
    def attention_regularization_loss(attention_maps, weight=0.1):
        """
        Regularization loss to prevent attention collapse
        """
        def loss_fn(y_true, y_pred):
            # Encourage attention maps to be diverse
            attention_entropy = -tf.reduce_sum(attention_maps * tf.math.log(attention_maps + 1e-8), axis=[1, 2, 3])
            entropy_loss = tf.reduce_mean(attention_entropy)
            return weight * entropy_loss
        return loss_fn
    
    @staticmethod
    def deep_supervision_loss(alpha=0.4, beta=0.3, gamma=0.3):
        """
        Deep supervision loss for multiple outputs
        """
        def loss_fn(y_true, y_pred):
            if isinstance(y_pred, list):
                main_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred[0])
                aux_losses = []
                for i, aux_pred in enumerate(y_pred[1:], 1):
                    # Resize ground truth to match auxiliary output size
                    aux_gt = tf.image.resize(y_true, tf.shape(aux_pred)[1:3])
                    aux_loss = tf.keras.losses.binary_crossentropy(aux_gt, aux_pred)
                    aux_losses.append(aux_loss)
                
                # Weighted combination
                total_loss = alpha * main_loss
                if len(aux_losses) >= 1:
                    total_loss += beta * aux_losses[0]
                if len(aux_losses) >= 2:
                    total_loss += gamma * aux_losses[1]
                
                return total_loss
            else:
                return tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return loss_fn

class MultiTaskUNet:
    """
    U-Net with both segmentation and classification heads
    """
    
    def __init__(self, input_shape=(256, 256, 3), seg_classes=2, cls_classes=4, filters=64):
        self.input_shape = input_shape
        self.seg_classes = seg_classes
        self.cls_classes = cls_classes
        self.filters = filters
        self.logger = logging.getLogger(__name__)
    
    def conv_block(self, x, filters, kernel_size=3, padding='same', activation='relu'):
        """
        Convolutional block with batch normalization
        """
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x
    
    def build_model(self):
        """
        Build multi-task U-Net model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder
        c1 = self.conv_block(inputs, self.filters)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = self.conv_block(p1, self.filters * 2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = self.conv_block(p2, self.filters * 4)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = self.conv_block(p3, self.filters * 8)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        # Bottleneck
        c5 = self.conv_block(p4, self.filters * 16)
        
        # Classification head (from encoder output)
        cls_global_pool = layers.GlobalAveragePooling2D()(c5)
        cls_dense = layers.Dense(512, activation='relu')(cls_global_pool)
        cls_dropout = layers.Dropout(0.5)(cls_dense)
        cls_output = layers.Dense(self.cls_classes, activation='softmax', name='classification')(cls_dropout)
        
        # Segmentation head (U-Net decoder)
        u6 = layers.Conv2DTranspose(self.filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = self.conv_block(u6, self.filters * 8)
        
        u7 = layers.Conv2DTranspose(self.filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = self.conv_block(u7, self.filters * 4)
        
        u8 = layers.Conv2DTranspose(self.filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = self.conv_block(u8, self.filters * 2)
        
        u9 = layers.Conv2DTranspose(self.filters, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = self.conv_block(u9, self.filters)
        
        # Segmentation output
        seg_output = layers.Conv2D(self.seg_classes, (1, 1), activation='sigmoid', name='segmentation')(c9)
        
        model = Model(inputs, [seg_output, cls_output], name='MultiTaskUNet')
        return model

class EfficientDetUNet:
    """
    U-Net with EfficientDet decoder (Bonus task)
    """
    
    def __init__(self, input_shape=(256, 256, 3), num_classes=2, filters=64):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = filters
        self.logger = logging.getLogger(__name__)
    
    def conv_block(self, x, filters, kernel_size=3, padding='same', activation='relu'):
        """
        Convolutional block with batch normalization
        """
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        return x
    
    def efficientdet_decoder_block(self, x, skip_connection, filters):
        """
        EfficientDet-inspired decoder block
        """
        # Feature fusion
        x_up = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        
        # Weighted feature fusion (simplified version of EfficientDet's FPN)
        if skip_connection is not None:
            skip_conv = layers.Conv2D(filters, 1, padding='same')(skip_connection)
            x_up = layers.Add()([x_up, skip_conv])
        
        # Apply activation and normalization
        x_up = layers.BatchNormalization()(x_up)
        x_up = layers.Activation('swish')(x_up)
        
        # Additional convolutions for feature refinement
        x_up = layers.Conv2D(filters, 3, padding='same')(x_up)
        x_up = layers.BatchNormalization()(x_up)
        x_up = layers.Activation('swish')(x_up)
        
        return x_up
    
    def build_model(self):
        """
        Build EfficientDet U-Net model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder (same as U-Net)
        c1 = self.conv_block(inputs, self.filters)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = self.conv_block(p1, self.filters * 2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = self.conv_block(p2, self.filters * 4)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = self.conv_block(p3, self.filters * 8)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        # Bottleneck
        c5 = self.conv_block(p4, self.filters * 16)
        
        # EfficientDet-inspired decoder
        d4 = self.efficientdet_decoder_block(c5, c4, self.filters * 8)
        d3 = self.efficientdet_decoder_block(d4, c3, self.filters * 4)
        d2 = self.efficientdet_decoder_block(d3, c2, self.filters * 2)
        d1 = self.efficientdet_decoder_block(d2, c1, self.filters)
        
        # Output layer
        outputs = layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(d1)
        
        model = Model(inputs, outputs, name='EfficientDetUNet')
        return model

class ClassifierModels:
    """
    Various classifier architectures for comparison
    """
    
    @staticmethod
    def simple_cnn(input_shape=(256, 256, 3), num_classes=4):
        """
        Simple CNN classifier
        """
        inputs = layers.Input(shape=input_shape)
        
        x = layers.Conv2D(32, 3, activation='relu')(inputs)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(128, 3, activation='relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(256, 3, activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='SimpleCNN')
        return model
    
    @staticmethod
    def resnet50_classifier(input_shape=(256, 256, 3), num_classes=4):
        """
        ResNet50-based classifier
        """
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False
        
        inputs = layers.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='ResNet50Classifier')
        return model
    
    @staticmethod
    def efficientnet_classifier(input_shape=(256, 256, 3), num_classes=4):
        """
        EfficientNet-based classifier
        """
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False
        
        inputs = layers.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='EfficientNetClassifier')
        return model
    
    @staticmethod
    def densenet_classifier(input_shape=(256, 256, 3), num_classes=4):
        """
        DenseNet-based classifier
        """
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False
        
        inputs = layers.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='DenseNetClassifier')
        return model
