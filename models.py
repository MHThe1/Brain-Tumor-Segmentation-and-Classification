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
import logging

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
    Attention U-Net architecture with attention gates
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
    
    def attention_gate(self, g, x, filters):
        """
        Attention gate mechanism
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
        
        # Attention coefficients
        attention = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(add)
        
        # Apply attention to feature map
        attended = layers.Multiply()([x, attention])
        
        return attended
    
    def build_model(self):
        """
        Build Attention U-Net model
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
        
        # Decoder with attention gates
        u6 = layers.Conv2DTranspose(self.filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
        a6 = self.attention_gate(u6, c4, self.filters * 8)
        u6 = layers.concatenate([u6, a6])
        c6 = self.conv_block(u6, self.filters * 8)
        
        u7 = layers.Conv2DTranspose(self.filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
        a7 = self.attention_gate(u7, c3, self.filters * 4)
        u7 = layers.concatenate([u7, a7])
        c7 = self.conv_block(u7, self.filters * 4)
        
        u8 = layers.Conv2DTranspose(self.filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
        a8 = self.attention_gate(u8, c2, self.filters * 2)
        u8 = layers.concatenate([u8, a8])
        c8 = self.conv_block(u8, self.filters * 2)
        
        u9 = layers.Conv2DTranspose(self.filters, (2, 2), strides=(2, 2), padding='same')(c8)
        a9 = self.attention_gate(u9, c1, self.filters)
        u9 = layers.concatenate([u9, a9])
        c9 = self.conv_block(u9, self.filters)
        
        # Output layer
        outputs = layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(c9)
        
        model = Model(inputs, outputs, name='AttentionUNet')
        return model

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
