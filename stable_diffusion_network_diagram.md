# Stable Diffusion 项目网络架构图

## 🏗️ 整体架构图

```mermaid
graph TB
    %% 输入层
    subgraph "输入层"
        TextPrompt["📝 文本提示词<br/>Text Prompt"]
        InputImage["🖼️ 输入图像<br/>Input Image (可选)"]
    end

    %% 编码器层
    subgraph "编码器层"
        CLIPEncoder["🔤 CLIP文本编码器<br/>CLIP ViT-L/14<br/>(123M参数)"]
        VAEEncoder["📷 VAE编码器<br/>AutoencoderKL<br/>512×512 → 64×64×4"]
    end

    %% 潜在空间扩散
    subgraph "潜在扩散层"
        NoiseScheduler["🎲 噪声调度器<br/>Linear Schedule<br/>β: 0.00085→0.012"]
        UNet["🧠 U-Net去噪网络<br/>860M参数<br/>32层注意力机制"]
        CrossAttention["⚡ 交叉注意力<br/>文本-图像特征融合<br/>Context Dim: 768"]
    end

    %% 采样器
    subgraph "采样算法"
        DDIM["🔄 DDIM采样器<br/>确定性采样"]
        PLMS["🌊 PLMS采样器<br/>Katherine Crowson实现"]
        DPMSolver["⚙️ DPM-Solver<br/>高效微分方程求解"]
    end

    %% 解码器层
    subgraph "解码器层"
        VAEDecoder["🎨 VAE解码器<br/>64×64×4 → 512×512×3"]
        SafetyChecker["🛡️ 安全检查器<br/>NSFW内容过滤"]
        Watermark["💧 不可见水印<br/>机器生成标识"]
    end

    %% 输出层
    OutputImage["🖼️ 生成图像<br/>Generated Image<br/>512×512 RGB"]

    %% 连接关系
    TextPrompt --> CLIPEncoder
    InputImage -.-> VAEEncoder
    
    CLIPEncoder --> CrossAttention
    VAEEncoder -.-> NoiseScheduler
    
    NoiseScheduler --> UNet
    CrossAttention --> UNet
    UNet --> DDIM
    UNet --> PLMS  
    UNet --> DPMSolver
    
    DDIM --> VAEDecoder
    PLMS --> VAEDecoder
    DPMSolver --> VAEDecoder
    
    VAEDecoder --> SafetyChecker
    SafetyChecker --> Watermark
    Watermark --> OutputImage

    %% 样式定义
    classDef inputNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef encoderNode fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef diffusionNode fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef samplerNode fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef decoderNode fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef outputNode fill:#fff8e1,stroke:#f57f17,stroke-width:3px

    class TextPrompt,InputImage inputNode
    class CLIPEncoder,VAEEncoder encoderNode
    class NoiseScheduler,UNet,CrossAttention diffusionNode
    class DDIM,PLMS,DPMSolver samplerNode
    class VAEDecoder,SafetyChecker,Watermark decoderNode
    class OutputImage outputNode
```

## 🔧 详细组件架构

### 1. 文本编码分支
```mermaid
graph LR
    subgraph "CLIP文本编码器流程"
        A["文本输入"] --> B["Tokenization<br/>分词处理"]
        B --> C["Text Encoder<br/>Transformer"]
        C --> D["文本特征<br/>77×768"]
        D --> E["交叉注意力<br/>Cross-Attention"]
    end
    
    classDef processNode fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    class A,B,C,D,E processNode
```

### 2. 图像编码分支
```mermaid
graph LR
    subgraph "VAE编码器流程"
        A["输入图像<br/>512×512×3"] --> B["卷积编码<br/>Down-sampling"]
        B --> C["潜在表示<br/>64×64×4"]
        C --> D["高斯分布<br/>μ, σ"]
        D --> E["重参数化<br/>z = μ + σ×ε"]
    end
    
    classDef vaeNode fill:#f1f8e9,stroke:#388e3c,stroke-width:2px
    class A,B,C,D,E vaeNode
```

### 3. U-Net扩散网络结构
```mermaid
graph TB
    subgraph "U-Net架构 (860M参数)"
        Input["噪声潜在向量<br/>64×64×4"]
        
        subgraph "下采样路径"
            Down1["Conv Block 1<br/>64×64 → 32×32<br/>320 channels"]
            Down2["Conv Block 2<br/>32×32 → 16×16<br/>640 channels"] 
            Down3["Conv Block 3<br/>16×16 → 8×8<br/>1280 channels"]
            Down4["Conv Block 4<br/>8×8 → 8×8<br/>1280 channels"]
        end
        
        subgraph "中间层"
            Mid["Middle Block<br/>Self + Cross Attention<br/>1280 channels"]
        end
        
        subgraph "上采样路径"
            Up1["Conv Block 1<br/>8×8 → 8×8<br/>1280 channels"]
            Up2["Conv Block 2<br/>8×8 → 16×16<br/>1280 channels"]
            Up3["Conv Block 3<br/>16×16 → 32×32<br/>640 channels"]
            Up4["Conv Block 4<br/>32×32 → 64×64<br/>320 channels"]
        end
        
        Output["预测噪声<br/>64×64×4"]
        
        %% 跳跃连接
        Input --> Down1
        Down1 --> Down2
        Down2 --> Down3  
        Down3 --> Down4
        Down4 --> Mid
        Mid --> Up1
        Up1 --> Up2
        Up2 --> Up3
        Up3 --> Up4
        Up4 --> Output
        
        %% 跳跃连接 (虚线)
        Down1 -.-> Up4
        Down2 -.-> Up3
        Down3 -.-> Up2
        Down4 -.-> Up1
    end
    
    subgraph "注意力机制"
        CrossAttn["交叉注意力<br/>文本条件融合"]
        SelfAttn["自注意力<br/>空间特征关联"]
        TimeEmb["时间步嵌入<br/>t ∈ [0,1000]"]
    end
    
    CrossAttn -.-> Mid
    SelfAttn -.-> Mid  
    TimeEmb -.-> Mid
    
    classDef downNode fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef midNode fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    classDef upNode fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef attnNode fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    
    class Down1,Down2,Down3,Down4 downNode
    class Mid midNode
    class Up1,Up2,Up3,Up4 upNode
    class CrossAttn,SelfAttn,TimeEmb attnNode
```

## 🔄 数据流时序图

```mermaid
sequenceDiagram
    participant User as 👤 用户
    participant Text as 📝 文本编码器
    participant Noise as 🎲 噪声生成器  
    participant UNet as 🧠 U-Net
    participant Sample as 🔄 采样器
    participant VAE as 🎨 VAE解码器
    participant Safety as 🛡️ 安全检查
    
    User->>Text: "a cat sitting on a chair"
    Text->>Text: CLIP编码 → 77×768特征
    
    Noise->>UNet: 随机噪声 z_T
    Text->>UNet: 文本条件 c
    
    loop 扩散采样过程 (T→0)
        UNet->>UNet: 预测噪声 ε_θ(z_t, t, c)
        Sample->>Sample: 去噪步骤 z_{t-1} = f(z_t, ε_θ)
    end
    
    Sample->>VAE: 最终潜在向量 z_0
    VAE->>VAE: 解码 → 512×512图像
    
    VAE->>Safety: 生成图像
    Safety->>Safety: NSFW检测
    Safety->>User: ✅ 安全图像输出
```

## 📊 模型参数统计

| 组件 | 参数量 | 输入维度 | 输出维度 | 功能 |
|------|--------|----------|----------|------|
| CLIP文本编码器 | 123M | 文本序列 | 77×768 | 文本理解 |
| VAE编码器 | ~50M | 512×512×3 | 64×64×4 | 图像压缩 |
| U-Net | 860M | 64×64×4 | 64×64×4 | 噪声预测 |
| VAE解码器 | ~50M | 64×64×4 | 512×512×3 | 图像重建 |
| **总计** | **~1.08B** | - | - | **完整流程** |

## 🚀 推理性能特点

- **显存需求**: 最低10GB VRAM
- **推理速度**: ~2秒/图 (RTX 3090)
- **压缩比例**: 8×8×3 = 192倍压缩
- **采样步数**: 20-50步 (DDIM) / 50步 (PLMS)

这个网络架构图展示了Stable Diffusion的完整数据流和核心组件交互关系。