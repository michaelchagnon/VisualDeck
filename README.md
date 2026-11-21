> **Archived version.** This is the source of VisualDeck 1.0b (August 2025), kept for reference and now licensed under the [Mozilla Public License 2.0](LICENSE). The current release is on the [main branch](https://github.com/michaelchagnon/VisualDeck) and in [Releases](https://github.com/michaelchagnon/VisualDeck/releases/latest).

# 🎬 VisualDeck - High-Performance Python VJ Engine

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0b-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.12+-green.svg" alt="Python">
</p>

<p align="center">
  <img src="https://visualdeck.org/visualdeck.png" alt="Visual Deck"><br><br>
  <img src="https://visualdeck.org/bitcoin.png" alt="Donate Bitcoin: bc1qv6aw43faucxyvne7z4hr7yystxu8vu5sf5fc3m"><br>
  Donate Bitcoin: bc1qv6aw43faucxyvne7z4hr7yystxu8vu5sf5fc3m
</p>

## 🎯 What is VisualDeck?

VisualDeck is a powerful, real-time video mixing and presentation software that bridges the gap between **PowerPoint's simplicity** and **Resolume's professional VJ capabilities**. Think of it as your Swiss Army knife for live visual performances, presentations, and video installations.

### 🌟 Perfect For:
- 🎪 **Live Events** - Theater productions, concerts, and performances
- 🏢 **Corporate Presentations** - Dynamic multi-screen presentations
- 🎨 **Art Installations** - Interactive video displays and exhibitions
- 🎮 **Gaming Events** - Tournament displays and live streaming backgrounds
- 🏛️ **Museums** - Multi-screen educational displays

## ✨ Key Features

### 🎛️ Core Capabilities
- **📊 Grid-Based Interface** - Intuitive cue/layer system like a lighting console
- **🖥️ Multi-Display Support** - Span content across multiple monitors/projectors
- **⚡ Real-Time Performance** - Hardware-accelerated video playback
- **🔄 Smooth Transitions** - Professional fade effects between cues
- **🎬 Live Preview** - See your output before it goes live

### 🚀 Advanced Features
- **🎯 Layer-Specific Routing** - Send different layers to different displays
- **🔧 Media Positioning** - Scale, rotate, and position content precisely
- **🎪 Presentation Mode** - Automated playback with timing control
- **💾 Project Management** - Save and load complete show files
- **🔍 Proxy Generation** - Automatic low-res previews for smooth editing

### 🛠️ Technical Highlights
- **OpenGL Rendering** - GPU-accelerated compositing
- **Multi-Threading** - Separate render and UI threads
- **Hardware Decoding** - CUDA, DXVA2, D3D11VA support
- **Flexible Architecture** - CPU-only fallback mode

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, untested on Linux (Ubuntu 20.04+) but may work
- **CPU**: Intel i5 or AMD Ryzen 5
- **RAM**: 8GB
- **GPU**: DirectX 11 compatible (for GPU acceleration)
- **Storage**: 2GB free space

### Recommended Specifications
- **CPU**: Intel i7/i9 or AMD Ryzen 7/9
- **RAM**: 16GB or more
- **GPU**: NVIDIA RTX 3060 or better
- **Storage**: SSD with 10GB+ free space

## 🔧 Installation Guide

### Step 1: Install Python 🐍
1. Download Python 3.7 or newer from [python.org](https://python.org)
2. During installation, **CHECK** "Add Python to PATH"
3. Verify installation:
   ```bash
   python --version
   ```

### Step 2: Install Visual C++ Redistributables (Windows Only) 🪟
1. Download from [Microsoft's official page](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
2. Install both x64 and x86 versions
3. Restart your computer

### Step 3: Download VisualDeck 📦
```bash
git clone https://github.com/michaelchagnon/visualdeck.git
cd visualdeck
```

### Step 4: Create Virtual Environment 🌍
```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 5: Install Dependencies 📚
```bash
pip install -r requirements.txt
```

### Step 6: Launch VisualDeck 🚀
```bash
python src/app.py
```

## 🎮 How to Use VisualDeck

### 🏁 Getting Started

1. **Launch the Application**
   - Run `python src/app.py`
   - The interface opens with three panels: Media Library, Grid, and Preview

2. **Understanding the Interface**
   ```
   ┌─────────────┬──────────────────┬──────────────┐
   │   Media     │    Grid/Cues     │   Preview    │
   │   Library   │                  │              │
   │             │  ┌─┬─┬─┬─┬─┬─┐   │  [Output]    │
   │  [Videos]   │  ├─┼─┼─┼─┼─┼─┤   │              │
   │  [Images]   │  ├─┼─┼─┼─┼─┼─┤   │  [Screen]    │
   │             │  └─┴─┴─┴─┴─┴─┘   │              │
   │             │                  │  [GO Button] │
   └─────────────┴──────────────────┴──────────────┘
   ```

### 📥 Adding Media

1. **Import Media Files**
   - Click `Insert → Add Media to Project...`
   - Select videos (MP4, MOV, AVI) or images (JPG, PNG)
   - Files appear in the left Media Library

2. **Drag & Drop to Grid**
   - Drag media from library to grid cells
   - Each column = one cue (like a PowerPoint slide)
   - Each row = one layer (stacked top to bottom)

### 🎬 Creating Your First Show

1. **Build Cues**
   ```
   Cue 1: Welcome Screen
   - Layer 1: Background video
   - Layer 2: Logo overlay
   
   Cue 2: Main Content
   - Layer 1: Presentation slides
   ```

2. **Configure Output**
   - Select output display from dropdown
   - Choose "Multiple" for multi-screen setups
   - Click layer settings (⚙️) to assign displays

3. **Edit Media Position**
   - Double-click any grid cell to open editor
   - Adjust position, scale, and rotation
   - Use presets: Center, Fit, Fill

4. **Set Transitions**
   - Choose transition type (None/Fade)
   - Set duration in seconds
   - Smooth fades between cues!

### 🎪 Running Your Show

1. **Manual Control**
   - Click column headers to jump to cues
   - Press "GO" button or Spacebar for next cue
   - Monitor preview while performing

2. **Presentation Mode**
   - Enable "Presentation Mode"
   - Set timing between cues
   - Optional loop for installations
   - Press "GO" to start/stop automation

## 💡 Best Practices

### 🎥 Media Optimization

1. **Video Formats**
   - Use H.264 MP4 for compatibility
   - Consider HAP codec for best performance
   - Keep resolution ≤ 1920x1080 for smooth playback

2. **File Management**
   - Store media on fast SSD drives
   - Keep project files with media
   - Use consistent naming conventions

### 🏗️ Show Design

1. **Layer Organization**
   ```
   Layer 1: Background/Environment
   Layer 2: Main Content
   Layer 3: Overlays/Graphics
   Layer 4: Effects/Particles
   ```

2. **Cue Structure**
   - Group related content in adjacent cues
   - Use empty cues for blackouts
   - Name cues descriptively

### ⚡ Performance Tips

1. **Rendering Modes**
   - **OpenGL**: Best performance, use when possible
   - **Pygame+GPU**: Good compatibility
   - **CPU Only**: Fallback for troubleshooting

2. **System Optimization**
   - Close unnecessary applications
   - Disable Windows Game Mode
   - Use wired connections for external displays

## 🛠️ Advanced Configuration

### 🖥️ Multi-Display Setup

1. **Configure Displays**
   ```
   Display 0: Control monitor (UI)
   Display 1: Main projection
   Display 2: Side screens
   Display 3: Confidence monitor
   ```

2. **Layer Routing**
   - Click layer settings (⚙️)
   - Assign each layer to specific displays
   - Different content on each screen!

### 🎨 Custom Workflows

1. **Theater Production**
   - Pre-show: Loop ambient visuals
   - Act 1: Cue 1-15 with scene backgrounds
   - Intermission: Logo/sponsor loop
   - Act 2: Cue 16-30 with effects

2. **Corporate Event**
   - Walk-in: Branded motion graphics
   - Presentation: Slide cues
   - Awards: Winner videos
   - Walk-out: Sponsor credits

## 🐛 Troubleshooting

### Common Issues

1. **"No video playback"**
   - Switch to CPU rendering mode
   - Check video codec compatibility
   - Verify file paths are correct

2. **"Lag during playback"**
   - Reduce video resolution
   - Enable hardware acceleration
   - Generate proxies for editing

3. **"Black output screen"**
   - Check display settings
   - Verify output selection
   - Restart application

### 📊 Performance Monitoring

- Watch the log file: `~/.visualdeck/app.log`
- Monitor GPU usage in Task Manager
- Check render thread status

## 🚀 Future Roadmap

### Coming Soon:
- **🎹 OSC/MIDI Control** - Integration with lighting consoles and controllers
- **🔲 Advanced Projection Mapping** - Stretch and warp media for irregular surfaces
- **🖼️ Image Optimization** - Automatic image compression and format conversion
- **🔊 Audio Playback** - Synchronized audio with video cues
- **📹 Live Inputs** - Webcam and capture card support
- **📡 Streaming Sources** - RTMP, NDI, and network stream inputs

## 🤝 Contributing

Contributing guide coming soon! We welcome all contributions to make VisualDeck even better.

---

<p align="center">
  <strong>🎬 Ready to create amazing visual experiences? Download VisualDeck and start mixing! 🚀</strong>
</p>

<p align="center">
  <a href="https://visualdeck.org/">VisualDeck.org</a>
</p>
