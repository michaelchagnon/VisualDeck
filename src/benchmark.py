# =====================================================================
# src/benchmark.py
# =====================================================================

import os
# MODIFICATION: Add this line to hide the Pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import sys
import time
import numpy as np
from PIL import Image
import pygame
from pygame.locals import *
import ctypes

# Add minimal OpenGL imports to avoid heavy dependencies
try:
    from OpenGL.GL import *
    from OpenGL.GL import shaders
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

# --- Constants copied from render_engine.py ---
VERTEX_SHADER = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aTexCoord;
out vec2 TexCoord;
uniform mat4 model;
uniform mat4 projection;
void main() {
    gl_Position = projection * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""
FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D ourTexture;
uniform float alpha;
void main() {
    vec4 texColor = texture(ourTexture, TexCoord);
    if(texColor.a < 0.1) discard;
    FragColor = vec4(texColor.rgb, texColor.a * alpha);
}
"""

def _ortho(l, r, b, t, n, f):
    return np.array([
        [2/(r-l), 0, 0, -(r+l)/(r-l)],
        [0, 2/(t-b), 0, -(t+b)/(t-b)],
        [0, 0, -2/(f-n), -(f+n)/(f-n)],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def _model_matrix(w, h):
    S = np.eye(4, dtype=np.float32)
    S[0,0] = w
    S[1,1] = h
    T = np.eye(4, dtype=np.float32)
    T[0,3] = 50 # center in 100x100 window
    T[1,3] = 50
    return T @ S

def benchmark_opengl():
    if not OPENGL_AVAILABLE:
        return float('inf')
    try:
        pygame.init()
        pygame.display.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        os.environ['SDL_VIDEO_WINDOW_POS'] = "-1000,-1000"
        pygame.display.set_mode((100, 100), NOFRAME | DOUBLEBUF | OPENGL)

        # --- Simplified GL setup ---
        shader = shaders.compileProgram(
            shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
        glUseProgram(shader)
        u_projection = glGetUniformLocation(shader, "projection")
        u_model = glGetUniformLocation(shader, "model")
        u_alpha = glGetUniformLocation(shader, "alpha")

        quad = np.array([0.5,0.5,0.0,1.0,0.0, 0.5,-0.5,0.0,1.0,1.0, -0.5,-0.5,0.0,0.0,1.0, -0.5,0.5,0.0,0.0,0.0], dtype=np.float32)
        indices = np.array([0,1,3,1,2,3], dtype=np.uint32)
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        projection_matrix = _ortho(0, 100, 0, 100, -1, 1)
        glUniformMatrix4fv(u_projection, 1, GL_TRUE, projection_matrix)

        # --- Texture setup ---
        img = Image.new('RGBA', (50, 50), 'blue')
        img_data = np.array(img).tobytes()
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 50, 50, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

        model_matrix = _model_matrix(50, 50)
        glUniformMatrix4fv(u_model, 1, GL_TRUE, model_matrix)
        glUniform1f(u_alpha, 1.0)

        # --- Benchmark loop ---
        start_time = time.perf_counter()
        for _ in range(100):
            glClear(GL_COLOR_BUFFER_BIT)
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
            pygame.display.flip()
        end_time = time.perf_counter()

        pygame.quit()
        return end_time - start_time
    except Exception:
        if pygame.get_init():
            pygame.quit()
        return float('inf')

def benchmark_pygame():
    try:
        pygame.init()
        pygame.display.init()
        os.environ['SDL_VIDEO_WINDOW_POS'] = "-1000,-1000"
        window = pygame.display.set_mode((100, 100), NOFRAME | DOUBLEBUF)
        
        img = Image.new('RGBA', (50, 50), 'blue')
        img_bytes = img.tobytes()
        surface = pygame.image.frombuffer(img_bytes, (50, 50), "RGBA")
        
        start_time = time.perf_counter()
        for _ in range(100):
            window.fill((0,0,0))
            window.blit(surface, (25, 25))
            pygame.display.flip()
        end_time = time.perf_counter()

        pygame.quit()
        return end_time - start_time
    except Exception:
        if pygame.get_init():
            pygame.quit()
        return float('inf')

if __name__ == "__main__":
    # This block runs when the script is executed directly
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    result = float('inf')
    if mode == 'opengl':
        result = benchmark_opengl()
    elif mode == 'pygame_gpu':
        result = benchmark_pygame()
    
    # Print the final time so the main app can read it
    print(result)
