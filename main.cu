#include <assert.h>
#include <err.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <GL/glew.h>
// I only use one GLX extension, and I call it before my context is
// ready, so I can't use glxew for it.  Just use basic GLX and I'll
// get the extension myself instead of using glxew.
#include <GL/glx.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#define FPS 30
#define MAX_SHADER_LEN 65536
#define NREDS 1024

#define WIDTH 800
#define HEIGHT 800

static int buffer_attributes[] = {
    GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
    GLX_RENDER_TYPE,   GLX_RGBA_BIT,
    GLX_DOUBLEBUFFER,  True,  /* Request a double-buffered color buffer with */
    GLX_RED_SIZE,      1,     /* the maximum number of bits per component    */
    GLX_GREEN_SIZE,    1, 
    GLX_BLUE_SIZE,     1,
    None
};

struct resources
{
    Display *dpy;
    GLXWindow glxWin;
    GLXContext context;

    GLuint gl_uniform_buffer;
    GLuint gl_time_uniform_loc;
    GLuint gl_vao;
    GLuint gl_vertex_buffer;
    GLuint gl_element_buffer;
    GLuint gl_program;

    cudaGraphicsResource_t cuda_uniform_buffer;
};

struct reds_buffer
{
    struct {
        GLfloat red;
        GLfloat unused[3];
    } reds[NREDS];
};

static void
cuda_errchk_inner(const char* file, unsigned long line)
{
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess)
        return;
    const char *errstr = cudaGetErrorName(err);
    fprintf(stderr, "%s:%lu: CUDA error: %s\n", file, line, errstr);
    exit(EXIT_FAILURE);
}
#define CUDA_ERRCHK() cuda_errchk_inner(__FILE__, __LINE__)

static void
gl_errchk_inner(const char* file, unsigned long line)
{
    GLenum err = glGetError();
    if (err == GL_NO_ERROR)
        return;
    const GLubyte *errstr = gluErrorString(err);
    fprintf(stderr, "%s:%lu: OpenGL error: %s\n", file, line, errstr);
    exit(EXIT_FAILURE);
}
#define GL_ERRCHK() gl_errchk_inner(__FILE__, __LINE__)

static Bool
WaitForNotify(Display *dpy, XEvent *event, XPointer arg)
{
    return (event->type == MapNotify) && (event->xmap.window == (Window)arg);
}

static void
start_gl(struct resources *rsrc)
{
    // For much of this, see:
    // https://www.khronos.org/opengl/wiki/Programming_OpenGL_in_Linux:_GLX_and_Xlib
    // https://www.khronos.org/opengl/wiki/Tutorial:_OpenGL_3.0_Context_Creation_(GLX)

    Window xWin;
    XEvent event;
    XVisualInfo *vInfo;
    XSetWindowAttributes swa;
    GLXFBConfig *fbConfigs;
    int swaMask;
    int numReturned;

    /* Open a connection to the X server */
    rsrc->dpy = XOpenDisplay(NULL);
    if (rsrc->dpy == NULL) {
        fprintf(stderr, "Unable to open a connection to the X server\n");
        exit(EXIT_FAILURE);
    }

    /* Request a suitable framebuffer configuration - try for a double 
     * buffered configuration first */
    fbConfigs = glXChooseFBConfig(rsrc->dpy, DefaultScreen(rsrc->dpy),
                                  buffer_attributes, &numReturned);

    /* Create an X colormap and window with a visual matching the first
     * returned framebuffer config */
    vInfo = glXGetVisualFromFBConfig(rsrc->dpy, fbConfigs[0]);

    swa.border_pixel = 0;
    swa.event_mask = StructureNotifyMask | ButtonPressMask | KeyPressMask;
    swa.colormap = XCreateColormap(rsrc->dpy,
                                   RootWindow(rsrc->dpy, vInfo->screen),
                                   vInfo->visual, AllocNone);

    swaMask = CWBorderPixel | CWColormap | CWEventMask;

    xWin = XCreateWindow(rsrc->dpy, RootWindow(rsrc->dpy, vInfo->screen),
                         0, 0, WIDTH, HEIGHT,
                         0, vInfo->depth, InputOutput, vInfo->visual,
                         swaMask, &swa);
    XStoreName(rsrc->dpy, xWin, "Blending CUDA and OpenGL");

    /* Create a GLX context for OpenGL rendering */
    int context_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
        GLX_CONTEXT_MINOR_VERSION_ARB, 2,
        None
    };
    /* I can't initialize GLXEW yet because I don't have a context.
     * That means that I need to get the glXCreateContextAttribsARB
     * address myself. */
    typedef GLXContext (*glXCreateContextAttribsARBProc)
        (Display*, GLXFBConfig, GLXContext, Bool, const int*);
    glXCreateContextAttribsARBProc glXCreateContextAttribsARB =
        (glXCreateContextAttribsARBProc)glXGetProcAddressARB(
            (const GLubyte*)"glXCreateContextAttribsARB");
    rsrc->context = glXCreateContextAttribsARB(rsrc->dpy, fbConfigs[0], NULL,
                                               True, context_attribs);
    /*
      Alternative for getting an OpenGL 2 context:
    context = glXCreateNewContext(rsrc->dpy, fbConfigs[0], GLX_RGBA_TYPE,
                                  NULL, True);
    */

    /* Create a GLX window to associate the frame buffer configuration
     * with the created X window */
    rsrc->glxWin = glXCreateWindow(rsrc->dpy, fbConfigs[0], xWin, NULL);
    
    /* Map the window to the screen, and wait for it to appear */
    XMapWindow(rsrc->dpy, xWin);
    XIfEvent(rsrc->dpy, &event, WaitForNotify, (XPointer)xWin);

    /* Bind the GLX context to the Window */
    glXMakeContextCurrent(rsrc->dpy, rsrc->glxWin, rsrc->glxWin, rsrc->context);
    GL_ERRCHK();

    /* Initialize GLEW for extensions */
    glewExperimental = True;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "GLEW error: %s\n", glewGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // GLEW's probes can leave an error in the context, so clear it.
    glGetError();
    // This is kinda redundant here, but I'm leaving it.
    GL_ERRCHK();
}

static void
start_cuda(struct resources *rsrc)
{
    // Deprecated, no longer necessary
    //cudaGLSetGLDevice(0);
    CUDA_ERRCHK();
}

static void
initialize_cuda_resources(struct resources *rsrc)
{
    cudaGraphicsGLRegisterBuffer(&rsrc->cuda_uniform_buffer,
                                 rsrc->gl_uniform_buffer,
                                 cudaGraphicsRegisterFlagsWriteDiscard);
    CUDA_ERRCHK();
}

static void
read_shader(const char* filename, GLchar **shader_src, GLint *shader_len)
{
    *shader_src = new char[MAX_SHADER_LEN];
    if (*shader_src == NULL)
        err(EXIT_FAILURE, "malloc");
    int fd = open(filename, O_RDONLY);
    if (fd < 0)
        err(EXIT_FAILURE, "%s", filename);
    *shader_len = read(fd, *shader_src, MAX_SHADER_LEN);
    if (*shader_len < 0)
        err(EXIT_FAILURE, "%s", filename);
    int close_err = close(fd);
    if (close_err)
        err(EXIT_FAILURE, "%s", filename);
    if (*shader_len == MAX_SHADER_LEN)
        errx(EXIT_FAILURE, "%s: Shader too long; increase MAX_SHADER_LEN",
             filename);
}

static GLuint
compile_shader(const char* filename, GLenum type)
{
    GLchar *shader_src;
    GLint shader_len;
    read_shader(filename, &shader_src, &shader_len);
    GLchar *util_src;
    GLint util_len;
    read_shader("util.glsl", &util_src, &util_len);
    
    const GLchar* shader_srcs[2];
    GLint shader_lens[2];
    shader_srcs[0] = shader_src;
    shader_lens[0] = shader_len;
    shader_srcs[1] = util_src;
    shader_lens[1] = util_len;

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, shader_srcs, shader_lens);
    delete[] shader_src;
    delete[] util_src;
    GL_ERRCHK();
    glCompileShader(shader);

    GLint is_compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &is_compiled);
    GLint max_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &max_length);
    GLchar error_log[max_length];
    glGetShaderInfoLog(shader, max_length, &max_length, &error_log[0]);
    if(is_compiled == GL_FALSE) {
        fprintf(stderr, "%s: Shader compile error:\n%s", filename, error_log);
        exit(EXIT_FAILURE);
    } else if (max_length) {
        fprintf(stderr, "%s: Shader compile messages:\n%s", filename, error_log);
    }
    
    GL_ERRCHK();
    return shader;
}

static void
initialize_gl_resources(struct resources *rsrc)
{
    /*
     * Allocate buffers
     */

    // Uniform buffer
    glGenBuffers(1, &rsrc->gl_uniform_buffer);
    // The buffer we're using would be more appropriate as a 1d
    // texture.  However, in practice, uniform buffers are more likely
    // to be used for CUDA-OpenGL interface, so I'm using that to make
    // better demo code.
    glBindBuffer(GL_UNIFORM_BUFFER, rsrc->gl_uniform_buffer);
    // This actually allocates the storage for the buffer.  The last
    // parameter determines where it will be allocated.  See also
    // https://www.khronos.org/opengl/wiki/Buffer_Object
    glBufferData(GL_UNIFORM_BUFFER, sizeof(struct reds_buffer),
                 NULL, GL_STREAM_DRAW);
    GL_ERRCHK();

    // For the following vertex-related stuff, see
    // https://www.khronos.org/opengl/wiki/Vertex_Specification

    // Vertex array object
    // This object holds all of the vertex state information that
    // we're about to set up.  We only need one for our program, so
    // we just create it and bind it.
    glGenVertexArrays(1, &rsrc->gl_vao);
    glBindVertexArray(rsrc->gl_vao);
    GL_ERRCHK();

    // Vertex buffer object
    // This holds the information that we'll pass for each vertex.
    // We'll pass the position and the blue channel.
    //
    // First, assign internal identifiers to each attribute.  We can
    // make these up; we'll assign them to actual variable names in
    // the shader later.
    const int position_attr = 0;
    const int blue_attr = 1;
    static struct vertex {
        // position is a vec3.
        GLfloat position[3];
        // blue is a unsigned byte.  We have the GPU convert it to a
        // float during the upload, so that the shader can use the
        // float-optimized hardware.  (This is a ridiculous way to handle
        // this in our case, but I'm just demonstrating float normalization
        // in VBOs.)
        GLubyte blue;
    } vertices[4] = {
        // This array shows all the vertices we'll use in this program.
        // We'll talk about the order in which they're used in the main
        // loop, but for now, note that these are conveniently arranged
        // in clockwise order starting with quadrant I.
        {{  1.0,  1.0, 0.0 }, 0},
        {{ -1.0,  1.0, 0.0 }, 255},
        {{ -1.0, -1.0, 0.0 }, 0},
        {{  1.0, -1.0, 0.0 }, 255},
    };
    // Create, bind, and populate the buffer holding this data.  We won't
    // ever change it, so use GL_STATIC_DRAW.
    glGenBuffers(1, &rsrc->gl_vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, rsrc->gl_vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    GL_ERRCHK();
    // Set each vertex attribute's location, so that GL knows which
    // part of the vertex to send to each variable.  (We just assign
    // these to numbers now; we'll connect those numbers to names
    // below when we compile the shaders.)
    // 
    // There are two ways to do this: one is with glVertexAttribPointer
    // (which is always available), or with glVertexAttribFormat and
    // friends (which requires the extension ARB_vertex_attrib_binding,
    // which is available in most cards supporting 3.3 and later).
    // We'll demonstrate both, although in practice you'd only use one
    // depending on your needs.
    // (You can use "0&&" etc to fiddle around with these.)
    if (GLEW_ARB_vertex_attrib_binding) {
        const int vbo_idx = 0;  // We only use one VBO; call it #0
        glBindVertexBuffer(vbo_idx, rsrc->gl_vertex_buffer, 
                           0, sizeof(struct vertex));
        glVertexAttribFormat(position_attr, 3, GL_FLOAT, GL_FALSE,
                             offsetof(struct vertex, position));
        glVertexAttribBinding(position_attr, vbo_idx);
        glVertexAttribFormat(blue_attr, 1, GL_UNSIGNED_BYTE, GL_TRUE,
                             offsetof(struct vertex, blue));
        glVertexAttribBinding(blue_attr, vbo_idx);
    } else {
        fprintf(stderr, "Huh, I'm not using ARB_vertex_attrib_binding.\n");
        glVertexAttribPointer(position_attr, 3, GL_FLOAT, GL_FALSE,
                              sizeof(struct vertex),
                              reinterpret_cast<void*>(
                                  offsetof(struct vertex, position)));
        glVertexAttribPointer(blue_attr, 1, GL_UNSIGNED_BYTE, GL_TRUE,
                              sizeof(struct vertex),
                              reinterpret_cast<void*>(
                                  offsetof(struct vertex, blue)));
    }
    glEnableVertexAttribArray(position_attr);
    glEnableVertexAttribArray(blue_attr);
    GL_ERRCHK();

    // Element Array
    //
    // This is an array that says which order we want to draw our
    // vertices in.  It's not necessary; we could put all our vertices
    // in the "vertices" array in the order desired, and use
    // glDrawArrays directly.  We're doing it this way because some of
    // our vertices are duplicates, so instead of uploading 67% more
    // vertices, we just send a list of the indices.
    //
    // Since the main loop is changing the order of the vertices,
    // we'll just set up the object, and let the main loop upload the
    // indices.
    glGenBuffers(1, &rsrc->gl_element_buffer);
    GL_ERRCHK();

    /*
     * Compile shaders
     */

    GLuint vertex_shader = compile_shader("vertex.glsl", GL_VERTEX_SHADER);
    GLuint fragment_shader =
        compile_shader("fragment.glsl", GL_FRAGMENT_SHADER);
    
    rsrc->gl_program = glCreateProgram();
    glAttachShader(rsrc->gl_program, vertex_shader);
    glAttachShader(rsrc->gl_program, fragment_shader);
    GL_ERRCHK();
    // Now we actually bind our attributes, which we assigned internal
    // numbers to earlier, to their locations in the shaders.
    glBindAttribLocation(rsrc->gl_program, position_attr, "position");
    glBindAttribLocation(rsrc->gl_program, blue_attr, "blue");
    GL_ERRCHK();
    glLinkProgram(rsrc->gl_program);
    
    GLint is_linked = 0;
    glGetProgramiv(rsrc->gl_program, GL_LINK_STATUS, &is_linked);
    GLint max_length = 0;
    glGetProgramiv(rsrc->gl_program, GL_INFO_LOG_LENGTH, &max_length);
    GLchar error_log[max_length];
    glGetProgramInfoLog(rsrc->gl_program, max_length, &max_length, error_log);
    if (is_linked == GL_FALSE) {
        fprintf(stderr, "Shader link error:\n%s", error_log);
        exit(EXIT_FAILURE);
    } else if (max_length) {
        fprintf(stderr, "Shader link messages:\n%s", error_log);
    }

    glDetachShader(rsrc->gl_program, vertex_shader);
    glDeleteShader(vertex_shader);
    glDetachShader(rsrc->gl_program, fragment_shader);
    glDeleteShader(fragment_shader);
    GL_ERRCHK();

    /*
     * Set up uniforms
     */

    // Set up the uniform block object, which is our "reds" array.
    // Get the index of the "reds" block.
    GLuint uniform_idx = glGetUniformBlockIndex(rsrc->gl_program, "reds_block");
    assert(uniform_idx != GL_INVALID_INDEX);
    // Set that up as uniform buffer #0.
    glUniformBlockBinding(rsrc->gl_program, uniform_idx, 0);
    // Bind it to our previously-created uniform buffer.
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, rsrc->gl_uniform_buffer);
    GL_ERRCHK();

    // Set up the "time" uniform.
    rsrc->gl_time_uniform_loc =
        glGetUniformLocation(rsrc->gl_program, "time");
}

static __global__ void
calculate_reds_kernel(struct reds_buffer* reds_block,
                      unsigned long long time)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < NREDS) {
        float theta = sinf(float(time) / 8.0) + float(thread_id) / 64.0;
        reds_block->reds[thread_id].red = pow(sinf(theta), 2);
    }
}

static void
calculate_reds(struct resources *rsrc, unsigned long long time)
{
    // Map the uniform buffer into CUDA space so the kernel can work on it.
    cudaGraphicsMapResources(1, &rsrc->cuda_uniform_buffer);
    CUDA_ERRCHK();
    // Get a CUDA-accessible pointer to the mapped buffer.
    struct reds_buffer* devptr;
    size_t devptr_size;
    cudaGraphicsResourceGetMappedPointer((void**)&devptr, &devptr_size,
                                         rsrc->cuda_uniform_buffer);
    CUDA_ERRCHK();
    assert(devptr_size == sizeof(struct reds_buffer));
    // Launch the kernel.
    calculate_reds_kernel<<<16, NREDS / 16>>>(devptr, time);
    CUDA_ERRCHK();
    // Unmap the uniform buffer so it's available to OpenGL again.  (This
    // includes an implicit sync point.)
    cudaGraphicsUnmapResources(1, &rsrc->cuda_uniform_buffer);
    CUDA_ERRCHK();
}

static void
load_elements(struct resources *rsrc, const GLuint* vertices,
              size_t vertices_size)
{
    glXMakeContextCurrent(rsrc->dpy, rsrc->glxWin, rsrc->glxWin, rsrc->context);
    glBindVertexArray(rsrc->gl_vao);
    glBindBuffer(GL_ARRAY_BUFFER, rsrc->gl_vertex_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rsrc->gl_element_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, vertices_size, vertices,
                 GL_DYNAMIC_DRAW);
    GL_ERRCHK();
}

static void
draw_frame(struct resources *rsrc, GLsizei nvertices, unsigned long long time)
{
    // Activate our context, shaders, and VAO.  (Not technically
    // necessary here, since they've been activated all along, but
    // it's always prudent to refresh the context on each drawing in a
    // big program.)
    glXMakeContextCurrent(rsrc->dpy, rsrc->glxWin, rsrc->glxWin, rsrc->context);
    glUseProgram(rsrc->gl_program);
    glBindVertexArray(rsrc->gl_vao);

    // Set the time uniform.
    glUniform1f(rsrc->gl_time_uniform_loc, float(time) / FPS);
    GL_ERRCHK();

    // Start drawing
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElements(GL_TRIANGLE_STRIP, nvertices, GL_UNSIGNED_INT, 0);
    glFlush();
    glXSwapBuffers(rsrc->dpy, rsrc->glxWin);
    GL_ERRCHK();
}

static Bool
is_quit_event(Display *dpy, XEvent *evt, XPointer arg)
{
    if (evt->type == KeyPress) {
        KeySym ks = XLookupKeysym(&evt->xkey, 0);
        return !IsModifierKey(ks);
    }
    if (evt->type == ButtonPress)
        return True;
    return False;
}

static void
check_input(struct resources *rsrc)
{
    XEvent evt;
    if (XCheckIfEvent(rsrc->dpy, &evt, is_quit_event, NULL))
        exit(EXIT_SUCCESS);
}

int
main(void)
{
    struct resources rsrc;
    start_gl(&rsrc);
    start_cuda(&rsrc);
    initialize_gl_resources(&rsrc);
    initialize_cuda_resources(&rsrc);

#if 0
    // This is an example of drawing under OpenGL 1 or 2.  This uses
    // the OpenGL built-in matrix stuff, and individual calls to
    // primitives.  The built-in matrices are not part of the core
    // profile in OpenGL 3.1 and above, but they're generally
    // available in the compatibility profile.  However, we've asked
    // for core profile, so this stuff isn't available.

    // Set up which portion of the window is being used
    glViewport(0, 0, WIDTH, HEIGHT);
    // Just set up an orthogonal system
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1.0f, 0, 1.0f, -1.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_DEPTH_TEST);
    glClearColor(1.0f, 1.0f, 1.0f, 1.5f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // Here's where you'd typically put drawing commands.
    glFlush();
    glXSwapBuffers(dpy, glxWin);
    GL_ERRCHK();
#endif

    // Keep a frame counter
    unsigned long long time = 0;

    // These are the triangles that we'll render at each stage of the loop.
    // Note that we always arrange these counterclockwise, so that the
    // front of the triangle is facing us.
    static GLuint triangle_indices[][3] = {
        { 0, 1, 2 },
        { 1, 2, 3 },
        { 2, 3, 0 },
        { 3, 0, 1 }
    };
    for (int i = 0; i < 4; i++) {
        load_elements(&rsrc, triangle_indices[i], sizeof(triangle_indices[i]));
        for (int j = 0; j < FPS; j++) {
            check_input(&rsrc);
            calculate_reds(&rsrc, time);
            draw_frame(&rsrc, 3, time);
            time++;
            usleep(1000000 / FPS);
        }
    }

    // This is the triangle strip we'll render at the end of the loop.
    // Note that we need to pick the order to correctly draw the strip.
    static GLuint quad_indices[] = { 0, 1, 3, 2 };
    load_elements(&rsrc, quad_indices, sizeof(quad_indices));
    while (1) {
        check_input(&rsrc);
        calculate_reds(&rsrc, time);
        draw_frame(&rsrc, 4, time);
        time++;
        usleep(1000000 / FPS);
    }
}

/*
 * Local Variables:
 * mode: c++
 * compile-command: "/usr/local/cuda/bin/nvcc -g -O -Xcompiler=-Wall -o main -lGL -lGLU -lGLEW -lX11 main.cu && optirun ./main"
 * End:
 */
