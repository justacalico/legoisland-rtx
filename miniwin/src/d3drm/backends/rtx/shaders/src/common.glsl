// Shared structures for the RTX backend shaders.
// Layouts must match the C++ structs in d3drmrenderer_rtx.h (std140/std430).

struct SceneLight {
	vec4 color; // rgb = light color, a unused
	vec4 position; // xyz = position (view-space origin handled CPU side), w = 1 if positional
	vec4 direction; // xyz = direction, w = 1 if directional
};

// Per-frame data (set 0, binding 0).
// D3DRMMATRIX4D data is uploaded verbatim and read as column-major mat4, which
// transparently converts the engine's row-vector convention to GLSL's
// column-vector convention.
layout(set = 0, binding = 0) uniform FrameUBO {
	mat4 proj;
	mat4 view;
	vec4 cameraPos; // xyz = world-space camera position
	vec4 ambient; // rgb = background/miss color, w = ambient scale
	vec4 rtParams; // x = rt effect flags, y = light count, z = front plane, w = back plane
	SceneLight lights[8];
} frame;

// Per-draw data (set 1, binding 0).
layout(set = 1, binding = 0) uniform DrawUBO {
	mat4 modelView;
	mat4 normal;
	mat4 world;
	vec4 color;
	vec4 params; // x = shininess, y = use texture, z = flat/unlit, w = unused
} draw;
